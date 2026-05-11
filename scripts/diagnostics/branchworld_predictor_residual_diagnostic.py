import argparse
import csv
import os
import random
import shlex
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.data_factory import data_provider
from models.BranchWorldModel import Model


class ArgumentParserWithFiles(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return shlex.split(arg_line, comments=True)


def build_parser():
    parser = ArgumentParserWithFiles(
        description="Diagnose BranchWorld predictor argmax and residual direction errors.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="predictor_residual_diag")
    parser.add_argument("--model", type=str, default="BranchWorldModel")

    parser.add_argument("--data", type=str, default="ETTh1")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--augmentation_ratio", type=int, default=0)

    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--factor", type=int, default=3)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--no_use_gpu", action="store_false", dest="use_gpu")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--wm_latent_dim", type=int, default=128)
    parser.add_argument("--wm_memory_size", type=int, default=4096)
    parser.add_argument("--wm_global_proto_num", type=int, default=32)
    parser.add_argument("--wm_kmeans_iters", type=int, default=8)
    parser.add_argument("--wm_horizons", type=int, nargs="*", default=None)
    parser.add_argument("--wm_use_memory", type=int, choices=[0, 1], default=1)
    parser.add_argument("--wm_attention_mode", type=str, choices=["dot", "linear", "oracle"], default="linear")
    parser.add_argument(
        "--wm_backbone",
        type=str,
        default="temporal_transformer",
        choices=["temporal_transformer", "patch_transformer", "inverted_transformer", "tcn", "mlp"],
    )
    parser.add_argument("--wm_patch_len", type=int, default=16)
    parser.add_argument("--wm_patch_stride", type=int, default=8)
    parser.add_argument("--wm_base_type", type=str, choices=["linear", "dlinear"], default="dlinear")
    parser.add_argument("--wm_base_model", type=str, default="dlinear")
    parser.add_argument("--wm_base_checkpoint", type=str, default="")
    parser.add_argument("--wm_freeze_base", type=int, choices=[0, 1], default=1)
    parser.add_argument("--wm_freeze_encoder", type=int, choices=[0, 1], default=1)
    parser.add_argument("--wm_correction_lambda", type=float, default=0.1)
    parser.add_argument("--wm_branch_temperature", type=float, default=1.0)
    parser.add_argument("--wm_delta_clamp", type=float, default=3.0)
    parser.add_argument("--wm_predictor_loss", type=str, choices=["ce", "mse"], default="mse")
    parser.add_argument("--wm_predictor_checkpoint", type=str, default="")
    parser.add_argument("--wm_predictor_ce_weight", type=float, default=1.0)
    parser.add_argument("--wm_predictor_ce_only", type=int, choices=[0, 1], default=0)
    parser.add_argument("--wm_predictor_entropy_weight", type=float, default=0.0)
    parser.add_argument("--wm_predictor_context", type=str, choices=["z0", "summary"], default="z0")
    parser.add_argument("--wm_predictor_type", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--wm_predictor_hidden_dim", type=int, default=128)
    parser.add_argument("--wm_residual_encoder", type=str, choices=["identity", "linear", "mlp"], default="identity")
    parser.add_argument("--wm_residual_latent_dim", type=int, default=128)
    parser.add_argument("--wm_residual_hidden_dim", type=int, default=128)
    parser.add_argument("--wm_hard_selection", type=int, choices=[0, 1], default=0)

    parser.add_argument("--diag_split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--diag_output_csv", type=str, default="")
    parser.add_argument("--seed", type=int, default=2021)
    return parser


def make_loader(args, flag):
    _, loader = data_provider(args, flag)
    return DataLoader(
        loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )


def unpack_batch(batch):
    if len(batch) == 5:
        batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
    else:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
    return batch_x, batch_y, batch_x_mark, batch_y_mark


def denorm(model, y, means, stdev):
    return model._denormalize(y, means, stdev)


@torch.no_grad()
def diagnose(model, loader, device):
    model.eval()
    all_cos = []
    all_hit = []
    all_pred = []
    all_oracle = []
    mse_sums = {"base": 0.0, "soft": 0.0, "hard": 0.0, "oracle": 0.0}
    mae_sums = {"base": 0.0, "soft": 0.0, "hard": 0.0, "oracle": 0.0}
    value_count = 0

    for batch in loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = unpack_batch(batch)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        future_y = batch_y[:, -model.pred_len :, :]
        if future_y.size(-1) != model.c_out:
            future_y = future_y[:, :, -model.c_out :]

        norm_x, means, stdev = model._normalize(batch_x)
        norm_future = (future_y - means[:, :, -future_y.size(-1) :]) / stdev[:, :, -future_y.size(-1) :]
        dec_inp = torch.zeros_like(batch_y[:, -model.pred_len :, :]).float()
        dec_inp = torch.cat([batch_y[:, : model.label_len, :], dec_inp], dim=1)

        y_base = model._base_forecast(
            norm_x,
            x_enc=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
            means=means,
            stdev=stdev,
        )
        _, key, trajectory = model._encode_trajectory(norm_x, norm_future)
        oracle_label = model._assign_cluster_labels(trajectory)
        predictor_features = model._predictor_features(key, norm_x, y_base)
        predictor_features = model.predictor_norm(predictor_features)
        logits = model.cluster_predictor(predictor_features)[:, : model.prototype_residuals.size(0)]
        weights = torch.softmax(logits / model.temperature, dim=-1)
        pred_label = weights.argmax(dim=1)

        r_pred = model.prototype_residuals[pred_label]
        r_oracle = model.prototype_residuals[oracle_label]
        r_soft = torch.einsum("bp,plc->blc", weights, model.prototype_residuals)

        cos = F.cosine_similarity(r_pred.flatten(start_dim=1), r_oracle.flatten(start_dim=1), dim=1)
        hit = pred_label == oracle_label
        all_cos.append(cos.detach().cpu())
        all_hit.append(hit.detach().cpu())
        all_pred.append(pred_label.detach().cpu())
        all_oracle.append(oracle_label.detach().cpu())

        preds = {
            "base": y_base,
            "soft": y_base + model.correction_lambda * r_soft,
            "hard": y_base + model.correction_lambda * r_pred,
            "oracle": y_base + model.correction_lambda * r_oracle,
        }
        true_raw = future_y
        n = true_raw.numel()
        value_count += n
        for name, pred_norm in preds.items():
            pred_raw = denorm(model, pred_norm, means, stdev)
            mse_sums[name] += (pred_raw - true_raw).pow(2).sum().item()
            mae_sums[name] += (pred_raw - true_raw).abs().sum().item()

    cos = torch.cat(all_cos)
    hit = torch.cat(all_hit)
    pred_label = torch.cat(all_pred)
    oracle_label = torch.cat(all_oracle)
    mismatched_cos = cos[~hit]

    quantile_points = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    quantiles = torch.quantile(cos, quantile_points)
    if mismatched_cos.numel() > 0:
        mismatch_quantiles = torch.quantile(mismatched_cos, quantile_points)
    else:
        mismatch_quantiles = torch.full_like(quantiles, float("nan"))

    bins = torch.linspace(-1.0, 1.0, 21)
    hist = torch.histc(cos, bins=20, min=-1.0, max=1.0)
    mismatch_hist = torch.histc(mismatched_cos, bins=20, min=-1.0, max=1.0) if mismatched_cos.numel() > 0 else torch.zeros(20)

    return {
        "count": int(cos.numel()),
        "hit_rate": float(hit.float().mean()),
        "negative_cos_rate": float((cos < 0).float().mean()),
        "mismatch_negative_cos_rate": float((mismatched_cos < 0).float().mean()) if mismatched_cos.numel() > 0 else float("nan"),
        "cos_mean": float(cos.mean()),
        "cos_std": float(cos.std(unbiased=False)),
        "mismatch_cos_mean": float(mismatched_cos.mean()) if mismatched_cos.numel() > 0 else float("nan"),
        "mismatch_count": int(mismatched_cos.numel()),
        "quantile_points": quantile_points.tolist(),
        "cos_quantiles": quantiles.tolist(),
        "mismatch_cos_quantiles": mismatch_quantiles.tolist(),
        "hist_edges": bins.tolist(),
        "hist_counts": hist.long().tolist(),
        "mismatch_hist_counts": mismatch_hist.long().tolist(),
        "mse": {k: v / value_count for k, v in mse_sums.items()},
        "mae": {k: v / value_count for k, v in mae_sums.items()},
        "pred_label": pred_label,
        "oracle_label": oracle_label,
        "cos": cos,
        "hit": hit,
    }


def write_csv(path, result):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["type", "name", "value"])
        for key in [
            "count",
            "hit_rate",
            "negative_cos_rate",
            "mismatch_negative_cos_rate",
            "cos_mean",
            "cos_std",
            "mismatch_cos_mean",
            "mismatch_count",
        ]:
            writer.writerow(["summary", key, result[key]])
        for name, value in result["mse"].items():
            writer.writerow(["mse", name, value])
        for name, value in result["mae"].items():
            writer.writerow(["mae", name, value])
        for q, value, mismatch_value in zip(
            result["quantile_points"],
            result["cos_quantiles"],
            result["mismatch_cos_quantiles"],
        ):
            writer.writerow(["cos_quantile", q, value])
            writer.writerow(["mismatch_cos_quantile", q, mismatch_value])
        edges = result["hist_edges"]
        for i, count in enumerate(result["hist_counts"]):
            writer.writerow(["cos_hist", "{:.2f}:{:.2f}".format(edges[i], edges[i + 1]), count])
        for i, count in enumerate(result["mismatch_hist_counts"]):
            writer.writerow(["mismatch_cos_hist", "{:.2f}:{:.2f}".format(edges[i], edges[i + 1]), count])


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unused args:", " ".join(unknown))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    train_loader = make_loader(args, "train")
    eval_loader = make_loader(args, args.diag_split)

    model = Model(args).to(device)
    model.build_memory(train_loader, device)
    result = diagnose(model, eval_loader, device)

    print("Split:", args.diag_split)
    print("Samples:", result["count"])
    print("Prototype count:", int(model.prototype_residuals.size(0)))
    print("Predictor context:", args.wm_predictor_context)
    print("Predictor type:", args.wm_predictor_type)
    print("Residual encoder:", args.wm_residual_encoder)
    print("Argmax hit rate: {:.6f}".format(result["hit_rate"]))
    print("Cosine mean/std: {:.6f}/{:.6f}".format(result["cos_mean"], result["cos_std"]))
    print("Negative cosine rate: {:.6f}".format(result["negative_cos_rate"]))
    print("Mismatch count:", result["mismatch_count"])
    print("Mismatch cosine mean: {:.6f}".format(result["mismatch_cos_mean"]))
    print("Mismatch negative cosine rate: {:.6f}".format(result["mismatch_negative_cos_rate"]))
    print("")
    print("Method,MSE,MAE")
    for name in ["base", "soft", "hard", "oracle"]:
        print("{},{:.6f},{:.6f}".format(name, result["mse"][name], result["mae"][name]))
    print("")
    print("Cosine quantiles")
    for q, value in zip(result["quantile_points"], result["cos_quantiles"]):
        print("{:.2f},{:.6f}".format(q, value))
    print("")
    print("Mismatch cosine quantiles")
    for q, value in zip(result["quantile_points"], result["mismatch_cos_quantiles"]):
        print("{:.2f},{:.6f}".format(q, value))
    print("")
    print("Cosine histogram")
    edges = result["hist_edges"]
    for i, count in enumerate(result["hist_counts"]):
        print("{:.2f},{:.2f},{}".format(edges[i], edges[i + 1], count))

    if args.diag_output_csv:
        write_csv(args.diag_output_csv, result)
        print("")
        print("Saved diagnostics:", args.diag_output_csv)


if __name__ == "__main__":
    main()
