import argparse
import csv
import os
import shlex
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_provider.data_factory import data_provider
from layers.BranchWorld import kmeans_torch
from models.BranchWorldModel import Model


class ArgumentParserWithFiles(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return shlex.split(arg_line, comments=True)


def build_parser():
    parser = ArgumentParserWithFiles(
        description="Diagnose BranchWorld latent trajectory clustering quality.",
        fromfile_prefix_chars="@",
    )

    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--model_id", type=str, default="cluster_diag")
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

    parser.add_argument("--diag_k_values", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--diag_output_csv", type=str, default="")
    parser.add_argument("--diag_probe_epochs", type=int, default=300)
    parser.add_argument("--diag_probe_lr", type=float, default=0.05)
    return parser


@torch.no_grad()
def collect_memory_items(model, train_loader, device):
    model.eval()
    keys = []
    trajectories = []
    residuals = []

    for batch in train_loader:
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch

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
        residual = norm_future - y_base.detach()
        if model.delta_clamp > 0:
            residual = residual.clamp(-model.delta_clamp, model.delta_clamp)

        _, key, trajectory = model._encode_trajectory(norm_x, norm_future)
        keys.append(key.detach().cpu())
        trajectories.append(trajectory.detach().cpu())
        residuals.append(residual.detach().cpu())

    keys = torch.cat(keys, dim=0)
    trajectories = torch.cat(trajectories, dim=0)
    residuals = torch.cat(residuals, dim=0)

    if keys.size(0) > model.memory_size:
        ids = torch.linspace(0, keys.size(0) - 1, model.memory_size).long()
        keys = keys[ids]
        trajectories = trajectories[ids]
        residuals = residuals[ids]

    return keys, trajectories, residuals


def residual_variance(residuals):
    center = residuals.mean(dim=0, keepdim=True)
    return (residuals - center).pow(2).mean()


def diagnose_k(flat_trajectories, residuals, k, kmeans_iters):
    k = min(k, flat_trajectories.size(0))
    centers, assign = kmeans_torch(flat_trajectories, k, kmeans_iters)
    distances = (flat_trajectories - centers[assign]).pow(2).sum(dim=1)
    inertia = distances.sum()
    inertia_per_sample = distances.mean()

    global_var = residual_variance(residuals).clamp_min(1e-12)
    weighted_cluster_var = residuals.new_tensor(0.0)
    cluster_rows = []
    for cluster_id in range(k):
        mask = assign == cluster_id
        size = int(mask.sum().item())
        if size == 0:
            cluster_var = residuals.new_tensor(float("nan"))
            ratio = residuals.new_tensor(float("nan"))
        else:
            cluster_var = residual_variance(residuals[mask])
            weighted_cluster_var = weighted_cluster_var + cluster_var * (size / residuals.size(0))
            ratio = cluster_var / global_var
        cluster_rows.append({
            "k": k,
            "cluster": cluster_id,
            "size": size,
            "residual_var": float(cluster_var),
            "residual_var_ratio": float(ratio),
        })

    return {
        "k": k,
        "inertia": float(inertia),
        "inertia_per_sample": float(inertia_per_sample),
        "global_residual_var": float(global_var),
        "weighted_cluster_residual_var": float(weighted_cluster_var),
        "weighted_residual_var_ratio": float(weighted_cluster_var / global_var),
    }, cluster_rows, centers, assign


def assign_to_centers(flat_trajectories, centers):
    return torch.cdist(flat_trajectories, centers).argmin(dim=1)


def normalize_features(train_x, val_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std


def linear_probe(train_x, train_y, val_x, val_y, num_classes, epochs, lr):
    train_x, val_x = normalize_features(train_x.float(), val_x.float())
    train_y = train_y.long()
    val_y = val_y.long()

    clf = torch.nn.Linear(train_x.size(1), num_classes)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(max(1, epochs)):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(train_x), train_y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_pred = clf(train_x).argmax(dim=1)
        val_pred = clf(val_x).argmax(dim=1)
        train_acc = (train_pred == train_y).float().mean()
        val_acc = (val_pred == val_y).float().mean()
    return float(train_acc), float(val_acc)


def prototype_key_attention(train_keys, train_labels, val_keys, num_classes, temperature):
    proto_keys = train_keys.new_zeros(num_classes, train_keys.size(1))
    for cluster_id in range(num_classes):
        mask = train_labels == cluster_id
        if mask.any():
            proto_keys[cluster_id] = train_keys[mask].mean(dim=0)
        else:
            proto_keys[cluster_id] = train_keys[0]
    proto_keys = torch.nn.functional.normalize(proto_keys, dim=-1)
    logits = val_keys @ proto_keys.t()
    weights = torch.softmax(logits / max(float(temperature), 1e-6), dim=-1)
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
    return float(entropy.mean()), float(entropy.mean() / torch.log(val_keys.new_tensor(float(num_classes))))


def main():
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unused args:", " ".join(unknown))

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    _, train_loader = data_provider(args, "train")
    _, val_loader = data_provider(args, "val")
    train_loader = DataLoader(
        train_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_loader.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = Model(args).to(device)
    model.eval()
    keys, trajectories, residuals = collect_memory_items(model, train_loader, device)
    val_keys, val_trajectories, _ = collect_memory_items(model, val_loader, device)

    flat = trajectories.flatten(start_dim=1)
    val_flat = val_trajectories.flatten(start_dim=1)
    print("Collected training memory items:", flat.size(0))
    print("Collected validation items:", val_flat.size(0))
    print("Trajectory dim:", flat.size(1))
    print("Residual shape:", tuple(residuals.shape))
    print("Encoder backbone:", args.wm_backbone)
    print("Encoder frozen:", bool(args.wm_freeze_encoder))
    print("")
    print(
        "K,inertia,inertia_per_sample,global_residual_var,weighted_cluster_residual_var,"
        "weighted_residual_var_ratio,linear_probe_train_acc,linear_probe_val_acc,"
        "random_acc,lift_over_random,attention_entropy,relative_attention_entropy"
    )

    summary_rows = []
    cluster_rows = []
    for k in args.diag_k_values:
        summary, clusters, centers, train_labels = diagnose_k(flat, residuals, k, args.wm_kmeans_iters)
        val_labels = assign_to_centers(val_flat, centers)
        train_acc, val_acc = linear_probe(
            keys,
            train_labels,
            val_keys,
            val_labels,
            summary["k"],
            args.diag_probe_epochs,
            args.diag_probe_lr,
        )
        random_acc = 1.0 / float(summary["k"])
        entropy, rel_entropy = prototype_key_attention(
            keys,
            train_labels,
            val_keys,
            summary["k"],
            args.wm_branch_temperature,
        )
        summary.update({
            "linear_probe_train_acc": train_acc,
            "linear_probe_val_acc": val_acc,
            "random_acc": random_acc,
            "lift_over_random": val_acc / random_acc,
            "attention_entropy": entropy,
            "relative_attention_entropy": rel_entropy,
        })
        summary_rows.append(summary)
        cluster_rows.extend(clusters)
        print(
            "{k},{inertia:.6f},{inertia_per_sample:.6f},{global_residual_var:.6f},"
            "{weighted_cluster_residual_var:.6f},{weighted_residual_var_ratio:.6f},"
            "{linear_probe_train_acc:.6f},{linear_probe_val_acc:.6f},{random_acc:.6f},"
            "{lift_over_random:.6f},{attention_entropy:.6f},{relative_attention_entropy:.6f}".format(**summary)
        )

    if args.diag_output_csv:
        out_dir = os.path.dirname(args.diag_output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.diag_output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "type",
                    "k",
                    "cluster",
                    "size",
                    "inertia",
                    "inertia_per_sample",
                    "global_residual_var",
                    "weighted_cluster_residual_var",
                    "weighted_residual_var_ratio",
                    "linear_probe_train_acc",
                    "linear_probe_val_acc",
                    "random_acc",
                    "lift_over_random",
                    "attention_entropy",
                    "relative_attention_entropy",
                    "residual_var",
                    "residual_var_ratio",
                ],
            )
            writer.writeheader()
            for row in summary_rows:
                out = {"type": "summary", **row}
                writer.writerow(out)
            for row in cluster_rows:
                out = {"type": "cluster", **row}
                writer.writerow(out)
        print("")
        print("Saved diagnostics:", args.diag_output_csv)


if __name__ == "__main__":
    main()
