from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        if getattr(args, 'model', None) == 'BranchWorldModel':
            setattr(args, 'return_index', True)
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _model_core(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _unpack_batch(self, batch):
        if len(batch) == 5:
            batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index = batch
            return batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        return batch_x, batch_y, batch_x_mark, batch_y_mark, None

    def _maybe_build_memory(self, train_loader, reason='refresh'):
        model_core = self._model_core()
        if hasattr(model_core, 'build_memory') and bool(getattr(model_core, 'use_memory', True)):
            print('>>>>>>>building world-trajectory memory ({})<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(reason))
            memory_loader = DataLoader(
                train_loader.dataset,
                batch_size=train_loader.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
            model_core.build_memory(memory_loader, self.device)

    def _should_refresh_memory(self, epoch):
        model_core = self._model_core()
        if not hasattr(model_core, 'build_memory'):
            return False
        if not bool(getattr(model_core, 'use_memory', False)):
            return False
        warmup_epochs = max(0, getattr(self.args, 'wm_memory_warmup_epochs', 0))
        if epoch < warmup_epochs:
            return False
        update_freq = getattr(self.args, 'wm_memory_update_freq', 1)
        if update_freq <= 0:
            return epoch == warmup_epochs
        return (epoch - warmup_epochs) % update_freq == 0

    def _forward_model(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, future_y=None, query_index=None):
        model_core = self._model_core()
        if future_y is not None and hasattr(model_core, 'get_auxiliary_loss'):
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, future_y=future_y, query_index=query_index)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    def _auxiliary_loss(self):
        model_core = self._model_core()
        if hasattr(model_core, 'get_auxiliary_loss'):
            return model_core.get_auxiliary_loss()
        return None

    def _memory_stats(self):
        if not bool(getattr(self.args, 'wm_log_stats', 1)):
            return None
        model_core = self._model_core()
        if hasattr(model_core, 'get_memory_stats'):
            return model_core.get_memory_stats()
        return None

    def _stats_to_float(self, stats):
        if not stats:
            return None
        values = {}
        for key, value in stats.items():
            if torch.is_tensor(value):
                values[key] = float(value.detach().float().cpu())
            else:
                values[key] = float(value)
        return values

    def _average_memory_stats(self, stats_list):
        stats_list = [stats for stats in stats_list if stats]
        if not stats_list:
            return None
        keys = sorted(set().union(*(stats.keys() for stats in stats_list)))
        return {
            key: float(np.mean([stats[key] for stats in stats_list if key in stats]))
            for key in keys
        }

    def _format_memory_stats(self, stats):
        if not stats:
            return "memory: unavailable"
        ordered_keys = [
            "alpha_mean",
            "alpha_max",
            "raw_alpha_mean",
            "raw_alpha_std",
            "reliability_gate_mean",
            "reliability_gate_std",
            "residual_agreement_mean",
            "memory_residual_abs_mean",
            "delta_abs_mean",
            "effective_delta_abs_mean",
            "correction_scale_mean",
            "branch_weight_max_mean",
            "branch_weight_entropy",
            "retrieval_top1_mean",
            "retrieval_topk_mean",
            "retrieval_topk_std",
            "retrieval_excluded_frac",
            "retrieval_empty_after_exclusion",
            "retrieval_scarce_after_exclusion",
            "retrieval_raw_top1_overlap",
            "retrieval_raw_top1_gap_mean",
            "oracle_soft_mean",
            "oracle_hard_mean",
            "oracle_gain_mean",
            "oracle_gain_std",
            "gate_acc",
            "gate_auc",
            "gate_gain_corr",
            "reliability_gain_corr",
            "branch_oracle_gain_mean",
            "branch_oracle_entropy",
            "branch_oracle_acc",
            "base_mse",
            "adapted_mse",
            "mse_gain",
            "base_mse_norm",
            "adapted_mse_norm",
            "mse_gain_norm",
            "stdev_mean",
            "stdev_min",
            "stdev_p01",
        ]
        parts = []
        for key in ordered_keys:
            if key in stats:
                parts.append("{}: {:.6f}".format(key, stats[key]))
        return "memory | " + " ".join(parts)

    def _branchworld_extra_loss(self, outputs, targets):
        if self.args.model != 'BranchWorldModel':
            return outputs.new_tensor(0.0)
        extra = outputs.new_tensor(0.0)
        mae_weight = getattr(self.args, 'wm_mae_weight', 0.0)
        if mae_weight > 0:
            extra = extra + mae_weight * torch.mean(torch.abs(outputs - targets))
        freq_weight = getattr(self.args, 'wm_freq_loss_weight', 0.0)
        if freq_weight > 0:
            pred_freq = torch.fft.rfft(outputs, dim=1)
            true_freq = torch.fft.rfft(targets, dim=1)
            freq_loss = torch.mean(torch.abs(pred_freq - true_freq))
            extra = extra + freq_weight * freq_loss
        return extra

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion, return_memory_stats=False):
        total_loss = []
        memory_stats = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            future_y=batch_y[:, -self.args.pred_len:, :].detach(),
                        )
                else:
                    outputs = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        future_y=batch_y[:, -self.args.pred_len:, :].detach(),
                    )
                stats = self._stats_to_float(self._memory_stats())
                if stats is not None:
                    memory_stats.append(stats)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        memory_stats = self._average_memory_stats(memory_stats)
        self.model.train()
        if return_memory_stats:
            return total_loss, memory_stats
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if self._should_refresh_memory(epoch):
                self._maybe_build_memory(train_loader, reason='epoch {}'.format(epoch + 1))

            iter_count = 0
            train_loss = []
            epoch_memory_stats = []

            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index = self._unpack_batch(batch)
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if batch_index is not None:
                    batch_index = batch_index.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                future_y = batch_y[:, -self.args.pred_len:, :].detach()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            future_y=future_y, query_index=batch_index,
                        )

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        loss = loss + self._branchworld_extra_loss(outputs, batch_y)
                        aux_loss = self._auxiliary_loss()
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    train_loss.append(loss.item())
                    stats = self._stats_to_float(self._memory_stats())
                    if stats is not None:
                        epoch_memory_stats.append(stats)
                else:
                    outputs = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        future_y=future_y, query_index=batch_index,
                    )

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    loss = loss + self._branchworld_extra_loss(outputs, batch_y)
                    aux_loss = self._auxiliary_loss()
                    if aux_loss is not None:
                        loss = loss + aux_loss
                    train_loss.append(loss.item())
                    stats = self._stats_to_float(self._memory_stats())
                    if stats is not None:
                        epoch_memory_stats.append(stats)

                log_interval = max(1, getattr(self.args, 'wm_log_interval', 100))
                if (i + 1) % log_interval == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    stats = self._stats_to_float(self._memory_stats())
                    if stats is not None:
                        print("\t{}".format(self._format_memory_stats(stats)))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_memory_stats = self._average_memory_stats(epoch_memory_stats)
            if bool(getattr(self.args, 'wm_rebuild_memory_before_val', 0)):
                self._maybe_build_memory(train_loader, reason='post-epoch {} validation'.format(epoch + 1))
            vali_loss, vali_memory_stats = self.vali(vali_data, vali_loader, criterion, return_memory_stats=True)
            if bool(getattr(self.args, 'eval_test_each_epoch', 1)):
                test_loss, test_memory_stats = self.vali(test_data, test_loader, criterion, return_memory_stats=True)
            else:
                test_loss, test_memory_stats = float('nan'), None

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            if train_memory_stats is not None:
                print("Train {}".format(self._format_memory_stats(train_memory_stats)))
            if vali_memory_stats is not None:
                print("Vali  {}".format(self._format_memory_stats(vali_memory_stats)))
            if test_memory_stats is not None:
                print("Test  {}".format(self._format_memory_stats(test_memory_stats)))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self._maybe_build_memory(train_loader, reason='best checkpoint')

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            _, train_loader = self._get_data(flag='train')
            self._maybe_build_memory(train_loader, reason='loaded checkpoint')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        memory_stats = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            future_y=batch_y[:, -self.args.pred_len:, :].detach(),
                        )
                else:
                    outputs = self._forward_model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        future_y=batch_y[:, -self.args.pred_len:, :].detach(),
                    )
                stats = self._stats_to_float(self._memory_stats())
                if stats is not None:
                    memory_stats.append(stats)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        memory_stats = self._average_memory_stats(memory_stats)
        if memory_stats is not None:
            print("Test {}".format(self._format_memory_stats(memory_stats)))
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
