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

    def _init_branchworld_diag(self):
        return {
            'count': 0,
            'base_mse_sum': 0.0,
            'final_mse_sum': 0.0,
            'decoded_oracle_mse_sum': 0.0,
            'target_oracle_mse_sum': 0.0,
            'useful_count': 0,
            'target_useful_count': 0,
            'advantage_sum': 0.0,
            'target_advantage_sum': 0.0,
            'mem_weight_sum': 0.0,
            'mem_weight_useful_sum': 0.0,
            'mem_weight_not_useful_sum': 0.0,
            'best_branch_weight_sum': 0.0,
            'best_branch_weight_useful_sum': 0.0,
            'final_better_count': 0,
        }

    def _update_branchworld_diag(self, stats, diag, true):
        base = diag['base']
        final = diag['pred']
        memory = diag['memory']
        weights = diag['weights']
        base_err = (base - true).pow(2).mean(dim=(1, 2))
        final_err = (final - true).pow(2).mean(dim=(1, 2))
        mem_err = (memory - true.unsqueeze(1)).pow(2).mean(dim=(2, 3))
        best_mem_err, best_mem_id = mem_err.min(dim=1)
        decoded_oracle_err = torch.minimum(base_err, best_mem_err)
        useful = best_mem_err < base_err
        mem_weight = weights[:, 1:].sum(dim=1)
        best_branch_weight = weights[:, 1:].gather(1, best_mem_id.unsqueeze(1)).squeeze(1)

        batch_count = true.size(0)
        stats['count'] += batch_count
        stats['base_mse_sum'] += base_err.sum().item()
        stats['final_mse_sum'] += final_err.sum().item()
        stats['decoded_oracle_mse_sum'] += decoded_oracle_err.sum().item()
        stats['useful_count'] += useful.sum().item()
        stats['advantage_sum'] += (base_err - best_mem_err).clamp_min(0).sum().item()
        stats['mem_weight_sum'] += mem_weight.sum().item()
        stats['best_branch_weight_sum'] += best_branch_weight.sum().item()
        stats['final_better_count'] += (final_err < base_err).sum().item()
        if useful.any():
            stats['mem_weight_useful_sum'] += mem_weight[useful].sum().item()
            stats['best_branch_weight_useful_sum'] += best_branch_weight[useful].sum().item()
        if (~useful).any():
            stats['mem_weight_not_useful_sum'] += mem_weight[~useful].sum().item()

        if 'target' in diag:
            target_err = (diag['target'] - true.unsqueeze(1)).pow(2).mean(dim=(2, 3))
            best_target_err = target_err.min(dim=1).values
            target_useful = best_target_err < base_err
            stats['target_oracle_mse_sum'] += torch.minimum(base_err, best_target_err).sum().item()
            stats['target_useful_count'] += target_useful.sum().item()
            stats['target_advantage_sum'] += (base_err - best_target_err).clamp_min(0).sum().item()

    def _print_branchworld_diag(self, stats):
        count = max(1, stats['count'])
        useful_count = max(1, stats['useful_count'])
        not_useful_count = max(1, stats['count'] - stats['useful_count'])
        print('BranchWorld diagnostics:')
        print('  base_mse:{:.6f} final_mse:{:.6f} decoded_oracle_mse:{:.6f} target_oracle_mse:{:.6f}'.format(
            stats['base_mse_sum'] / count,
            stats['final_mse_sum'] / count,
            stats['decoded_oracle_mse_sum'] / count,
            stats['target_oracle_mse_sum'] / count if stats['target_oracle_mse_sum'] > 0 else float('nan'),
        ))
        print('  memory_useful_rate:{:.2f}% target_useful_rate:{:.2f}% final_better_than_base:{:.2f}%'.format(
            100.0 * stats['useful_count'] / count,
            100.0 * stats['target_useful_count'] / count,
            100.0 * stats['final_better_count'] / count,
        ))
        print('  avg_positive_advantage:{:.6f} target_avg_positive_advantage:{:.6f}'.format(
            stats['advantage_sum'] / count,
            stats['target_advantage_sum'] / count,
        ))
        print('  mem_weight_mean:{:.6f} mem_weight_when_useful:{:.6f} mem_weight_when_not_useful:{:.6f} best_branch_weight_when_useful:{:.6f}'.format(
            stats['mem_weight_sum'] / count,
            stats['mem_weight_useful_sum'] / useful_count,
            stats['mem_weight_not_useful_sum'] / not_useful_count,
            stats['best_branch_weight_useful_sum'] / useful_count,
        ))

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.model == 'BranchWorldModel' and getattr(self.args, 'wm_train_gate_only', 0):
            trainable = []
            for name, param in self.model.named_parameters():
                keep = '.gate.' in name or '.base_gate.' in name or name.startswith('gate.') or name.startswith('base_gate.')
                param.requires_grad = keep
                if keep:
                    trainable.append(param)
            return optim.Adam(trainable, lr=self.args.learning_rate)
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
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

        init_checkpoint = getattr(self.args, 'wm_init_checkpoint', '')
        if self.args.model == 'BranchWorldModel' and init_checkpoint:
            print('>>>>>>>loading BranchWorld init checkpoint: {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(init_checkpoint))
            self.model.load_state_dict(torch.load(init_checkpoint, map_location=self.device))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            if self._should_refresh_memory(epoch):
                self._maybe_build_memory(train_loader, reason='epoch {}'.format(epoch + 1))

            iter_count = 0
            train_loss = []

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

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
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
        branchworld_diag = None
        model_core = self._model_core()
        if self.args.model == 'BranchWorldModel' and hasattr(model_core, 'memory_diagnostics'):
            branchworld_diag = self._init_branchworld_diag()
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_index = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if batch_index is not None:
                    batch_index = batch_index.to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                diag = None
                if branchworld_diag is not None:
                    diag = model_core.memory_diagnostics(batch_x, query_index=batch_index)
                    outputs = diag['pred']
                elif self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                true_for_diag = batch_y[:, :, f_dim:]
                if diag is not None:
                    diag_for_update = {}
                    for key, value in diag.items():
                        if key == 'weights':
                            diag_for_update[key] = value
                        elif value.dim() == 4:
                            diag_for_update[key] = value[:, :, :, f_dim:]
                        else:
                            diag_for_update[key] = value[:, :, f_dim:]
                    self._update_branchworld_diag(branchworld_diag, diag_for_update, true_for_diag)
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
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        if branchworld_diag is not None:
            self._print_branchworld_diag(branchworld_diag)
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
