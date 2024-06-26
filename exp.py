import copy
import os
import os.path as osp
import json
import torch
import pickle
import logging
import numpy as np
from tdmvp_model import TDMVP_Model
from tqdm import tqdm
from API import *
from utils import *

class Exp:
    def __init__(self, args):
        super(Exp, self).__init__()
        self.args = args
        self.config = self.args.__dict__
        self.device = self._acquire_device()

        self._preparation()
        print_log(output_namespace(self.args))

        self._select_optimizer()
        self._select_criterion()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(0))
            print_log('Use GPU: {}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print_log('Use CPU')
        return device

    def _preparation(self):
        # seed
        set_seed(self.args.seed)
        # log and checkpoint
        self.path = osp.join(self.args.res_dir, self.args.ex_name)
        check_dir(self.path)

        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        with open(sv_param, 'w') as file_obj:
            json.dump(self.args.__dict__, file_obj)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, filename=osp.join(self.path, 'log.log'),
                            filemode='a', format='%(asctime)s - %(message)s')
        # prepare data
        self._get_data()
        # build the model
        self._build_model()

    def _build_model(self):
        args = self.args
        self.model = TDMVP_Model(tuple(args.in_shape), args.out_seq_len, args.time_kernel_size, args.groups, args.hid_S,
                                 args.hid_T, args.N_S, args.N_T, model_type='moga', drop_path=args.drop_path_rate).to(self.device)

    def _get_data(self):
        config = self.args.__dict__
        self.train_loader, self.vali_loader, self.test_loader, self.data_mean, self.data_std = load_data(**config)
        self.vali_loader = self.test_loader if self.vali_loader is None else self.vali_loader

    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.args.lr, steps_per_epoch=len(self.train_loader), epochs=self.args.epochs)
        return self.optimizer
    
    def _select_criterion(self):
        self.criterion = torch.nn.MSELoss()
        #self.criterion2 = torch.nn.L1Loss()

    def _save(self, name=''):
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoints_path, name + '.pth'))
        torch.save(self.ema_model.state_dict(), os.path.join(
            self.checkpoints_path, name + '_ema.pth'))

    def train(self, args):
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = 10
        self.step_start_ema = len(self.train_loader)*4
        self.step = 0
        
        config = args.__dict__
        recorder = Recorder(verbose=True)
        prev_vali_loss = 9999999999

        for epoch in range(config['epochs']):
            train_loss = []
            self.model.train()
            train_pbar = tqdm(self.train_loader)

            for batch_x, batch_y in train_pbar:
                self.optimizer.zero_grad()
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x)
                
                loss = self.criterion(pred_y, batch_y)# + self.criterion2(pred_y, batch_y)
                train_loss.append(loss.item())
                train_pbar.set_description('train loss: {:.4f}'.format(loss.item()))

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                self.step += 1

            train_loss = np.average(train_loss)
            for param_group in self.optimizer.param_groups:
                print('now_lr: ', param_group['lr'])

            #if epoch % args.log_step == 0:
                
            with torch.no_grad():
                vali_loss = self.vali(self.vali_loader)
                if prev_vali_loss > vali_loss:
                    self._save(name='best')
                    prev_vali_loss = vali_loss
                
                if epoch % (args.log_step * 100) == 0:
                    self._save(name=str(epoch))
            print_log("Epoch: {0} | Train Loss: {1:.4f} Vali Loss: {2:.4f}\n".format(
                epoch + 1, train_loss, vali_loss))
            recorder(vali_loss, self.model, self.path)

        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def vali(self, vali_loader):
        self.model.eval()
        self.ema_model.eval()
        '''
        self.model.eval()
        print('model eval')
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if i * batch_x.shape[0] > 1000:
                break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion1(pred_y, batch_y) + self.criterion2(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        self.model.train()
        '''
        print('ema model eval')
        preds_lst, trues_lst, total_loss = [], [], []
        vali_pbar = tqdm(vali_loader)
        for i, (batch_x, batch_y) in enumerate(vali_pbar):
            if self.args.dataname == 'mmnist' or self.args.dataname == 'mmnist30':
                if i * batch_x.shape[0] > 1000:
                    break

            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.ema_model(batch_x)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 pred_y, batch_y], [preds_lst, trues_lst]))

            loss = self.criterion(pred_y, batch_y)# + self.criterion2(pred_y, batch_y)
            vali_pbar.set_description(
                'vali loss: {:.4f}'.format(loss.mean().item()))
            total_loss.append(loss.mean().item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds_lst, axis=0)
        trues = np.concatenate(trues_lst, axis=0)
        mse, mae, ssim, psnr = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std, True)
        print_log('vali mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))
        return total_loss

    def test(self, args):
        self.model.eval()
        self.ema_model.eval()
        inputs_lst, trues_lst, preds_lst = [], [], []
        test_pbar = tqdm(self.test_loader)
        for batch_x, batch_y in test_pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            pred_y = self.ema_model(batch_x, is_train=False)
            list(map(lambda data, lst: lst.append(data.detach().cpu().numpy()), [
                 batch_x, batch_y, pred_y], [inputs_lst, trues_lst, preds_lst]))

        inputs, trues, preds = map(lambda data: np.concatenate(
            data, axis=0), [inputs_lst, trues_lst, preds_lst])

        folder_path = self.path+'/results/{}/sv/'.format(args.ex_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mse, mae, ssim, psnr = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std, True)
        print_log('mse:{:.4f}, mae:{:.4f}, ssim:{:.4f}, psnr:{:.4f}'.format(mse, mae, ssim, psnr))

        for np_data in ['inputs', 'trues', 'preds']:
            np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return mse
