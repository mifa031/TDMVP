import argparse
from exp import Exp
import torch

import warnings
warnings.filterwarnings('ignore')

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--device', default='cuda', type=str, help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='Debug', type=str)
    parser.add_argument('--use_gpu', default=True, type=bool)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--seed', default=1, type=int)

    # dataset parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--data_root', default='./data/')
    parser.add_argument('--dataname', default='mmnist', choices=['mmnist', 'taxibj'])
    parser.add_argument('--num_workers', default=10, type=int)

    # model parameters
    parser.add_argument('--in_shape', default=[10, 1, 64, 64], type=int,nargs='*') # [10, 1, 64, 64] for mmnist, [4, 2, 32, 32] for taxibj
    parser.add_argument('--hid_S', default=64, type=int) # spatial channels of encoder&decoder, hid_S * in_seq_len = mid input channels
    parser.add_argument('--hid_T', default=640, type=int) # mid channels (must divisible by out_seq_len) 
    parser.add_argument('--N_S', default=4, type=int)   # determines mid resolution
    parser.add_argument('--N_T', default=8, type=int)  # determines mid depth
    parser.add_argument('--groups', default=1, type=int)
    parser.add_argument('--drop_path_rate', default=0, type=float) #
    parser.add_argument('--out_seq_len', default=10, type=int) #
    parser.add_argument('--time_kernel_size', default=3, type=int) #

    # Training parameters
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--log_step', default=1, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    return parser


if __name__ == '__main__':
    args = create_parser().parse_args()
    config = args.__dict__

    exp = Exp(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>  start <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

    #load trained model
    best_model_path = './results/best.pth'
    #exp.model.load_state_dict(torch.load(best_model_path)) #same model load
    saved_weights = torch.load(best_model_path) #diff model load
    model_weights = exp.model.state_dict()
    updated_weights = {k: v for k, v in saved_weights.items() if k in model_weights}
    model_weights.update(updated_weights)
    exp.model.load_state_dict(model_weights)
    
    exp.ema_model = exp.model
    #exp.train(args)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>> testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    mse = exp.test(args)
