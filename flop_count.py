from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
from tdmvp_model import TDMVP_Model

######## Models ########
## Moving Mnist ##
model = TDMVP_Model(tuple([10, 1, 64, 64]), 10, 3, 4, 64,
                                 640, 4, 8, model_type='moga').to('cuda') ## (simvpv2)46.8M, 16.5G
######## Models End ########

######## Dummy Inputs ########
## Input Moving Mnist ##
input_dummy = torch.randn(1,10,1,64,64).to('cuda')
######## Dummy Inputs End ########

flops = FlopCountAnalysis(model, input_dummy)
flops = flop_count_table(flops)
print(flops)
