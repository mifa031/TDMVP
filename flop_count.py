from fvcore.nn import FlopCountAnalysis, flop_count_table
import torch
from tdmvp_model import TDMVP_Model

######## Models ########
## Moving Mnist ##
model = TDMVP_Model(tuple([10, 1, 64, 64]), 10, 3, 1, 64,
                                 640, 4, 8, model_type='moga').to('cuda') ## 21.27M, 10.41G <-> (simvpv2)46.8M, 16.5G
## KTH20 ##
#model = TDMVP_Model(tuple([10, 1, 128, 128]), 20, 3, 1, 64,
#                                  320, 2, 6, model_type='moga').to('cuda') #6.591M, 53.66G <-> 12.2M, 62.8G
## KTH40 ##
#model = TDMVP_Model(tuple([10, 1, 128, 128]), 40, 3, 1, 64,
#                                  320, 2, 6, model_type='moga').to('cuda') #6.903M, 75.145G
## kitticaltech ##
#model = TDMVP_Model(tuple([10, 3, 128, 160]), 1, 3, 1, 64,
#                                  320, 2, 6, model_type='moga').to('cuda') ## 7.958M, 46.96G <-> 15.6M, 96.3G
######## Models End ########

######## Dummy Inputs ########
## Input Moving Mnist ##
input_dummy = torch.randn(1,10,1,64,64).to('cuda')
## Input KTH ##
#input_dummy = torch.randn(1,10,1,128,128).to('cuda')
## Input kitticaltech ##
#input_dummy = torch.randn(1,10,3,128,160).to('cuda')
######## Dummy Inputs End ########

flops = FlopCountAnalysis(model, input_dummy)
flops = flop_count_table(flops)
print(flops)
