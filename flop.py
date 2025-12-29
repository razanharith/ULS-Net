from LM_Net import LM_Net
import torch
from ptflops import get_model_complexity_info
import re

# Ensure the model is moved to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model that's already available
model = LM_Net(3, 1).to(device)

# Ensure the input tensor is on the same device as the model
macs, params = get_model_complexity_info(
    model, (3, 224, 112), as_strings=True, print_per_layer_stat=False, verbose=True
)

# Extract the numerical value
flops = eval(re.findall(r'([\d.]+)', macs)[0]) * 2
# Extract the unit
flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]

print('Computational complexity: {:<8}'.format(macs))
print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
print('Number of parameters: {:<8}'.format(params))