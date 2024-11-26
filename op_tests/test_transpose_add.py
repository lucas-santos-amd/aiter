import torch
import torch.nn.functional as F
import ater
from test_common import checkAllclose, perftest
from torch.profiler import profile, record_function, ProfilerActivity

# input shape: torch.Size([4096, 64, 160]) (20480, 1, 128) 
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

# input shape: torch.Size([4096, 64, 160]) (47360, 1, 296)
# other shape: torch.Size([4096, 64, 160]) (10240, 160, 1)

shape = (4096, 64, 160)
stride1 = (47360, 1, 296)
# stride1 = (20480, 1, 128)
stride2 = (10240, 160, 1)

tensor1 = torch.empty_strided(shape, stride1, dtype=torch.float16, device='cuda')
tensor2 = torch.empty_strided(shape, stride2, dtype=torch.float16, device='cuda')
random_data = torch.rand(shape)
tensor1.copy_(random_data)
random_data = torch.rand(shape)
tensor2.copy_(random_data)


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    with_stack=True, with_modules=True, record_shapes = True) as prof:
    result = torch.add(tensor1, tensor2)
    result_con = result.contiguous()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True,
    with_stack=True, with_modules=True, record_shapes = True) as prof:
    output = torch.empty_like(tensor2)
    ater.transpose_add(output, tensor1, tensor2)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print(result_con== output)