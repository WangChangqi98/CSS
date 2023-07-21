
import torch

__all__ = ['torch_dist_sum']

def torch_dist_sum(gpu, *args):
    process_group = torch.distributed.group.WORLD
    tensor_args = []
    pending_res = []
    for arg in args:
        # if isinstance(arg, torch.Tensor):
        #     tensor_arg = arg.clone().reshape(-1).detach().cuda(gpu)
        # else:
        #     tensor_arg = torch.tensor(arg).reshape(-1).cuda(gpu)
        tensor_arg = arg.clone().detach().cuda(gpu)
        tensor_args.append(tensor_arg)
        pending_res.append(torch.distributed.all_reduce(tensor_arg, group=process_group, async_op=True))
    for res in pending_res:
        res.wait()
    return tensor_args
