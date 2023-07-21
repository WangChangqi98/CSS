def dist_init(port):
    import torch
    import os

    def init(host_addr, rank, local_rank, world_size, port):
        host_addr_full = 'tcp://' + host_addr + ':' + str(port)
        torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                            rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()

    def parse_host_addr(s):
        if '[' in s:
            left_bracket = s.index('[')
            right_bracket = s.index(']')
            prefix = s[:left_bracket]
            first_number = s[left_bracket+1:right_bracket].split(',')[0].split('-')[0]
            return prefix + first_number
        else:
            return s

    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    
    ip = parse_host_addr(os.environ['SLURM_STEP_NODELIST'])

    init(ip, rank, local_rank, world_size, port)

    return rank, local_rank, world_size

def local_dist_init(args, rank):
    import torch
    import os
    import torch.distributed as dist
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    os.environ['WORLD_SIZE'] = args.world_size
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    dist.init_process_group(backend='nccl', world_size=int(args.world_size), rank=rank)
    torch.cuda.set_device(rank)
    torch.autograd.set_detect_anomaly(True)

