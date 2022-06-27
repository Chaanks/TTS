# edited from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/distributed.py
import os
import hostlist

import torch
import torch.distributed as dist


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(rank_, num_gpus_, group_name, dist_backend, dist_url):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    
    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    size = int(os.environ['SLURM_NTASKS'])
    cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])

    # get node list from SURM
    hostname = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])

    # get IDs of reserved GPU
    gpu_ids = os.environ['SLURM_STEP_GPUS'].split(",")

    # define MASTER_ADD & MASTER_PORT
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = str(12345 + int(min(gpu_ids))) # to avoid port conflict on the same node

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f"rank: {rank}")
    print(f"size: {size}")
    print(f"master addr : {os.environ['MASTER_ADDR']}")
    print(f"master port : {os.environ['MASTER_PORT']}")
    
    # Initialize distributed communication
    dist.init_process_group(dist_backend, init_method=dist_url, world_size=size, rank=rank)
    #dist.init_process_group('nccl', init_method='env://', world_size=num_gpus, rank=rank, group_name=group_name)
