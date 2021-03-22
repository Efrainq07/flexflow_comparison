import os
import tempfile
import torch,torchvision
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.models as models
import torchprof
import time
import pandas as pd


class RandomDataset(torch.utils.data.Dataset):
  def __init__(self,*args):
    super(RandomDataset,self).__init__()
    self.values = torch.randn(*args, dtype=torch.float)
    self.labels = torch.zeros(args[0])

  def __len__(self):
    return len(self.values)  # number of samples in the dataset

  def __getitem__(self, index):
    return self.values[index], self.labels[index]

def setup(rank, world_size):
    print("Running init_process_group...")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("Finished init_process_group...")

def cleanup():
    dist.destroy_process_group()

def train(local_rank, args):
    rank = args.nr * args.gpus + local_rank	
    setup(rank, args.world_size)
    transform = transforms.Compose([
                torchvision.transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
            ])
    batch_size = args.batchsize
    train_dataset = RandomDataset(args.batchsize*100*args.world_size,3,224,224)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size,rank=rank)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=2,sampler=sampler)

    model = models.resnet18()
    model.eval()
    torch.cuda.set_device(local_rank)
    model.cuda()
    print("GPU initialization")
    dummy_input = torch.randn(1, 3,224,224, dtype=torch.float).to(local_rank)
    for _ in range(10):
        _ = model(dummy_input)
    model = nn.parallel.DistributedDataParallel(model,device_ids=[local_rank])
    training_run_data = pd.DataFrame(columns=['epoch','batch','batch_size','gpu_number','time'])
    prof_file = open("../results/resnet18_mem_profiling.txt", "w")
    for i, data in enumerate(trainloader, 0):
        starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        starter.record()
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            with torchprof.Profile(model, use_cuda=True, profile_memory=True) as prof:
                outputs = model(inputs)
        ender.record()
        if rank==0:
            torch.cuda.synchronize()
            timer = starter.elapsed_time(ender)
            training_run_data=training_run_data.append(
                        {'batch':i,'batch_size':batch_size,'gpu_number':args.gpus*args.nodes,'time (ms)':timer/(batch_size*args.gpus),'throughput':1000*(batch_size*args.gpus)/timer},
                    ignore_index=True)
            training_run_data.to_csv(args.output,index=False)
            print("Batch: %d  Time per Image: %.2f ms Throughput:%.2f"%(i,timer/(batch_size*args.gpus),1000*(batch_size*args.gpus)/timer))
            if i%20==19:
                prof_file.write(prof.display(show_events=False))
    cleanup()

def main():
    print("Parsing arguments...")
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-b','--batchsize', default=12, type=int) 
    parser.add_argument('-o','--output', default='../results/resnet18_forward_stats', type=str,
                        help='output file name or location') 
    args = parser.parse_args()
    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '1234'
    if 'SLURMD_NODENAME' in os.environ:
        if os.environ['SLURMD_NODENAME']==os.environ['MASTER_ADDR']:     
            args.nr=0
        else:
            args.nr=1
    else:
        args.nr=0
    args.world_size = args.gpus * args.nodes   
    print("Spawning processes...")       
    mp.spawn(train, nprocs=args.gpus, args=(args,))

if __name__=='__main__':
    main()
    
