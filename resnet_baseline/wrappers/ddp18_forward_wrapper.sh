export MASTER_ADDR=`scontrol show hostnames $SLURM_STEP_NODELIST | head -1`
export MASTER_PORT="23458"
echo $MASTER_ADDR
python ../training/resnet_DDP_forward.py -g 8 -b 627 -o resnet18_forward_stats_DDP_maxb_GPU8.csv



