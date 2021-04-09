export MASTER_ADDR=`scontrol show hostnames $SLURM_STEP_NODELIST | head -1`
export MASTER_PORT="23458"
echo $MASTER_ADDR
python ../training/resnet18_DDP_training.py -g 8 --epochs 1 -b 250

