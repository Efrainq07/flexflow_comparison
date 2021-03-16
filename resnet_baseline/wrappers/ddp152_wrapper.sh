export MASTER_ADDR=`scontrol show hostnames $SLURM_STEP_NODELIST | head -1`
export MASTER_PORT="23458"
echo $MASTER_ADDR
python resnet152_DDP_training.py -g 1 --epochs 1
python resnet152_DDP_training.py -g 2 --epochs 1
python resnet152_DDP_training.py -g 4 --epochs 1
python resnet152_DDP_training.py -g 8 --epochs 1
