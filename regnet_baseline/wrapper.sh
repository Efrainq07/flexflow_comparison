export MASTER_ADDR=`scontrol show hostnames $SLURM_STEP_NODELIST | head -1`
export MASTER_PORT="23458"
echo $MASTER_ADDR
python regnet_DDP_training.py -n 2 -g 8 --epochs 10
