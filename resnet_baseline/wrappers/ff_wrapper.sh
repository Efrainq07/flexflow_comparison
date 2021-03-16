$FF_HOME/python/flexflow_python $FF_HOME/examples/python/pytorch/resnet.py -ll:py 1 -ll:gpu 1 -ll:fsize 10000 -ll:zsize 12192 --epochs 1 --batch-size 330  --fusion -ll:util 8 -ll:bgwork 8
mv training_run.csv resnet18_training_stats_GPU_1_FF.csv
$FF_HOME/python/flexflow_python $FF_HOME/examples/python/pytorch/resnet.py -ll:py 1 -ll:gpu 2 -ll:fsize 10000 -ll:zsize 12192 --epochs 1 --batch-size 330  --fusion -ll:util 8 -ll:bgwork 8
mv training_run.csv resnet18_training_stats_GPU_2_FF.csv
$FF_HOME/python/flexflow_python $FF_HOME/examples/python/pytorch/resnet.py -ll:py 1 -ll:gpu 4 -ll:fsize 10000 -ll:zsize 12192 --epochs 1 --batch-size 330 --fusion -ll:util 8 -ll:bgwork 8
mv training_run.csv resnet18_training_stats_GPU_4_FF.csv
$FF_HOME/python/flexflow_python $FF_HOME/examples/python/pytorch/resnet.py -ll:py 1 -ll:gpu 8 -ll:fsize 10000 -ll:zsize 12192 --epochs 1 --batch-size 330 --fusion -ll:util 8 -ll:bgwork 8
mv training_run.csv resnet18_training_stats_GPU_8_FF.csv
