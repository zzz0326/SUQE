# Stable Viewport-Based Unsupervised Compressed 360° Video Quality Enhancement

The enhancement videos can be seen in the result folder.

The overall training consists of three steps: training of the baseline model(train_baseline.py), shift prediction model(train_SPM.py) and unsupervised domain adaption(train_UDA.py). Each step requires updating the yml file according to the directory of the data, processing the specified panoramic video data, and training. 

## Dataset preparing:

# 1. Download the dataset through https://github.com/Archer-Tatsu/VQA-ODV.

# 2. Preparing data through the sequence extraction method:
```
python sequence_extraction.py
```
# 3. Generate LMDB for training:
```
python create_lmdb.py --opt_path step1.yml
```

## Examples of instructions required during training(4, 2 and single GPUs):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train_step1.py --opt_path step1.yml
```
```
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train_step1 --opt_path step1.yml
```
```
#CUDA_VISIBLE_DEVICES=0 python train_step1 --opt_path step1.yml
```

## Test the trained model and obtain the corresponding indicator values:
```
python PanEnh.py 
```


