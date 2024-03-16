# Stable Viewport-Based Unsupervised Compressed 360° Video Quality Enhancement

The enhancement videos can be seen in the result folder.

You need to download the dataset through https://github.com/Archer-Tatsu/VQA-ODV.

The overall training consists of three steps: training of the baseline model, shift prediction model and unsupervised domain adaption. Each step requires processing the specified panoramic video data and updating the yml file according to the directory of the processed data. Programs that process data and yml file examples are stored in the process_data and config files respectively. 

Examples of instructions required during training:

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train_step1.py --opt_path step1.yml

# 2 GPUs
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train_step1 --opt_path step1.yml

# 1 GPU
#CUDA_VISIBLE_DEVICES=0 python train_step1 --opt_path step1.yml

Test the trained model:
```
python PanEnh.py 
```


