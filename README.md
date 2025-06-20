# Qwen2-Audio-finetune  
This is a repository prepared for fine-tuning Qwen2-Audio, supporting both GPU and NPU. It supports reading and writing data in ark and wav formats, and is compatible with DDP, DeepSpeed, and LoRA.  

# Requirements  
python 3.11
cuda 12.4
pip install -r requirements.txt

# Train  
## Data Preparation  
Please refer to the sample data path `/data/aishell-1` to prepare your data.  
use wav instead of ark
```  
multitask.jsonl  
my_wav.scp  
multiprompt.jsonl  
```  
Modify the following two variables in `train.sh`, or simply use the default provided data without preparation:  
```  
TRAIN_DATA_PATH  
EVAL_DATA_PATH  
```  

## Configuration Preparation  
Set the following necessary environment variables in `train.sh`:  
```  
LOCAL_DIR=  
MODEL_PATH=  
```  
Set the required variables in `config/config.py`.  

## Running the Code  
Run the following command to start training:  
```  
bash train.sh  
```  

# TODO

add swanlab / wandb
