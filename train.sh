export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
LOCAL_DIR=/work/2024/lixuanchen/project/Qwen2-Audio-finetune
cd $LOCAL_DIR

MODEL_PATH=/work/2024/lixuanchen/models/Qwen2-Audio-7B
TRIAN_DATA_PATH=/work/2024/lixuanchen/project/Qwen2-Audio-finetune/data/fleurs/train
EVAL_DATA_PATH=/work/2024/lixuanchen/project/Qwen2-Audio-finetune/data/fleurs/dev
TRAIN_STRATEGY=deepspeed # ddp deepspeed
DEVICE_TYPE=cuda # npu cuda
#parameters
num_workers=4
prefetch_factor=1
if [[ $TRAIN_STRATEGY == ddp ]]
then
# export ASCEND_VISIBLE_DEVICES=0,1 #CPU
# export CUDA_VISIBLE_DEVICES=0,1 #GPU
	torchrun \
		--nnodes 1 \
		--nproc_per_node 2 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH \
		++data.train_data_path=$TRIAN_DATA_PATH \
		++data.eval_data_path=$EVAL_DATA_PATH \
		++data.num_workers=$num_workers \
		++data.prefetch_factor=$prefetch_factor

else
export DEEPSPEED_CONFIG=./config/deepspeed.json
	deepspeed \
		--num_nodes 1 \
		--num_gpus 2 \
		./main.py \
		++train.train_strategy=$TRAIN_STRATEGY \
		++train.deepspeed_config=$DEEPSPEED_CONFIG \
		++env.device_type=$DEVICE_TYPE \
		++env.model_path=$MODEL_PATH \
		++data.train_data_path=$TRIAN_DATA_PATH \
		++data.eval_data_path=$EVAL_DATA_PATH \
		++train.eval_step=5000 \
		++train.lr=1e-5 \
		++train.train_epoch=5 \
		++train.grad_accumulate_step=32 \
		++train.warmup_steps=2000 \
		++peft.target_modules=["q_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"] \
		++peft.r=32 \
    	++peft.lora_alpha=4 \
		++peft.lora_dropout=0.05


fi