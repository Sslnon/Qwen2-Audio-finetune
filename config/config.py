from dataclasses import dataclass, field
from typing import Optional, List
@dataclass
class PeftConfig:
    r: int = 32
    lora_alpha: int = 4
    target_modules: List = field(default_factory=lambda: ["q_proj","v_proj","o_proj","up_proj","gate_proj","down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False

@dataclass
class TrainConfig:
    train_strategy : str = "ddp"
    deepspeed_config: str = "./config/deepspeed.json"
    seed : int = 42
    lr: float = 1e-5
    batch_size: int = 1
    total_train_steps: int = 100000
    grad_accumulate_step : int = 16
    eval_step: int = 5000
    train_epoch: int = 5
    warmup_steps: int = 2000

@dataclass
class DataConfig:
    train_data_path: str = "./data/aishell-1/asr/test"
    eval_data_path: str = "./data/aishell-1/asr/test"
    prompt_path: str = "./data/multiprompt.jsonl"
    wav_type: str = "wav"
    num_workers: int = 8
    prefetch_factor: int  = 1

@dataclass
class EnvConfig:
    device_type: str = "cuda" # npu gpu
    save_path: str = "./exp"
    model_path: str = ""

@dataclass
class SLAMLLMConfig:
    encoder_path: str = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper/large-v3.pt" 
    encoder_dim : int = 1280
    ds_rate: int = 5
    llm_path: str = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct"
    llm_dim : int = 3584
# @dataclass
# class SwanConfig:
#     use_swanlab: bool = False
#     swanlab_dir: str = "test_wandb"
#     swanlab_entity_name: str = "sdinger"
#     swanlab_project_name: str = "project_name"
#     swanlab_exp_name: str = "exp_name"
#     log_file: str = "./test.log"
#     decode_log: str = "./test.log"
#     log_interval: int = 50

@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    peft: PeftConfig = field(default_factory=PeftConfig)
    slam: SLAMLLMConfig= field(default=SLAMLLMConfig)