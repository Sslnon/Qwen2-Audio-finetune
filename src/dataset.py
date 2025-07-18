import torch
# import torch_npu
import os
import json
import numpy as np
import kaldiio
from torch.nn.utils.rnn import pad_sequence
import copy
import time
import soundfile

class AudioDatset(torch.utils.data.Dataset):
    def __init__(self,data_path,prompt_path=None,wav_type="wav"):
        print("[Dataset] Processing")
        self.wav_scp = {}
        self.tasks = []
        self.utt2num_samples = {}
        self.prompt = {}
        self.wav_type = wav_type
        with open(os.path.join(data_path,"wav.scp")) as f:
            for line in f:
                id,wav_path = line.strip().split(" ",1)
                self.wav_scp[id] = wav_path
        with open(os.path.join(data_path,"multitask.jsonl")) as f:
            for line in f:
                self.tasks.append(json.loads(line))
        with open(os.path.join(prompt_path)) as f:
            for line in f:
                item = json.loads(line)
                self.prompt[item["task"]] = item["prompt"]

    def __len__(self):
        return len(self.tasks)
    def __getitem__(self,idx):
        key = self.tasks[idx]["key"]
        target = self.tasks[idx]["target"]
        prompt = self.prompt[self.tasks[idx]["task"]]
        audio = soundfile.read(self.wav_scp[key])[0]
        return {
            "prompt":prompt,
            "audio":audio,
            "target":target
        }
    
    
def collate_fn_qwen2audio(samples,processor):
    prompt = [_["prompt"] for _ in samples]
    audio = [ _["audio"] for _ in samples]
    target = [ _["target"] for _ in samples]
    processed_data = processor(text=[i+j for i,j in zip(prompt,target)], audios=audio, sampling_rate=processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
    # 处理labels的生成
    labels = copy.deepcopy(processed_data["input_ids"])
    text_ids = processor(prompt,return_tensors="pt", padding=True)
    for i,attention_mask  in enumerate(text_ids["attention_mask"]):
        labels[i,:sum(attention_mask )] = -100
    processed_data["labels"]=labels
    return  processed_data

def collate_fn_slamllm(samples,tokenizer):
    prompt = [_["prompt"] for _ in samples]
    audio = [ _["audio"] for _ in samples]
    target = [ _["target"] for _ in samples]
    special_tokens = ["<|audio_bos|>","<|AUDIO|>","<|audio_eos|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    processed_data = tokenizer(text=[i+j for i,j in zip(prompt,target)], return_tensors="pt")
    # 处理labels的生成
    labels = copy.deepcopy(processed_data["input_ids"])
    text_ids = tokenizer(prompt,return_tensors="pt", padding=True)
    for i,attention_mask  in enumerate(text_ids["attention_mask"]):
        labels[i,:sum(attention_mask )] = -100
    processed_data["labels"]=labels
    return  processed_data



