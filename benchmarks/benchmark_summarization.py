# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

import tqdm
import numpy as np
from rouge import Rouge
from xopen import xopen



parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--batch", type=int, default=1)

parser.add_argument("--input_path", type=str, default=None)
parser.add_argument("--sample_num", type=int, default=100)
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba-")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


print(f"Loading dataset")
requests = []
with open(args.input_path, 'r') as f:
    for line in f:
        if line.strip() != '':
            requests.append(json.loads(line))

print(len(requests))
if args.sample_num < len(requests):
    print('Sample {} Examples from {} samples'.format(args.sample_num, len(requests)))
requests = requests[:args.sample_num]


torch.random.manual_seed(0)
results = []
rouge = Rouge()
rouge1_score_list = []
rouge2_score_list = []
rougel_score_list = []

with torch.no_grad():
    for request in tqdm.tqdm(requests):
        result = {'request': request, 'result': {}}
        prompt = request['article']
        label = request['summary_gt']
        temperature = request['temperature']
        stop = request['stop']

        tokens_encode = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        input_ids = tokens_encode.input_ids.to(device)
        attn_mask = tokens_encode.attention_mask.to(device)

        if is_mamba:
            output_sequences = model.generate(
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=temperature,
                top_k=args.topk,
                top_p=request['top_p'],
            )
        else:
            output_sequences = model.generate(
                attention_mask=attn_mask,
                input_ids=input_ids,
                max_length=request['max_tokens'] + len(input_ids[0]),
                temperature=temperature,
                top_p=request['top_p'],
                return_dict_in_generate=True, output_scores=True,
                pad_token_id=tokenizer.eos_token_id
            )

        tokens = tokenizer.convert_ids_to_tokens(output_sequences['sequences'].squeeze(0))[len(input_ids[0]):]
        logprobs = [logits.log_softmax(dim=-1).max().item() for logits in output_sequences['scores']]
        top_logprobs = [{i: v for i, v in zip(tokens, logprobs)}]

        generate_text = tokenizer.decode(output_sequences['sequences'].squeeze(0)[len(input_ids[0]):])
        generate_text = generate_text[: generate_text.find(stop[0])]

        scores = rouge.get_scores(generate_text, label)[0]
        rouge1_score_list.append(scores['rouge-1']['f'])
        rouge2_score_list.append(scores['rouge-2']['f'])
        rougel_score_list.append(scores['rouge-l']['f'])

        result['result'] = {
            "choices": [
                {
                    "text": generate_text,
                    "logprobs": {
                        "tokens": tokens, 
                        "token_logprobs": logprobs, 
                        "top_logprobs": top_logprobs, 
                        "text_offset": []
                    }, 
                    "finish_reason": "length"
                }
            ], 
            "request_time": {
                "batch_time": 0, 
                "batch_size": 1}
        }
        
        results.append(result)
        print('rouge-1: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))

print('Final Results: {:.6f}, rouge-2: {:.6f}, rouge-l: {:.6f}'.format(np.mean(rouge1_score_list), np.mean(rouge2_score_list), np.mean(rougel_score_list)))







