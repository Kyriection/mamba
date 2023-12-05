# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from tqdm import tqdm
import math
import numpy as np
from rouge import Rouge
from xopen import xopen
from copy import deepcopy
import os

from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
    get_qa_prompt_index,
    get_qa_prompt_only_true_index
)

def format_instruct_prompt(instruction):
    INSTRUCTION_KEY = "### Instruction:"
    RESPONSE_KEY = "### Response:"
    INTRO_BLURB = (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    )
    PROMPT_FOR_GENERATION = "{intro}\n{instruction_key}\n{instruction}\n{response_key}\n".format(
        intro=INTRO_BLURB,
        instruction_key=INSTRUCTION_KEY,
        instruction=instruction,
        response_key=RESPONSE_KEY,
    )
    return PROMPT_FOR_GENERATION

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--batch", type=int, default=1)

parser.add_argument('--only_true', action='store_true')
parser.add_argument("--batch_size", type=int, default=1)


parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--answer_idx", type=int, default=0)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map={"": device}, torch_dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

output_path = args.output_path

# Process input data with prompting
input_path = 'data/mdqa_10documents.jsonl.gz'
examples = []
prompts = []
all_model_documents = []
with xopen(input_path, 'r') as f:
    for line in f:
        if line.strip() != '':
            input_example = json.loads(line)
            question = input_example["question"]
            documents = []
            for ctx in deepcopy(input_example["ctxs"]):
                documents.append(Document.from_dict(ctx))
            if not documents:
                raise ValueError(f"Did not find any documents for example: {input_example}")
            if args.only_true:
                prompt = get_qa_prompt_only_true_index(
                    question,
                    documents,
                    mention_random_ordering=False,
                    query_aware_contextualization=False,
                    answer_idx=args.answer_idx
                )
            else:
                prompt = get_qa_prompt_index(
                    question,
                    documents,
                    mention_random_ordering=False,
                    query_aware_contextualization=False,
                    answer_idx=args.answer_idx
                )
            if "instruct" in args.model_name:
                prompt = format_instruct_prompt(prompt)
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

if len(prompts) > args.sample_num:
    print('Evaluate on {} samples (Total: {})'.format(args.sample_num, len(prompts)))
    prompts = prompts[:args.sample_num]
    examples = examples[:args.sample_num]
    all_model_documents = all_model_documents[:args.sample_num]


responses = []
with torch.no_grad():
    for batched_prompts in tqdm(chunks(prompts, args.batch_size), total=math.ceil(len(prompts) / args.batch_size)):
    #     inputs = tokenizer(batched_prompts, return_tensors="pt", padding=True).to(model.device)
    # for prompt in tqdm(prompts):
        if args.batch_size > 1:
            tokens_encode = tokenizer(batched_prompts, add_special_tokens=False, return_tensors='pt', truncation=True, padding=True)
            input_ids = tokens_encode.input_ids.to(device)
            attn_mask = tokens_encode.attention_mask.to(device)

        else:
            tokens_encode = tokenizer(batched_prompts, add_special_tokens=False, return_tensors='pt', truncation=True)
            input_ids = tokens_encode.input_ids.to(device)
            attn_mask = tokens_encode.attention_mask.to(device)

        if is_mamba:
            outputs = model.generate(
                input_ids=input_ids,
                max_length=100 + len(input_ids[0]),
                cg=True,
                enable_timing=False,
                top_k=args.topk,
            )
        else:
            outputs = model.generate(
                attention_mask=attn_mask,
                input_ids=input_ids,
                max_length=100 + len(input_ids[0]),
                pad_token_id=tokenizer.eos_token_id
            )

        for i, generated_sequence in enumerate(outputs):
            # input_ids = inputs[i].ids
            text = tokenizer.decode(generated_sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            if input_ids is None:
                prompt_length = 0
            else:
                prompt_length = len(
                    tokenizer.decode(
                        input_ids[i],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                )
            new_text = text[prompt_length:]
            responses.append(new_text)

    out_dir=os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = args.model_name
            output_example["model_temperature"] = 0
            output_example["model_top_p"] = "None"
            output_example["model_prompt_mention_random_ordering"] = False
            output_example["model_use_random_ordering"] = False
            f.write(json.dumps(output_example) + "\n")










