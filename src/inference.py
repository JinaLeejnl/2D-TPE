from transformers import AutoConfig, AutoTokenizer
import torch
import numpy as np
from typing import List
import multiprocessing
import datasets
import queue
import time
import os
import pickle
import json
import logging
import re
from tqdm import tqdm
from sft_minicpm import PROMPT_DICT
import sys
import os
import random
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)

from TPE-Llama.modeling_llama import LlamaForCausalLM

torch.set_printoptions(profile="full")
torch.multiprocessing.set_start_method('spawn',force=True)
num_workers = 32
gpu_num = 8
max_length = 4096
max_new_tokens = 100

def generate_prompt(instruction, question, input_seg=None):
  if input_seg:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input_seg=input_seg, question=question)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def read_data(input_file, to_tokenize_queue):
    with open(input_file, "r") as f:
        ds = json.load(f)

    print(len(ds))

    for i, data in tqdm(enumerate(ds), total=len(ds)):
        data['idx'] = i
        to_tokenize_queue.put(data)

    for i in range(num_workers):
        to_tokenize_queue.put(None)


def encode_and_insert_separators(table_array, tokenizer):
    separator_col = [1425] # '▁|'
    separator_row = [48017] # '-'

    separator_row_end = [3] # '<SEP>'
    separator_col_end = [4] # '<CLS>'

    new_table = []
    
    for k, row in enumerate(table_array):
        new_row, new_separator = [], []
        for col in row:
            encoded_col = tokenizer.encode(str(col), add_special_tokens=False)
            new_row.append(encoded_col)
            new_row.append(separator_col)  # Insert '|' between each coded column

            new_separator.append(separator_col_end if k == len(table_array) - 1 else separator_row)
            new_separator.append(separator_col)
        new_row.append(separator_row_end)
        new_separator.append(separator_row_end)
        new_table.append(new_row)
        new_table.append(new_separator)
    return new_table


def tokenize_data(to_tokenize_queue, to_output_queue, rank):
    model_name = '/output/rel_extraction_2d'
    config = AutoConfig.from_pretrained(model_name)
    config.remove_unused_columns = False
    config._flash_attn_2_enabled = True
    config.output_loss = False
    config.pad_token_id = 0
    model = LlamaForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16).to(f"cuda:{rank%gpu_num}")
    model = model.to(dtype=torch.bfloat16)
    model.eval()
    
    while True:
        data = to_tokenize_queue.get()
        if data is None:
            break

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        tokenizer.add_special_tokens({"pad_token":"<pad>"})
        tokenizer.pad_token_id = 0  
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left" 

        source = generate_prompt(instruction = data["instruction"], input_seg = data["input_seg"], question = data["question"])

        parts = re.split(r'(\[TAB\] )|(\n\n### Response)', source)
        parts = [part for part in parts if part is not None]

        part1 = parts[0] + parts[1]
        table_data = parts[2]
        part3 = parts[3] + parts[4]

        # Convert a table from text format to list format
        if 'col:' in table_data and 'row 1:' in table_data:
            headers_part, rows_part = table_data.split(' row 1:', 1)
            headers = headers_part.strip('col: ').split(' | ')
            headers = [header.strip(" |") if header.strip(" |") else 'None' for header in headers]
            rows_part = 'row 1:' + rows_part

            rows = rows_part.split(' [SEP]')
            data_rows = []
            for row in rows:
                if row:
                    parts = row.strip().split(' | ')[1:]
                    cleaned_parts = [part.strip(" |") if part.strip(" |") else 'None' for part in parts]
                    data_rows.append(cleaned_parts)

            table_array = [headers] + data_rows
        elif 'col:' in table_data and 'row 1:' not in table_data:
            rows = table_data.split(" [SEP] ")
            headers = rows[0].split(" | ") if rows[0].endswith("|") else (rows[0] + " |").split(" | ")
            headers = [header.strip(" |") if header.strip(" |") else 'None' for header in headers][1:]
            data_rows = []
            for row in rows[1:]:
                if row:
                    parts = row.strip("").split(' | ')
                    cleaned_parts = [part.strip(" |") if part.strip(" |") else 'None' for part in parts]
                    data_rows.append(cleaned_parts)

            table_array = [headers] + data_rows
        else:
            rows = table_data.split(" [SEP] ")
            headers = rows[0].split(" | ") if rows[0].endswith("|") else (rows[0] + " |").split(" | ")
            headers = [header.strip(" |") if header.strip(" |") else 'None' for header in headers]
            data_rows = []
            for row in rows[1:]:
                if row:
                    parts = row.strip("").split(' | ')
                    cleaned_parts = [part.strip(" |") if part.strip(" |") else 'None' for part in parts]
                    data_rows.append(cleaned_parts)

            table_array = [headers] + data_rows
        

        # Determine whether the table is a rectangle
        expected_columns = len(table_array[0])
        flag = True
        for row in table_array:
            if len(row) != expected_columns:
                flag = False
                break
        
        if not flag:
            logging.error(f"table_array at index {data['idx']}")
            continue

        #add sep
        new_table = encode_and_insert_separators(table_array, tokenizer)


        # Determine whether the table is a rectangle
        expected_columns = len(new_table[0])
        flag = True
        for row in new_table:
            if len(row) != expected_columns:
                flag = False
                break
        
        if not flag:
            problematic_indices.append(idx)
            logging.error(f"new_table at index {idx}")
            continue

        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(text=part1, add_special_tokens=False)
        l_part1 = len(input_ids)
        tx = list(range(l_part1))
        ty = list(range(l_part1))

        px = list(range(l_part1))
        py = list(range(l_part1))

        height = len(new_table)
        for i, row in enumerate(new_table):
            width = len(row)
            row_x = l_part1 - 1 + (width + 1) * (i + 1)
            for j, item in enumerate(row):
                row_y = l_part1 - 1 + (height + 1) * (j + 1)
                item_en = item
                px.extend([row_x] * len(item_en))
                py.extend([row_y] * len(item_en))
                input_ids.extend(item_en)

        for i, row in enumerate(new_table):
            for j, item in enumerate(row):
                tx_count = len(tx)
                tx.extend(list(range(tx_count, tx_count + len(item))))
        transpose_new_table = np.transpose(new_table).tolist()
        ty_list, count = [], len(ty)
        for i, row in enumerate(transpose_new_table):
            ty_list.append([])
            for j, item in enumerate(row):
                ty_list[-1].append(list(range(count, count + len(item))))
                count += len(item)
        transpose_ty_list = np.transpose(ty_list).tolist()
        for i, row in enumerate(transpose_ty_list):
            for j, item in enumerate(row):
                ty.extend(item)

        k_part3_start = l_part1 - 1 + (width + 1) * (height + 1)

        part3_en = tokenizer.encode(text=part3, add_special_tokens=False)
        input_ids.extend(part3_en)
        tx_count = len(tx)
        ty_count = len(ty)
        assert tx_count == ty_count
        tx.extend(list(range(tx_count, tx_count + len(part3_en))))
        ty.extend(list(range(ty_count, ty_count + len(part3_en))))

        k_part3_end = k_part3_start + len(part3_en)
        px.extend(list(range(k_part3_start, k_part3_end)))
        py.extend(list(range(k_part3_start, k_part3_end)))

        if len(input_ids) > tokenizer.model_max_length-1:
            continue
            input_ids = input_ids[-tokenizer.model_max_length+1:]
            px = px[-tokenizer.model_max_length+1:]
            py = py[-tokenizer.model_max_length+1:]
            tx = tx[-tokenizer.model_max_length+1:]
            ty = ty[-tokenizer.model_max_length+1:]

        pi = np.concatenate([px, py])
        ti = np.concatenate([tx, ty])

        input_ids = torch.tensor(input_ids).reshape(1, -1)
        token_ids = torch.tensor(ti).reshape(1, -1)
        position_ids = torch.tensor(pi).reshape(1, -1)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids.to(f"cuda:{rank%gpu_num}"),
                token_ids=token_ids.to(f"cuda:{rank%gpu_num}"),
                position_ids=position_ids.to(f"cuda:{rank%gpu_num}"),
                use_cache=True,
            )
            pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            past_key_values = outputs.past_key_values
            npx = [px[-1] + 1]
            npy = [py[-1] + 1]

            ntx = [tx[-1] + 1]
            nty = [ty[-1] + 1]

            pi = np.concatenate([npx, npy])
            position_ids = torch.tensor(pi).reshape(1, -1)
            ti = np.concatenate([ntx, nty])
            token_ids = torch.tensor(ti).reshape(1, -1)
            generated_ids = [pred_token_idx.item()]

            for _ in range(max_new_tokens - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    token_ids=token_ids.to(f"cuda:{rank%gpu_num}"),
                    position_ids=position_ids.to(f"cuda:{rank%gpu_num}"),
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                npx = [npx[-1] + 1]
                npy = [npy[-1] + 1]
                ntx = [ntx[-1] + 1]
                nty = [nty[-1] + 1]

                pi = np.concatenate([npx, npy])
                position_ids = torch.tensor(pi).reshape(1, -1)
                ti = np.concatenate([ntx, nty])
                token_ids = torch.tensor(ti).reshape(1, -1)
                generated_ids.append(pred_token_idx.item())

                if pred_token_idx == tokenizer.eos_token_id:
                    break
            
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            result = { 'idx': data['idx'],
                       'instruction': data['instruction'],
                       'input_seg': data['input_seg'],
                       'question': data['question'],
                       'output': data['output'],
                       'predict': generated_text}
       
        to_output_queue.put(result)

    to_output_queue.put(None)

def output_data(to_output_queue):
    count = 0
    start_time = None
    finish_tag = 0
    
    while True:
        data = to_output_queue.get()
        if start_time is None:
            start_time = time.time()
        if data is None:
            finish_tag += 1
            if finish_tag == num_workers:
                print("End")
                break
            else:
                continue
        else:
            with open('./res/rel_extraction_2d_res.json', 'a') as f:
                try:
                    json.dump(data, f)
                    f.write('\n')
                except:
                    continue
            
            count += 1
            if count % 100 == 0:
                end_time = time.time()
                print(count)
                print(f"Spend:{(end_time-start_time)} s")
        

if __name__ == "__main__":
    import sys

    to_tokenize_queue = multiprocessing.Queue(maxsize=100000)
    to_output_queue = multiprocessing.Queue(maxsize=100000)
    
    # start
    reader_process = multiprocessing.Process(target=read_data, args=("/eval_data/rel_extraction_test.json", to_tokenize_queue))
    tokenizer_processes = [multiprocessing.Process(target=tokenize_data, args=(to_tokenize_queue, to_output_queue, rank)) for rank in range(num_workers)]
    output_process = multiprocessing.Process(target=output_data, args=(to_output_queue,))
    
    reader_process.start()
    for p in tokenizer_processes:
        p.start()
    output_process.start()

    start_time =  time.time()
    reader_process.join()
    for p in tokenizer_processes:
        p.join()
    output_process.join()
    end_time = time.time()
    print(end_time-start_time)