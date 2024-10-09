import traceback
import io
import os
import copy
import re
import json
import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from multiprocessing import cpu_count
from datasets import load_dataset
from tqdm import tqdm
import psutil

import torch
import transformers
from torch.utils.data import Dataset, IterableDataset
from datasets.iterable_dataset import IterableDataset
from transformers import Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import sys
import os
import random
from tqdm import tqdm
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)

from TPE-Llama.modeling_llama import LlamaForCausalLM


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.json'):
                fullname = os.path.join(root,f)
            yield fullname


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Question:\n{question}\n\n### Input:\n{input_seg}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/model/MiniCPM-2B-sft-bf16-llama-format")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_size: int = field(default=None, metadata={"help": "for calculate max steps."})
    gpu_size: int = field(default=None, metadata={"help": "for calculate max steps and for logging for calcuated intervel."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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

tok_example_count = 0

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)
        print(len(list_data_dict))
        global tok_example_count

        input_ids_all = []
        labels_all = []
        token_ids_all = []
        position_ids_all = []
        problematic_indices = []
        substart_all = []
        subend_all = []

        for idx, example in enumerate(list_data_dict):
            try:
                tok_example_count += 1
                if tok_example_count % 128 == 0:
                    logging.warning(f"tok_example_count: {tok_example_count}")
        
                # logging.warning("Formatting inputs...")
                prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
                source = prompt_input.format_map(example) if example.get("input_seg", "") != "" else prompt_no_input.format_map(example)
                target = f"{example['output']}"

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
                    problematic_indices.append(idx)
                    logging.error(f"table_array at index {idx}")
                    continue

                # add sep
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

                # Part I Encoded
                input_ids = [tokenizer.bos_token_id] + tokenizer.encode(text=part1, add_special_tokens=False)
                l_part1 = len(input_ids)
                tx = list(range(l_part1))
                ty = list(range(l_part1))
                
                px = list(range(l_part1))
                py = list(range(l_part1))

                substart = input_ids[-4:]

                
                # Table Encoded
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

                part3_target = part3 + target
                part3_target_en = tokenizer.encode(text=part3_target, add_special_tokens=False) + [tokenizer.eos_token_id]
                input_ids.extend(part3_target_en)
                
                tx_count = len(tx)
                ty_count = len(ty)
                assert tx_count == ty_count
                tx.extend(list(range(tx_count, tx_count + len(part3_target_en))))
                ty.extend(list(range(ty_count, ty_count + len(part3_target_en))))


                # Part III Encoded

                k_part3_end = k_part3_start + len(part3_target_en)
                px.extend(list(range(k_part3_start, k_part3_end)))
                py.extend(list(range(k_part3_start, k_part3_end)))

                subend = part3_target_en[:4]

                target_en = tokenizer.encode(text=target, add_special_tokens=False) + [tokenizer.eos_token_id]
                target_len = len(target_en)
                labels = copy.deepcopy(input_ids)
                labels[:-target_len] = [IGNORE_INDEX] * (len(input_ids) - target_len)


                if len(input_ids) > tokenizer.model_max_length:
                    problematic_indices.append(idx)
                    continue
                    input_ids = input_ids[-tokenizer.model_max_length:]
                    labels = labels[-tokenizer.model_max_length:]
                    px = px[-tokenizer.model_max_length:]
                    py = py[-tokenizer.model_max_length:]
                    tx = tx[-tokenizer.model_max_length:]
                    ty = ty[-tokenizer.model_max_length:]

                pi = np.concatenate([px, py])
                ti = np.concatenate([tx, ty])

                input_ids_all.append(torch.tensor(input_ids))
                labels_all.append(torch.tensor(labels))
                token_ids_all.append(torch.tensor(ti))
                position_ids_all.append(torch.tensor(pi))
                substart_all.append(torch.tensor(substart))
                subend_all.append(torch.tensor(subend))
            
            except Exception as e:
                problematic_indices.append(idx)
                logging.error(f"Error processing example at index {idx}: {str(e)}")
        
       

        print(len(input_ids_all))
        self.input_ids = input_ids_all
        self.labels = labels_all
        self.token_ids = token_ids_all
        self.position_ids = position_ids_all
        self.substart = substart_all
        self.subend = subend_all

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i], token_ids=self.token_ids[i], position_ids=self.position_ids[i], substart=self.substart[i], subend=self.subend[i])

    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # logging.warning(f"instances: {instances}")
        input_ids, labels, token_ids, position_ids, substart, subend = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "token_ids", "position_ids", "substart", "subend"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        substart = torch.nn.utils.rnn.pad_sequence(substart, batch_first=True, padding_value=IGNORE_INDEX)
        subend = torch.nn.utils.rnn.pad_sequence(subend, batch_first=True, padding_value=IGNORE_INDEX)
        # token_ids = torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True, padding_value=0)

        px_list = []
        py_list = []

        for pid in position_ids:
            s = len(pid) // 2
            px = pid[:s]
            py = pid[s:]
            px_list.append(px)
            py_list.append(py)

        px_padded = self.efficient_custom_pad_sequences(px_list)
        py_padded = self.efficient_custom_pad_sequences(py_list)
        position_ids = torch.cat((px_padded, py_padded), dim=-1)



        tx_list = []
        ty_list = []

        for tid in token_ids:
            s = len(tid) // 2
            tx = tid[:s]
            ty = tid[s:]
            tx_list.append(tx)
            ty_list.append(ty)

        tx_padded = self.efficient_custom_pad_sequences(tx_list)
        ty_padded = self.efficient_custom_pad_sequences(ty_list)

        token_ids = torch.cat((tx_padded, ty_padded), dim=-1)


        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            token_ids=token_ids,
            position_ids=position_ids,
            substart=substart, 
            subend=subend,
        )
    
    def efficient_custom_pad_sequences(self, sequence_list):
        tensors = [torch.tensor(seq) for seq in sequence_list]
        max_len = max(t.size(0) for t in tensors)

        pad_sizes = [max_len - t.size(0) for t in tensors]

        max_pad_size = max(pad_sizes)
        increment_ranges = torch.arange(1, max_pad_size + 1).unsqueeze(0)

        padded_tensors = []
        for tensor, pad_size in zip(tensors, pad_sizes):
            if pad_size > 0:
                padded_tensor = torch.cat([tensor, tensor[-1] + increment_ranges[:, :pad_size].squeeze(0)])
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        
        padded_tensor_batch = torch.stack(padded_tensors)

        return padded_tensor_batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    training_args.remove_unused_columns = False
    config.use_cache = False
   
    config._flash_attn_2_enabled = True
    config.output_loss = True
    config.pad_token_id = 0

    config.lamda = 1

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if training_args.low_rank_training:
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.01,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]

    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    logging.warning(f"data_module: {data_module}")
    
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()