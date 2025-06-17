from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import time
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from datasets import (
        Dataset,
        DatasetDict,
        Features,
        Sequence,
        Value,
        concatenate_datasets,
        load_dataset,
    )
except ImportError:
    warnings.warn("Datasets not installed, you'll be unable to use these dataset processing functions.")

# Import SFT processing functions for backward compatibility


def clm_process(
    raw_dataset: "Dataset",
    tokenizer,
    text_column_name: str,
    dataset_processing_num_proc_per_process: int,
    dataset_overwrite_cache: bool,
    sequence_length: int,
    streaming: bool = False,
    add_eos_token: bool = False,
):
    """
    Concatenate all texts from raw_dataset and generate chunks of `sequence_length + 1`,
    where chunks overlap by a single token.

    Args:
        raw_dataset: Dataset containing raw text
        tokenizer: HuggingFace tokenizer
        text_column_name: Name of the column containing text data
        dataset_processing_num_proc_per_process: Number of processes for parallelization
        dataset_overwrite_cache: Whether to overwrite the cache
        sequence_length: Maximum sequence length

    Returns:
        Processed dataset with tokenized sequences
    """    
    # Adapted from https://github.com/huggingface/transformers/blob/47e1676255e5dd86b9541f734cd4f4bdcbb50f4a/examples/pytorch/language-modeling/run_clm.py#L391-L439
    def group_texts(examples: Dict[str, List[np.ndarray]]) -> Dict[str, List[np.ndarray]]:
        # Concatenate all texts.
        concatenated_examples = {k: np.concatenate(v) for k, v in examples.items()}
        total_length = len(concatenated_examples[next(iter(examples.keys()))])
        # WARNING: We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= sequence_length + 1:
            total_length = ((total_length - 1) // sequence_length) * sequence_length + 1
        # Split by chunks of sequence_length.
        result = {
            k: [
                t[i : i + sequence_length + 1] for i in range(0, total_length - (sequence_length + 1), sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    def _tokenize_and_group_texts(texts: List[str]) -> Dict[str, List[np.ndarray]]:
        if add_eos_token:
            texts = [i+tokenizer.eos_token for i in texts]
            tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
        else:
            tokenized_batch = tokenizer.batch_encode_plus(texts, return_attention_mask=False, return_token_type_ids=False)
            # assert tokenizer.eos_token_id in tokenized_batch.input_ids[0]
        tokenized_batch = {k: [np.array(tokenized_texts) for tokenized_texts in v] for k, v in tokenized_batch.items()}
        return group_texts(tokenized_batch)
        
    if streaming:
        train_dataset = raw_dataset.map(
            _tokenize_and_group_texts,
            input_columns=text_column_name,
            remove_columns=raw_dataset.column_names,
            features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
            batched=True)
    else:
        train_dataset = raw_dataset.map(
            _tokenize_and_group_texts,
            input_columns=text_column_name,
            remove_columns=raw_dataset.column_names,
            features=Features({"input_ids": Sequence(feature=Value(dtype="int64"), length=sequence_length + 1)}),
            batched=True,
            num_proc=dataset_processing_num_proc_per_process,
            load_from_cache_file=not dataset_overwrite_cache,
            desc=f"Grouping texts in chunks of {sequence_length+1}",
        )
    return train_dataset


def is_add_eos_token(tokenizer):
    sample = '안녕하세요'
    tokenized = tokenizer.tokenize(sample, add_special_tokens=True)
    if tokenizer.eos_token not in tokenized:
        return True
    return False

def read_jsonl(file_path):
  import json
  with open(file_path, 'r', encoding='utf-8') as f:
      data = [json.loads(line) for line in f]
  return data

def save_jsonl(file_path, data, append_mode = False, batch = 1):
    
  import json
  """
  data = [
      {"name": "Alice", "age": 30},
      {"name": "Bob", "age": 25}
  ]
  """
  append = 'a' if append_mode else 'w'
  with open(file_path, append, encoding='utf-8') as f:
      if type(data)==list:
          for start_idx in range(0, len(data), batch):
              batch = data[start_idx:start_idx+batch]
              f.write("\n".join([json.dumps(row, ensure_ascii=False) for row in batch]) + "\n")
      else:
          f.write(json.dumps(data, ensure_ascii=False) + '\n')


# raw_dataset = load_dataset(data_name, streaming=True, split='train')
# add_eos_token = is_add_eos_token(tokenizer)
# print(add_eos_token)
# test = tokenize_process(raw_dataset, tokenizer, 'text', 1, True, True, add_eos_token)
## streaming with map styled 

if __name__ == '__main__':
    # use name="sample-10BT" to use the 10BT sample
    # data = load_dataset("HuggingFaceFW/fineweb-2", name="kor_Hang")
    # ./data/HuggingFaceFW___fineweb/sample-100BT/0.0.0/0f039043b23fe1d4eed300b504aa4b4a68f1c7ba
    data_name = "../data/fineweb-100BT"
    # data = load_dataset(, split='train')
    output_path = '../grouped_dataset/fineweb-100BT/train-2'
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    STREAMING = True
    SPLIT = 'train'
    BATCH = 10000000
    
    NUM_PROC = 1 #int(os.cpu_count()*0.5)
    SKIP = 7
    now = time.time()
    if STREAMING:
        raw_dataset = load_dataset(data_name, streaming=True, split=SPLIT)
        tmp = []
        tmp_dataset = None
        pbar = tqdm()
        cnt = 0
        for i in raw_dataset:
            if len(tmp) == BATCH:
                # break
                if cnt>SKIP:
                    tmp_dataset = Dataset.from_list(tmp)
                    tmp_dataset = clm_process(tmp_dataset, tokenizer, 'text', NUM_PROC, False, 2048, False, True)
                    # XXX NUMPY 파일로 저장. NP.MEMMAP
                    save_path = os.path.join(output_path, str(cnt))
                    tmp_dataset.save_to_disk(save_path)
                    cnt+=1
                    tmp = []
                    tmp_dataset.cleanup_cache_files()
                    tmp_dataset = None
                else:
                    cnt+=1
                    tmp = []
                    if tmp_dataset is not None:
                        tmp_dataset.cleanup_cache_files()
                    tmp_dataset = None
                # reset
            tmp.append(i)
            pbar.update(1)
        pbar.close()
        # last for test
        if tmp_dataset is None:
            tmp_dataset = Dataset.from_list(tmp)
            tmp_dataset = clm_process(tmp_dataset, tokenizer, 'text', NUM_PROC, False, 2048, False, True)
            save_path = os.path.join(output_path, str(cnt))
            tmp_dataset.save_to_disk(save_path, num_proc=NUM_PROC)
        
    else:
        print('reading........')
        try:
            raw_dataset = load_dataset(data_name, split=SPLIT)
        except:
            from datasets import load_from_disk
            raw_dataset = load_from_disk(data_name)
            print(raw_dataset)

        print('reading is done.........')
        try:
            tmp_dataset = clm_process(raw_dataset[SPLIT], tokenizer, 'text', NUM_PROC, False, 2048, False, True)
        except:
            tmp_dataset = clm_process(raw_dataset, tokenizer, 'text', NUM_PROC, False, 2048, False, True)
        #raw_dataset.cleanup_cache_files()
        save_path = output_path 
        tmp_dataset.save_to_disk(save_path)
    print(time.time()-now)
