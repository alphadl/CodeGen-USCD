import torch
from transformers import LogitsProcessor
from typing import List, Union, Tuple, Set, Optional
import torch.nn.functional as F
import gzip
import json
import os

class EnsembleLogitsProcessor(LogitsProcessor):

    def __init__(self, filter_num:float, mean_num:float, num_beams: int, source_weights: List[float] = None, preserve_bos_token: bool = False):
        self.filter_num = filter_num
        self.mean_num = mean_num
        self.num_beams = num_beams
        self.source_weights = source_weights
        self.preserve_bos_token = preserve_bos_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.preserve_bos_token and cur_len <= 1:
            return scores
        scores = F.softmax(scores, dim=-1)
        std_number = torch.std(scores[0])
        mean_value = torch.mean(scores[0])

        if std_number.item() > self.mean_num:
            scores = torch.log(scores)
        else:
            mean_value = torch.mean(scores[0])
            filter_low = mean_value * torch.Tensor([self.filter_num]).to(scores.device)
            mask = scores[0] >= filter_low
            scores[0, ~mask] = torch.Tensor([0.00]).to(scores.device)
            scores[1, ~mask] = torch.Tensor([0.00]).to(scores.device)
            batch_size = int(input_ids.size(0) / self.num_beams)
            if self.source_weights is not None:
                assert len(self.source_weights) == batch_size
                source_weights = torch.Tensor(self.source_weights).to(scores.device)
            else:
                source_weights = 1/(batch_size-1) * torch.ones((batch_size,), device=scores.device)
            for i in range(self.num_beams):
                beam_indices = self.num_beams * torch.arange(batch_size, device=scores.device, dtype=torch.long) + i
                cands = scores[beam_indices]
                mean_scores = torch.log((source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands).sum(dim=0))
                for j in beam_indices:
                    scores[j] = mean_scores

        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=float('-inf'))

        return scores

def stream_jsonl(filename):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding='utf-8') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def data_handle(task_name):
    multi = ['humaneval-cpp', 'humaneval-cs', 'humaneval-d', 'humaneval-go', 'humaneval-java', 'humaneval-jl', 'humaneval-js', 'humaneval-lua', 'humaneval-php', 'humaneval-pl', 'humaneval-py', 'humaneval-r', 'humaneval-rb', 'humaneval-rkt', 'humaneval-rs', 'humaneval-scala', 'humaneval-sh', 'humaneval-swift', 'humaneval-ts']
    if task_name == "humaneval":
        input_file_neg = "./CodeGen-USCD/data/humaneval/human_eval_neg.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_name == "humaneval-unstripped":
        input_file_neg = "./CodeGen-USCD/data/humaneval/human_eval_neg.jsonl"
        data_neg = list(sample_neg["prompt"] for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_name in multi:
        input_file_neg = os.path.join(f"./CodeGen-USCD/data/data_MultiPL-E_neg/{task_name}", f"{task_name}-neg.jsonl")
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    return data_neg

def data_example(task_neg):
    if task_neg == 1:
        input_file_neg = "./CodeGen-USCD/data/humaneval/humaneval_reduce1.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg == 2:
        input_file_neg = "./CodeGen-USCD/data/humaneval/humaneval_reduce2.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg == 3:
        input_file_neg = "./CodeGen-USCD/data/humaneval/humaneval_reduce3.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg == 4:
        input_file_neg = "./CodeGen-USCD/data/humaneval/humaneval_reduce4.jsonl"
        data_neg = list(sample_neg["prompt"].strip() for idx_neg, sample_neg in enumerate(stream_jsonl(input_file_neg)))
    elif task_neg > 5:
        data_neg = None
    return data_neg
          
    