from torch.utils.data import Dataset
import torch
import numpy as np
import json
import random


def get_examples(data_path, split_tuples):
    inputs, targets = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        for line in fp:
            line = line.strip()
            if line != '':
                sent, tuples = line.split('####')
                if split_tuples:
                    tuples = tuples.split('||||')
                    for tuple in tuples:
                        inputs.append(sent)
                        targets.append(tuple)
                else:
                    inputs.append(sent)
                    targets.append(tuples)
    return inputs, targets


class MyDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, split_tuples, cls_weight, max_len=128):
        self.data_path = f'data/{task}/{data_dir}/{data_type}.txt'
        self.task = task
        self.split_tuples = split_tuples
        self.cls_weight = cls_weight
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.data_type = data_type
        self.loss_weights = []
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        loss_weight = self.loss_weights[index].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "loss_weight": loss_weight}

    def _build_examples(self):
        inputs, targets = get_examples(self.data_path, split_tuples=self.split_tuples)
        for i in range(len(inputs)):
            input = inputs[i]
            target = targets[i]
            tokenized_input = self.tokenizer.batch_encode_plus([input], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt", )
            tokenized_target = self.tokenizer.batch_encode_plus([target], max_length=self.max_len, pad_to_max_length=True, truncation=True, return_tensors="pt")
            loss_weight = get_loss_weight(tokenized_target['attention_mask'], target.endswith('true'), cls_weight=self.cls_weight)
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)
            self.loss_weights.append(loss_weight)


def get_loss_weight(attention_mask, if_true_example, cls_weight):
    attention_mask = attention_mask.tolist()[0]
    k = np.where(attention_mask)[0].max()
    if if_true_example:
        loss_weight = attention_mask
    else:
        loss_weight = [1 if i in (k-1, k) else 0 for i in range(len(attention_mask))]
    loss_weight = torch.tensor([loss_weight], dtype=torch.float)
    return loss_weight


def get_constrained_decoding_token_ids(tokenizer, task_config):
    tokens = ['true', 'false', '|']
    for element in task_config['elements']:
        if element != 'extraction':
            tokens += task_config[element]
    if 'null_extraction' in task_config:
        tokens.append(task_config['null_extraction'])
    tokens = ' '.join(tokens)
    ids = tokenizer.encode(tokens)[:-1]     # not including the eos token
    return list(set(ids))


def get_aug_data(sents, targets, raw_outputs, task_config):
    aug_data = []
    for sent, target, raw_output in zip(sents, targets, raw_outputs):
        aug_target = []
        # true examples
        if target == '':
            true_target = []
        else:
            true_target = target.split('||||')
        aug_target += true_target
        # false examples from beam search (D2)
        for prob, seq in raw_output:
            if seq not in true_target and seq.endswith('true'):
                aug_target.append(seq[:-4] + 'false')
        # false examples from replacing (D1)
        aug_target += get_replaced_false_targets(true_target, task_config)
        # remove dup
        aug_target = list(set(aug_target))
        aug_target = '||||'.join(aug_target)
        aug_data.append(sent + '####' + aug_target)
    return aug_data


def get_replaced_false_targets(true_target, task_config):
    elements = task_config['elements']
    n = len(elements)
    cands = []
    for i in range(n):
        if true_target==[]:
            cand=[]
        else:
            cand = set([t.split(' | ')[i] for t in true_target])
        cands.append(cand)
    false_target = [[]]
    for i in range(n):
        cand = cands[i]
        false_target = [lst + [x] for x in cand for lst in false_target]
    false_target = [' | '.join(lst) for lst in false_target]
    false_target  = [s + ' | false' for s in false_target if s + ' | true' not in true_target]
    return false_target


def get_task_config(task_config_path, task, dataset):
    json_data = json.load(open(task_config_path, 'r'))[task]
    config = {}
    config['elements'] = json_data['elements']
    config['task'] = task
    config['dataset'] = dataset
    if 'null_extraction' in json_data:
        config['null_extraction'] = json_data['null_extraction']
    cls_keys = [key for key in json_data['elements'] if key != 'extraction']
    for key in cls_keys:
        d = json_data[key]
        if 'all_datasets' in d:
            config[key] = d['all_datasets']
        elif dataset in d:
            config[key] = d[dataset]
        else:
            raise NotImplementedError
    return config


if __name__ == "__main__":
    task_config = get_task_config('task_config.json', 'jer', 'conll04')
    false_target = get_replaced_false_targets(['a1 | o2 | positive | true', 'a2 | o1 | negative | true'], task_config)
    from transformers.models.t5.tokenization_t5 import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    token_ids = get_constrained_decoding_token_ids(tokenizer, task_config)
    inputs, targets = get_examples('data/jer/conll04/test.txt', split_tuples=False)
    print()
