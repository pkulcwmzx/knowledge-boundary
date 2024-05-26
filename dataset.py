import random
import json
import os
import copy
from torch.utils.data import Dataset

class Knowledgedata(Dataset):
    def __init__(self, dataset, real_data=False, q_subject=None):
        self.data = []
        self.dataset = dataset
        self.sub2examples = {}
        self.rel2examples = {}
        self.all_knowledge = []
        if dataset == "kass":
            path = "./datasets/kass/data.jsonl"
        elif dataset == "rome":
            if real_data:
                path = "./datasets/rome/data_1.jsonl"
            else:
                path = "./datasets/rome/data_0.jsonl"
        elif dataset == "alcuna":
            path = "./datasets/alcuna/data.jsonl"
        elif dataset == "mmlu":
            path = "./datasets/mmlu/filtered_test/" + f'{q_subject}' + ".jsonl"

        with open(path, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                json_data = json.loads(line)
                self.data.append(json_data)
                one_knowledge = random.choice(json_data['questions'])+random.choice(json_data['answers'])
                self.all_knowledge.append(one_knowledge)
                if 'relation' not in json_data:
                    continue
                if json_data["relation"][0] not in self.rel2examples:
                    self.rel2examples[json_data["relation"][0]] = []
                self.rel2examples[json_data["relation"][0]].append((one_knowledge, json_data['object'][0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["idx"], self.data[idx]["questions"], self.data[idx]["answers"]

    def get_random_example(self, num):
        return random.choices(self.all_knowledge, k=num)

    def get_same_examples(self, shot_type, subject=None, relation=None, object=None, num=0):
        examples = []
        if self.dataset == 'alcuna':
            return '\t'.join(self.get_random_example(num))
        from_dict = self.sub2examples[subject] if shot_type == "subject" else self.rel2examples[relation]
        for example in from_dict:
            if example[1] != object:
                examples.append(example[0])
            if len(examples) == num:
                break
        if len(examples) < num:
            examples += self.get_random_example(num-len(examples))
        return '\t'.join(examples)

    def shot_getitem(self, shot_type, shot_num):
        for json_data in self.data:
            if self.dataset == "alcuna":
                yield json_data["idx"], json_data["questions"], json_data["answers"], self.get_same_examples(shot_type, num=shot_num)
            else:
                yield json_data["idx"], json_data["questions"], json_data["answers"], self.get_same_examples(shot_type, subject=None, relation=json_data["relation"][0], object=json_data["object"][0], num=shot_num)