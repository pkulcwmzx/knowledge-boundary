import random
import json
import os
import copy
from torch.utils.data import Dataset

class triplet:
    def __init__(self, sbj, rel, obj):
        self.sbj = sbj
        self.rel = rel
        self.obj = obj

    def make_sent(self, ent_dict):
        pass
"""
class triplet_database:
    def __init__(self, ent_path):
        self.triplets = []
        sbj_file = "allsub2alias.json"
        obj_file = "allobj2alias.json"
        rel_file = "relation2template.json"

        with open(os.path.join(ent_path, sbj_file), 'r', encoding='utf-8') as f:
            self.sbj_dict = json.loads(f.read())
        with open(os.path.join(ent_path, obj_file), 'r', encoding='utf-8') as f:
            self.obj_dict = json.loads(f.read())
        with open(os.path.join(ent_path, rel_file), 'r', encoding='utf-8') as f:
            self.rel_dict = json.loads(f.read())
    def add_item(self, sbj, rel, obj):
        new_triplet = triplet(sbj, rel, obj)
        self.triplets.append(new_triplet)

    def make_sents(self, idx, item_size):
        triplet = self.triplets[idx]
        sbj, rel, obj = triplet.sbj, triplet.rel, triplet.obj
        templates = self.rel_dict[rel]
        sbjs = self.sbj_dict[sbj]
        objs = self.obj_dict[obj]
        questions = []
        answers = []
        for i in range(item_size):
            sbj = random.choice(sbjs)
            template = random.choice(templates)
            obj = random.choice(objs)

            question = template.replace("[X]", sbj).replace("[Y]", "").rstrip()
            ans = obj
            questions.append(question)
            answers.append(ans)

        return questions, answers
"""
class triplet_database(Dataset):
    def __init__(self, ent_path):
        #super.__init__(ent_path)
        self.triplets = []
        self.sub2relobj = {}
        self.rel2subobj = {}
        sbj_file = "allsub2alias.json"
        obj_file = "allobj2alias.json"
        rel_file = "relation2template.json"

        with open(os.path.join(ent_path, sbj_file), 'r', encoding='utf-8') as f:
            self.sbj_dict = json.loads(f.read())
        with open(os.path.join(ent_path, obj_file), 'r', encoding='utf-8') as f:
            self.obj_dict = json.loads(f.read())
        with open(os.path.join(ent_path, rel_file), 'r', encoding='utf-8') as f:
            self.rel_dict = json.loads(f.read())

    def add_item(self, sbj, rel, obj):

        if len(self.sbj_dict[sbj]) == 0 or len(self.obj_dict[obj]) == 0 or len(self.rel_dict[rel]) == 0:
            pass
        else:
            new_triplet = triplet(sbj, rel, obj)
            self.triplets.append(new_triplet)
            self.sub2relobj.setdefault(sbj, []).append((rel, obj))
            self.rel2subobj.setdefault(rel, []).append((sbj, obj))

    def __len__(self):
        return len(self.triplets)

    def filltemplate(self, sbjs, rel, objs):
        questions = []
        templates = self.rel_dict[rel]
        for template in templates:
            for sbj in sbjs:
                question = template.replace("[X]", sbj).replace("[Y]", "").rstrip().rstrip('.').rstrip()
                question = question.split()[0].capitalize() + ' ' + ' '.join(question.split()[1:])
                questions.append(question)
        return questions

    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        sbj, rel, obj = triplet.sbj, triplet.rel, triplet.obj
        sbjs = self.sbj_dict[sbj]
        objs = self.obj_dict[obj]
        questions = self.filltemplate(sbjs, rel, objs)
        return questions, objs
    
    def get_shot_triplets(self, shot_type, mykey, myvalue, shot_num=5):
        all_values = self.sub2relobj.get(mykey, []) if shot_type == 'sub' else self.rel2subobj.get(mykey, [])
        all_values = copy.deepcopy(all_values)
        all_values.remove(myvalue)
        random.shuffle(all_values)
        all_values = all_values[:shot_num]
        shot_triplets = [(mykey, rel, obj) for rel, obj in all_values] if shot_type == 'sub' else [(sbj, mykey, obj) for sbj, obj in all_values]
        if len(shot_triplets) < shot_num:
            shot_triplets = [(triplet.sbj, triplet.rel, triplet.obj) for triplet in random.sample(self.triplets, shot_num-len(shot_triplets))] \
                  + shot_triplets
        return shot_triplets


    def shot_getitem(self, shot_type='rel', shot_num=5):
        for triplet in self.triplets:
            sbj, rel, obj = triplet.sbj, triplet.rel, triplet.obj
            sbjs = self.sbj_dict[sbj]
            objs = self.obj_dict[obj]
            questions = self.filltemplate(sbjs, rel, objs)
            if shot_type == 'rel':
                shot_triplets = self.get_shot_triplets('rel', rel, (sbj, obj), shot_num)
            elif shot_type == 'sub':
                shot_triplets = self.get_shot_triplets('sub', sbj, (rel, obj), shot_num)

            shot_statements = []
            for shot_sbj, shot_rel, shot_obj in shot_triplets:
                shot_sbjs = self.sbj_dict[shot_sbj][:1]
                shot_objs = self.obj_dict[shot_obj][:1]
                shot_statements.append(self.filltemplate(shot_sbjs, shot_rel, shot_objs)[0]+' '+shot_objs[0]+'.')
            shot_statements = '\t'.join(shot_statements)
            questions = [shot_statements + '\t' + question for question in questions]
            yield questions, objs
                
                
