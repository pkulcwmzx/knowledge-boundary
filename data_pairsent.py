import random
import json
import os
import copy
from torch.utils.data import Dataset

class pairsent:
    def __init__(self, query, answer):
        self.query = query
        self.answer = answer


class pairsent_database(Dataset):
    def __init__(self, pair_path):
        #super.__init__(ent_path)
        self.pairsents = []
        self.pair_path = pair_path
    def add_item(self, query, answer):
        new_pairsent = pairsent(query, answer)
        self.pairsents.append(new_pairsent)

    def __len__(self):
        return len(self.pairsents)


    def __getitem__(self, idx):
        pairsent = self.pairsents[idx]
        query = pairsent.query
        answer = pairsent.answer
        return query, answer



    def shot_getitem(self, shot_type='rel', shot_num=5):
        for pairsent in self.pairsents:
            query = pairsent.query
            answer = pairsent.answer
            shot_pairsents = random.sample(self.pairsents, shot_num)
            shot_statements = []
            for shot_pairsent in shot_pairsents:
                shot_statements.append('Question: '+shot_pairsent.query+'Answer: '+shot_pairsent.answer)
            shot_statements = '\t'.join(shot_statements)
            query = [shot_statements + '\t' + one_query for one_query in query]
            yield query, answer

                
                
