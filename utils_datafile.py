from data_triplet import triplet_database
from data_pairsent import pairsent_database
from utils_helper import get_str_time, NpEncoder
import os
import json
import random
import csv
import copy
import torch

def read_input_file(triplet_path, ent_path, debug=False):
    data = triplet_database(ent_path)
    file_names = sorted(os.listdir(triplet_path))
    if debug:
        file_names = random.choices(file_names, k=20)
    for file in file_names:
        file_path = os.path.join(triplet_path, file)
        with open(file_path, 'r') as f:
            content = f.read()
            content = content.split('\n')[:-1]
            content = list(set(content))
            if debug:
                content = random.choices(content, k=5)
            for item in content:
                triplet_item = item.split('\t')
                assert len(triplet_item) == 3
                sbj, rel, obj = triplet_item[0], triplet_item[1], triplet_item[2]
                assert file.split('.')[0] == rel
                data.add_item(sbj, rel, obj)

    return data


def rome2pairs(jsonpath, real_data=False):
    with open(jsonpath, 'r') as f:
        data = json.load(f)
    pairs = []
    for item in data:
        query = []
        answer = []

        for prompt in item['paraphrase_prompts']:
            query.append(prompt)
        query.append(item['requested_rewrite']['prompt'].format(item['requested_rewrite']['subject']))
        rels = item['requested_rewrite']['relation_id']

        if real_data:
            answer.append(item['requested_rewrite']['target_true']['str'])
            objs = item['requested_rewrite']['target_true']['id']
        else:
            answer.append(item['requested_rewrite']['target_new']['str'])
            objs = item['requested_rewrite']['target_new']['id']
        pairs.append([query, answer, rels, objs])
    return pairs


def alcuna2pairs(jsonpath, question_type):
    with open(jsonpath, 'r') as f:
        data = json.load(f)
    pairs = []
    for item in data.values():
        # print(item)
        for question in item:
            if question['form'] == question_type:
                query = []
                answer = []
                query.append(question['question'])
                if question_type == 'multi-choice':
                    answer = [str(a) for a in question['answers']]
                else:
                    answer = question['answers']
                pairs.append((query, answer))
    return pairs


def mmlu2pairs(csv_dir, q_subject, question_type):
    pairs = []
    for file in os.listdir(csv_dir):
        if file.endswith(f'{q_subject}.csv'):
            with open(os.path.join(csv_dir, file), 'r') as f:
                content = csv.reader(f)
                content = [line for line in content]
                for line in content:
                    query = []
                    answer = []
                    if question_type == 'choice':
                        prompt = line[0]
                        choices = 'A. ' + line[1] + ' B. ' + line[2] + ' C. ' + line[3] + ' D. ' + line[4]
                        # join them
                        query.append(prompt + ' ' + choices)
                        answer.append(line[5])
                    elif question_type == 'completion':
                        prompt = line[0]
                        query.append(prompt)
                        correct_answer = line[ord(line[5]) - ord('A') + 1]
                        answer.append(correct_answer)
                    pairs.append((query, answer))
    return pairs


def read_pair_file(pair_path, q_subject=None, real_data=None, question_type=None, debug=False):
    data = pairsent_database(pair_path)
    if 'rome' in pair_path:
        pairs = rome2pairs(pair_path, real_data)
    elif 'alcuna' in pair_path:
        pairs = alcuna2pairs(pair_path, question_type)
    elif 'mmlu' in pair_path:
        pairs = mmlu2pairs(pair_path, q_subject, question_type)
    else:
        raise NotImplementedError
    if debug:
        pairs = random.choices(pairs, k=20)
    for pair in pairs:
        query = pair[0]
        answer = pair[1]
        rel = pair[2]
        obj = pair[3]
        data.add_item(query, answer, rel, obj)
    return data

def get_output_file(args, name, output_dir='outputs', file_type='jsonl'):
    datetime_str = get_str_time()
    file_name = name + "_model_" + str(args.model_id) + '_data_' + str(args.dataset) + "_seed_" + str(args.seed) + \
                "_lr_" + str(args.lr) + "_semantic_weight_" + str(args.semantic_weight) + "_reg_weight" + str(args.regular_weight) + \
                '_time_'  + datetime_str
    return os.path.join(output_dir, f'{file_name}.{file_type}')

def preprocess_data(args, qs, ans, tokenizer):
    questions = []
    answers = []
    for item in qs:
        questions.append(item[0])
    for item in ans:
        answers.append(' ' + item[0])
    if len(questions) > args.question_num:
        questions = random.sample(questions, args.question_num)

    tokenized_questions = tokenizer(questions)["input_ids"]
    tokenized_ans = tokenizer(answers)["input_ids"]
    for i in range(len(tokenized_questions)):
        tokenized_questions[i] = [tokenizer.bos_token_id] + tokenized_questions[i]

    tokenized_questions_eos = copy.deepcopy(tokenized_questions)
    for i in range(len(tokenized_questions_eos)):
        tokenized_questions_eos[i] = tokenized_questions_eos[i] + [tokenizer.eos_token_id]

    eos_question_length = [len(tokenized_questions_eos[i]) for i in range(len(tokenized_questions_eos))]
    max_q_len = max(eos_question_length)
    for i in range(len(tokenized_questions_eos)):
        tokenized_questions_eos[i] = tokenized_questions_eos[i] + (max_q_len - len(tokenized_questions_eos[i])) * [
            tokenizer.pad_token_id]

    question_length = [len(tokenized_questions[i]) for i in range(len(tokenized_questions))]
    answer_length = [len(tokenized_ans[i]) for i in range(len(tokenized_ans))]
    input_data = []
    for question in tokenized_questions:
        for answer in tokenized_ans:
            sentence = question + answer
            input_data.append(sentence)

    answer_num = len(answers)
    question_num = len(questions)
    input_question_len = []
    for i in range(question_num):
        input_question_len += answer_num * [question_length[i]]
    input_answer_len = question_num * answer_length
    input_len = [len(input_data[i]) for i in range(len(input_data))]
    assert len(input_question_len) == len(input_answer_len) == len(input_data)
    for i in range(len(input_data)):
        assert input_len[i] == input_question_len[i] + input_answer_len[i]
    max_len = max(input_len)

    for i in range(len(input_data)):
        input_data[i] = input_data[i] + (max_len - len(input_data[i])) * [tokenizer.pad_token_id]
    return questions, answers, input_data, input_question_len, input_answer_len, tokenized_questions, tokenized_ans, max_len, answer_num

def preprocess(args, qs, ans, tokenizer):
    questions = []
    answers = []
    for item in qs:
        questions.append(item[0])
    for item in ans:
        answers.append(item[0])
    tokenized_questions = tokenizer(questions, add_special_tokens=False, padding=True)

    input_ids = tokenized_questions["input_ids"]
    attention_mask = tokenized_questions["attention_mask"]

    tokenized_ans = tokenizer(answers)["input_ids"]

    if args.model_id == "llama" or args.model_id == "alpaca" or args.model_id == "vicuna" or args.model_id == "mistral":
        for i in range(len(tokenized_ans)):
            tokenized_ans[i] = tokenized_ans[i][2:]
    # every args.question_num store in one list
    new_input_ids = []
    new_attention_mask = []
    # new_tokenized_ans = []
    for i in range(0, len(input_ids), args.question_num):
        new_input_ids.append(input_ids[i:i + args.question_num])
        new_attention_mask.append(attention_mask[i:i + args.question_num])
        # new_tokenized_ans.append(tokenized_ans[i:i + args.question_num])
    return new_input_ids, new_attention_mask, tokenized_ans


def get_label(args, full_embeddings, input_data, input_question_len, input_answer_len, max_len):
    label_list = []
    for i in range(full_embeddings.shape[0]):
        tmp_label = [-100] * max_len
        tmp_label[input_question_len[i]:input_question_len[i] + input_answer_len[i]] = input_data[i][
                                                                                       input_question_len[i]:
                                                                                       input_question_len[i] +
                                                                                       input_answer_len[i]]
        label_list.append(tmp_label)
    labels = torch.Tensor(label_list).long().to(args.device)

    return labels


def to_jsonl(dicts, save_file, silence=False):
    if not os.path.isdir(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with open(save_file, 'w') as f:
        for line_dict in dicts:
            if not silence:
                print(line_dict)
            jsonl_line = f'{json.dumps(line_dict, cls=NpEncoder)}\n'
            f.write(jsonl_line)
