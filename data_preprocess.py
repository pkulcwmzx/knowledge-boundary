import os
from tqdm import tqdm
from args import parse_args
from torch.utils.data import DataLoader
import torch
from utils_datafile import read_pair_file, read_input_file, get_output_file, to_jsonl, preprocess
from local_model import get_model_and_tokenizer, get_raw_embedding_table
from transformers import GPTJForCausalLM, AutoTokenizer

def data_preprocess(args):
    dataset_list = ['mmlu']
    model_list = ['gpt2-small', 'llama']  # gptj = gpt2, mistral = vicuna = llama

    tokenizers = []
    for model_id in model_list:
        if model_id == 'gpt2-small':
            path = "/dataA/PLM/gpt2/gpt2-small/"
        elif model_id == 'llama':
            path = "/dataA/PLM/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
        tokenizers.append(tokenizer)

    for data_name in dataset_list:
        if data_name == 'kass':
            dataset = read_input_file('./datasets/kass/triplets', './datasets/kass/entities', debug=False)
            output_path = "./datasets/kass/data.jsonl"
        elif data_name == 'rome':
            dataset = read_pair_file('./datasets/rome/counterfact.json', real_data=args.real_data, debug=False)
            output_path = "./datasets/rome/data_"+str(args.real_data)+".jsonl"
        elif data_name == 'mmlu':
            file_names = sorted(os.listdir('./datasets/mmlu/test'))
            dataset = []
            subjects = []
            for file in file_names:
                #q_subject = '_'.join(file.split('.')[0].split('_')[:-1])
                q_subject = file.split('.')[0]
                dataset_ = read_pair_file('./datasets/mmlu/test/', q_subject=q_subject, question_type=args.q_type,
                                     debug=False)
                dataset.append(dataset_)
                subjects.append(q_subject)
            output_path = "./datasets/mmlu/filtered_test/"
        elif data_name == 'alcuna':
            dataset = read_pair_file('./datasets/alcuna/id2question.json', q_subject=args.q_subject,
                                     question_type=args.q_type, debug=False)
            output_path = "./datasets/alcuna/data.jsonl"

        if data_name == 'mmlu':
            for dataset_, subject in zip(dataset, subjects):
                data = DataLoader(dataset=dataset_, batch_size=1, num_workers=args.num_workers)
                output_path_ = output_path + subject + ".jsonl"
                output_data = []
                idx = 0

                for qs, ans in tqdm(data):
                    data_dict = {}
                    questions = []
                    answers = []
                    for item in qs:
                        questions.append(item[0])
                    for item in ans:
                        answers.append(' ' + item[0])
                    for tokenizer in tokenizers:
                        questions = list(
                            filter(lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"]) <= args.max_len,
                                   questions))
                        answers = list(filter(
                            lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"]) <= args.max_ans_len,
                            answers))
                    if len(questions) > 0 and len(answers) > 0:
                        data_dict["idx"] = idx
                        idx += 1
                        data_dict["questions"] = questions
                        data_dict["answers"] = answers
                        output_data.append(data_dict)
                to_jsonl(output_data, output_path_, silence=True)

        else:
            data = DataLoader(dataset=dataset, batch_size=1, num_workers=args.num_workers)

            output_data = []
            idx = 0
            for qs, ans in tqdm(data):
                data_dict = {}
                questions = []
                answers = []
                for item in qs:
                    questions.append(item[0])
                for item in ans:
                    answers.append(' ' + item[0])
                for tokenizer in tokenizers:
                    questions = list(filter(lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"]) <= args.max_len,
                                            questions))
                    answers = list(filter(lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"]) <= args.max_ans_len,
                                            answers))
                if len(questions) > 0 and len(answers) > 0:
                    data_dict["idx"] = idx
                    idx += 1
                    data_dict["questions"] = questions
                    data_dict["answers"] = answers
                    output_data.append(data_dict)
            to_jsonl(output_data, output_path, silence=True)

if __name__ == '__main__':
    args = parse_args()
    data_preprocess(args)
