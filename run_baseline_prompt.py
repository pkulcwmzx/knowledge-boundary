from vllm import LLM, SamplingParams
from utils_datafile import to_jsonl
from dataset import Knowledgedata
from tqdm import tqdm
import random
import os
from utils_helper import set_seed
model2path = {
    "gpt2": "/data2/pretrain/gpt2",
    "llama2": "/data2/pretrain/llama2/Llama-2-7b-chat-hf",
    'gptj': '/data2/pretrain/EleutherAI/gpt-j-6B',
    'mistral': '/data2/pretrain/Mistral-7B-Instruct-v0.2',
    'vicuna': '/data2/pretrain/vicuna/vicuna-7B'
}



def get_discriminate(prompts, answer):
    discriminate_prompt = list(map(lambda x: f'Check whether the following statement is correct.\n{x+answer}\nThe statement is (True/False): ', prompts))
    return discriminate_prompt

def run_discriminate_prompt(llm, sampling_params, dataset, random_one=False, outputfile=None):
    overall_success = 0
    overall_test = 0
    results_dicts = []
    for idx, prompts, answers in tqdm(dataset):
        results_dict = {}
        overall_test += 1
        if random_one:
            prompts = random.choices(prompts, k=1)
        prompts = get_discriminate(prompts, random.choice(answers))
        answers = ['True']
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        prompt, ori_a = "", ""
        success = -1
        for output in outputs:
            is_success = False
            prompt = output.prompt
            answer = output.outputs[0].text
            for ori_a in answers:
                # if answer.strip().startswith(ori_a):
                if ori_a.lower() in answer.strip().lower() and 'false' not in answer.strip().lower():
                    is_success = True
                    break
            if is_success:
                overall_success += 1
                success = 1
                break
        results_dict['idx'] = idx
        results_dict["success"] = success
        results_dict["prompt"] = prompt
        results_dict["ans"] = answer
        results_dict["original question"] = prompt
        results_dict["original answer"] = ori_a
        results_dicts.append(results_dict)
        # print("Saving...")
        if overall_test % 100 == 0:
            to_jsonl(results_dicts, outputfile, True)
    results_dicts.append({"success": overall_success / overall_test})
    to_jsonl(results_dicts, outputfile, True)
    # print("success rate: {}".format(overall_success / overall_test))




def zero_shot(llm, sampling_params, dataset, random_one=False, outputfile=None):
    overall_success = 0
    overall_test = 0
    results_dicts = []
    for idx, prompts, answers in tqdm(dataset):
        results_dict = {}
        overall_test += 1
        if random_one:
            prompts = random.choices(prompts, k=1)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        prompt, ori_a = "", ""
        success = -1
        for output in outputs:
            is_success = False
            prompt = output.prompt
            answer = output.outputs[0].text
            for ori_a in answers:
                # if answer.strip().startswith(ori_a):
                if ori_a.lower() in answer.strip().lower():
                    is_success = True
                    break
            if is_success:
                overall_success += 1
                success = 1
                break
        results_dict['idx'] = idx
        results_dict["success"] = success
        results_dict["prompt"] = prompt
        results_dict["ans"] = answer
        results_dict["original question"] = prompt
        results_dict["original answer"] = ori_a
        results_dicts.append(results_dict)
        # print("Saving...")
        if overall_test % 100 == 0:
            to_jsonl(results_dicts, outputfile, True)
    results_dicts.append({"success": overall_success / overall_test})
    to_jsonl(results_dicts, outputfile, True)

def few_shot(llm, sampling_params, dataset, shot_type='sub', shot_num=4, random_one=False, outputfile=None):
    overall_success = 0
    overall_test = 0
    results_dicts = []
    # outputfile = f'outputs/{model}_few_shot_same{shot_type}_{shot_num}!shot_one_{str(random_one)}.jsonl'
    outputfile = outputfile.split('.')[0] + f'_{shot_type}_{shot_num}.jsonl'
    for idx, prompts, answers, examples in tqdm(dataset.shot_getitem(shot_type=shot_type, shot_num=shot_num)):
        results_dict = {}
        overall_test += 1
        if random_one:
            prompts = random.choices(prompts, k=1)
        prompts = [examples+'\t'+ prompt for prompt in prompts]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        prompt, ori_a = "", ""
        success = -1
        for output in outputs:
            is_success = False
            prompt = output.prompt
            answer = output.outputs[0].text
            for ori_a in answers:
                if ori_a.lower() in answer.strip().lower():
                    is_success = True
                    break
            if is_success:
                overall_success += 1
                success = 1
                break
        results_dict['idx'] = idx
        results_dict["success"] = success
        results_dict["prompt"] = prompt
        results_dict["ans"] = answer
        results_dict["original question"] = prompt
        results_dict["original answer"] = ori_a
        results_dicts.append(results_dict)
        # print("Saving...")
        if overall_test % 100 == 0:
            to_jsonl(results_dicts, outputfile, True)
    results_dicts.append({"success": overall_success / overall_test})
    to_jsonl(results_dicts, outputfile, True)

def answer_in_question(dataset):
    overall_success = 0
    overall_test = 0
    results_dicts = []
    for prompts, answers in tqdm(dataset):
        results_dict = {}
        overall_test += 1
        outputs = prompts
        prompt, ori_a = "", ""
        success = -1
        for output in outputs:
            is_success = False
            prompt = output
            answer = output
            for ori_a in answers:
                if answer.strip().startswith(ori_a):
                    is_success = True
                    break
            if is_success:
                overall_success += 1
                success = 1
                break
        results_dict["success"] = success
        results_dict["prompt"] = prompt
        results_dict["ans"] = answer
        results_dict["original question"] = prompt
        results_dict["original answer"] = ori_a
        results_dicts.append(results_dict)
        # print("Saving...")
        to_jsonl(results_dicts, 'outputs/answer_in_question.jsonl', True)
    to_jsonl(results_dicts, 'outputs/answer_in_question.jsonl', True)
    print("success rate: {}".format(overall_success / overall_test))

if __name__ == "__main__":

    models = ['llama2']
    test_sets = ['kass', 'alcuna', 'rome']

    real_datas = ['0', '1']
    q_subject = 'bs'
    methods = ['few_shot']
    for model in models:
        llm = LLM(model=model2path[model])
        sampling_params = SamplingParams(temperature=0.0, top_p=0.1, max_tokens=10)
        for test_set in test_sets:
            for real_data in real_datas:
                if test_set != 'rome' and real_data == '1':
                    continue
                set_seed(10)
                dataset = Knowledgedata(test_set, int(real_data), q_subject)
                for method in methods:
                    print(f'running {model} {test_set} {real_data} {method} {q_subject}')
                    outputfile = f'baselines/{model}_{test_set}_{real_data}_{method}_{q_subject}.jsonl'
                    if os.path.exists(outputfile):
                        continue
                    if method == 'zero_shot':
                        zero_shot(llm, sampling_params, dataset, random_one=False, outputfile=outputfile)
                    if method == 'few_shot':
                        few_shot(llm, sampling_params, dataset, shot_type='rel', shot_num=4, random_one=False, outputfile=outputfile)
                    if method == 'discriminate':
                        run_discriminate_prompt(llm, sampling_params, dataset, random_one=False, outputfile=outputfile)

