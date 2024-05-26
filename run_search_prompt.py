from tqdm import tqdm
from args import parse_args
from torch.utils.data import DataLoader
import torch
from local_model import get_model_and_tokenizer, get_raw_embedding_table
from utils import nn_project
from utils_helper import set_seed
from utils_datafile import get_output_file, to_jsonl, preprocess
from utils_loss import get_target_loss, get_semantic_loss, gradient_reg
from utils_generate import detail_generate
from utils_helper import get_forbidden_tokens
import gc
from dataset import Knowledgedata

def run_knowledge_iter(args, qs, ans, model, tokenizer, embedding_table):
    multi_tokenized_qs, multi_q_attention_mask, tokenized_ans = preprocess(args, qs, ans, tokenizer)

    forbidden_toks, allowed_toks = get_forbidden_tokens(multi_tokenized_qs, tokenized_ans, tokenizer)
    forbidden_toks = forbidden_toks.to(args.device)
    allowed_toks = allowed_toks.to(args.device)
    for tokenized_qs, q_attention_mask in zip(multi_tokenized_qs, multi_q_attention_mask):
        curr_q_tensor = torch.Tensor(tokenized_qs).long().to(args.device)
        q_attention_mask = torch.Tensor(q_attention_mask).long().to(args.device)
        input_embedding = embedding_table[curr_q_tensor].to(args.device)
        eos_embedding = embedding_table[tokenizer.eos_token_id].to(args.device)
        eos_embedding = eos_embedding.unsqueeze(0).repeat(curr_q_tensor.shape[0], 1).unsqueeze(1)
        for_semantic_embedding = torch.cat([input_embedding, eos_embedding], dim=1)
        new_attention_mask = torch.cat(
            [q_attention_mask, torch.ones((q_attention_mask.shape[0], 1)).long().to(args.device)], dim=1)
        _decode_toks, _logits, semantic_feature = detail_generate(model, tokenizer, embedding_table,
                                                                for_semantic_embedding, new_attention_mask, args.device,
                                                                generate_length=1)
        optimizer = torch.optim.Adam([input_embedding], lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        final_generate_length = max(args.generate_length, max([len(tokenized_ans[i]) for i in range(len(tokenized_ans))]))+1
        #print(min([len(tokenized_ans[i]) for i in range(len(tokenized_ans))]))
        for iter in range(args.iter_num):
            #start_time = time.time()
            # delta_embedding = torch.zeros(input_embedding.shape, requires_grad=True).to(args.device)
            input_embedding.requires_grad = True
            optimizer.zero_grad()
            decode_toks, logits, _semantic_feature = detail_generate(model, tokenizer, embedding_table, input_embedding,
                                                                    q_attention_mask, args.device,
                                                                    final_generate_length)

            #generation_time = time.time()
            true_input = embedding_table[curr_q_tensor]
            decode_toks, _, _ = detail_generate(model, tokenizer, embedding_table, true_input,
                                                                    q_attention_mask, args.device,
                                                                    final_generate_length)
            for output_id in range(decode_toks.shape[0]):
                for ans_id in range(len(tokenized_ans)):
                    ans_len = torch.Tensor(tokenized_ans[ans_id]).long().shape[0]
                    for idx in range(decode_toks.shape[1] - ans_len):
                        output_match = (decode_toks[output_id, idx:idx + ans_len] == torch.Tensor(tokenized_ans[ans_id]).long().to(args.device)).all()
                        if output_match:
                            #print("match")
                            decode_tok = decode_toks[output_id, :idx]
                            output_prompt = torch.cat([curr_q_tensor[output_id], decode_tok], dim=0)
                            return 1, tokenizer.decode(output_prompt), ans[ans_id][0], tokenizer.decode(tokenized_qs[output_id]), ans[ans_id][0], iter
            #evaluate_time = time.time()
            target_loss = get_target_loss(logits, tokenized_ans, args.device)
            now_for_semantic_embedding = torch.cat([input_embedding, eos_embedding], dim=1)
            _, _, now_semantic_feature = detail_generate(model, tokenizer, embedding_table, now_for_semantic_embedding,
                                                        new_attention_mask, args.device,
                                                        generate_length=1)
            semantic_loss = get_semantic_loss(semantic_feature, now_semantic_feature, args.device)
            reg_loss = gradient_reg(input_embedding, embedding_table, allowed_toks)
            loss = target_loss + args.semantic_weight * semantic_loss + args.regular_weight * reg_loss
            loss.backward()
            #loss_time = time.time()
            optimizer.step()

            if (iter + 1) % args.schedule_step == 0:
                scheduler.step()
            with torch.no_grad():
                print(f'iteration {iter}, target loss is {target_loss.item()}, semantic loss is {semantic_loss.item()}, reg loss is {reg_loss.item()}')
                # print(input_embedding.grad)
                if (iter + 1) % args.project_period == 0:
                    new_embedding, tokenized_prompt = nn_project(input_embedding, model.get_input_embeddings(), forbidden_toks, args.ceil, args.device)
                    curr_q_tensor = tokenized_prompt

                    input_embedding.data.copy_(new_embedding.data)
    return -1, None, None, qs[0][0], ans[0][0], args.iter_num


def run_knowledge(args):
    set_seed(args.seed)
    dataset = Knowledgedata(args.dataset, args.real_data, args.q_subject)
    data = DataLoader(dataset=dataset, batch_size=1, num_workers=args.num_workers)
    print("The dataset has {} cases in total".format(len(data)))
    model, tokenizer = get_model_and_tokenizer(args.model_id, args.device)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    embedding_table = get_raw_embedding_table(model)
    output_filename = get_output_file(args, name='search', output_dir='outputs')
    results_dicts = []
    overall_success = 0
    overall_test = 0
    i = 0

    for idx, qs, ans in tqdm(data):
        results_dict = {}
        success, prompt, answer, ori_q, ori_a, iter_round = run_knowledge_iter(args, qs, ans, model, tokenizer, embedding_table)
        torch.cuda.empty_cache()
        gc.collect()
        results_dict["idx"] = idx.item()
        results_dict["success"] = success
        results_dict["prompt"] = prompt
        results_dict["ans"] = answer
        results_dict["original question"] = ori_q
        results_dict["original answer"] = ori_a
        results_dict["iter round"] = iter_round
        if success != 0:
            overall_test += 1
        if success > 0:
            overall_success += 1
        results_dicts.append(results_dict)
        if (i + 1) % args.save_every == 0:
            print("Saving...")
            all_dicts = [vars(args)] + results_dicts
            to_jsonl(all_dicts, output_filename, silence=False)

        i = i + 1
    all_dicts = [vars(args)] + results_dicts
    to_jsonl(all_dicts, output_filename)
    print(f'total test examples is {overall_test}, success rate is {overall_success / overall_test}')


if __name__ == '__main__':
    args = parse_args()
    run_knowledge(args)
