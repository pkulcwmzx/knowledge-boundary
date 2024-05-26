import argparse

def parse_args(joint = False):
    parser = argparse.ArgumentParser()
    # ------- dataset -------
    parser.add_argument('--dataset', type=str, default='kass',
                        help='the name of the dataset: kass rome mmlu alcuna')
    parser.add_argument('--q_subject', type=str, default='abstract_algebra_test',
                        help='the subject of the question for mmlu')
    parser.add_argument('--q_type', type=str, default='fill-in-blank',
                        help='the type of the question; fill-in-blank or multi-choice or boolean for alcuna; completion or choice for mmlu')
    parser.add_argument('--real_data', type=int, default=0,
                        help='whether the data is real or fake for rome')
    # ------- model -------
    parser.add_argument('--model_id', type=str, default="llama",
                        help='LLM model name or path, the name or path of the target model')
    # ------- experiment -------
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--question_num', type=int, default=4,
                        help='the total number of seed questions')
    parser.add_argument('--max_len', type=int, default=50,
                        help='the maximum length allowed for our method')
    parser.add_argument('--max_ans_len', type=int, default=11,
                        help='the maximum answer length allowed for our method')
    parser.add_argument('--update_num', type=int, default=1,
                        help='the total update tokens in each iteration')
    parser.add_argument('--candidate_num', type=int, default=64,
                        help='the number of candidate tokens for each position in the prompt')
    parser.add_argument('--iter_num', type=int, default=25,
                        help='the total iteration number of the algorithm')
    parser.add_argument('--evaluate_num', type=int, default=2,
                        help='the total number of prompts selected for the evaluation')
    # ------- optimization -------
    parser.add_argument('--lam_perp', type=int, default=-0.1,
                        help='coefficient of the perplexity loss')
    parser.add_argument('--semantic_weight', type=float, default=0,
                        help='coefficient of the semantic similarity loss')
    parser.add_argument('--regular_weight', type=float, default=0.01,
                        help='coefficient of the gradient regulation loss')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--prompt_type', type=str,
                        help='the type of the prompt', default='raw')
    parser.add_argument('--generate_length', type=int, default=10,
                        help='the length of the generated tokens')
    parser.add_argument('--project_period', type=int, default=1,
                        help='the period of the projection')
    parser.add_argument('--ceil', type=float, default=9.9,
                        help='the ceil of the projection')
    parser.add_argument('--schedule_step', type=int, default=10,
                        help='optimization step for scheduler to step once')

    args = parser.parse_args()
    return args