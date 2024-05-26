import torch
import torch.nn.functional as F
def test_one_case(model, test_embeddings, batched_inverse_label_mask):
    out = model(inputs_embeds=test_embeddings)
    log_probs = F.log_softmax(out.logits, dim=-1)[:, :-1, :]
    gen_output = log_probs.argmax(dim=2)
    gen_output[batched_inverse_label_mask] = -100
    return gen_output, log_probs


def evaluate(model, tokenizer, candidate_tokens, full_embeddings, tok_id, labels, output_length):
    """

    :param model: target model
    :param tokenizer: target tokenizer
    :param candidate_tokens: candidate tokens for generation
    :param full_embeddings: embedding matrix for input
    :param tok_id: the token position
    :param labels:
    :return: whether the target is successfully obtained
    """
    out = model(input_embeds=full_embeddings)
    log_probs = F.log_softmax(out.logits[:, -1 - output_length: -1, :], dim=2)
    gen_output = log_probs.argmax(dim=2)
    for i in range(log_probs.shape[0]):
        output_matches = (labels == gen_output[i, :]).all().item()
        if output_matches:
            return i

    return -1


def pass_test(args, decode_toks, tokenized_ans):
    tokenized_ans_ = [torch.Tensor(tokenized_ans[i]).long().to(args.device) for i in range(len(tokenized_ans))]
    for i in range(decode_toks.shape[0]):
        for k in range(len(tokenized_ans_)):
            for j in range(decode_toks.shape[1] - tokenized_ans_[k].shape[0] + 1):
                if (decode_toks[i, j: j + tokenized_ans_[k].shape[0]] == tokenized_ans_[k]).all():
                    return True
    return False