from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import GPTJForCausalLM, AutoTokenizer
from transformers import OPTForCausalLM
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from datetime import datetime
import torch

def get_raw_embedding_table(model):
    return model.get_input_embeddings()._parameters['weight']
def get_model_and_tokenizer(model_id, device):
    print("Loading model and tokenizer...")
    start = datetime.now()
    if model_id.startswith('gpt2'):
        path = "/data2/pretrain/" + model_id
        model = GPT2LMHeadModel.from_pretrained(path)
        tokenizer = GPT2Tokenizer.from_pretrained(path, padding_side="left")
    elif model_id == 'DAPT':
        path = "/data/pretrain/gpt2/DAPT"
        model = GPT2LMHeadModel.from_pretrained(path)
        tokenizer = GPT2Tokenizer.from_pretrained(path)
    elif model_id.startswith('opt'):
        path = "/data1/PLM/opt/" + model_id
        model = OPTForCausalLM.from_pretrained(path)
        tokenizer = GPT2Tokenizer.from_pretrained(path)
    elif model_id.startswith('bloom'):
        path = "/data1/PLM/bloom/" + model_id
        model = BloomForCausalLM.from_pretrained(path)
        tokenizer = BloomTokenizerFast.from_pretrained(path)
    elif model_id == 'gptj':
        path = "/dataA/PLM/gptj"
        model = GPTJForCausalLM.from_pretrained(path, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(path)
    elif model_id == "llama":
        path = "/data2/pretrain/llama2/Llama-2-7b-chat-hf"
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
    elif model_id == "alpaca":
        path = "/data1/PLM/alpaca-7b"
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
    elif model_id == "mistral":
        path = '/data2/pretrain/Mistral-7B-Instruct-v0.2'
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
    else:
        raise NotImplementedError
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    print(f"Finished in {str(datetime.now() - start)}")
    return model, tokenizer
