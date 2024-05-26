# Knowledge Boundary

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for "[Benchmarking Knowledge Boundary for Large Language Model: A Different Perspective on Model Evaluation](https://arxiv.org/abs/2402.11493)" by Xunjian Yin, Xu Zhang, Jie Ruan and Xiaojun Wan, published in ACL 2024.

### Illustration of PGDC
![PGDC](https://github.com/pkulcwmzx/knowledge_boundary/blob/main/image/method.png)

## Datasets
We have processed the datasets: ALCUNA, PARAREL, KAssess, CFACT and MMLU. The dataset file are provided in `datasets`.

## Models
We use the weights provided by Huggingface. To modify the paths to your models and tokenizers， please change the model path in `model_utils.py`. An example of loading GPT-2 is given as follows.
```python
    if model_id.startswith('gpt2'):
        path = "your path"
        model = GPT2LMHeadModel.from_pretrained(path)
        tokenizer = GPT2Tokenizer.from_pretrained(path, padding_side="left")
```

## Experiments
To perform experiments on PGDC, run the following code:
```bash
 python run_search_prompt.py --model_id llama --iter_num 25 --dataset kass --real_data 1 --lr 5e-3 --ceil 9.9
```
To perform experiments with baseline methods, run the following code:
```bash
 python run_baseline_prompt.py
```
Hyper-parameters can be tuned in `args.py`.

## Citation
If you find this useful in your research, please consider citing:

```
@misc{yin2024benchmarking,
      title={Benchmarking Knowledge Boundary for Large Language Model: A Different Perspective on Model Evaluation}, 
      author={Xunjian Yin and Xu Zhang and Jie Ruan and Xiaojun Wan},
      year={2024},
      eprint={2402.11493},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

