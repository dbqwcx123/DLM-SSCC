## Setup
Please properly prepare the environment including installing `LLaMA-Factory`. For the human eval infilling tasks, please install the utils following [this](https://github.com/openai/human-eval-infilling).

## Details
We build the zero-shot evaluation, please call the function in `main`. For example:
```python
eval_hellaswag(model, tokenizer, args)
```
We also open-source the implementation of baseline models [Plaid](https://github.com/igul222/plaid/tree/main) and [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/tree/main). Most of the datasets are loaded using Huggingface datasets. 