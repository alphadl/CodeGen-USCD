# :zap: USCD: Improving Code Generation of LLMs by Uncertainty-Aware Selective Contrastive Decoding

This is the repository for **USCD** (**U**ncertainty-Aware **S**elective **C**ontrastive **D**ecoding), a simple and effective method for significantly improving the code generation performance of large language models (LLMs). The USCD first induces noise present in standard prompts using meticulously designed lame prompts. Then, USCD eliminates the noise through uncertainly-aware selective contrastive decoding, thereby improving the code generation quality of LLMs. Extensive experiments have shown that the plug-and-play USCD method can effectively improve the performance of LLMs in code generation. For more details, please refer to our paper "[:zap:*USCD: Improving Code Generation of LLMs by Uncertainty-Aware Selective Contrastive Decoding*](https://arxiv.org/abs/2409.05923)".

<div align="center">
    <img width="80%" alt="image" src="https://github.com/alphadl/CodeGen-USCD/blob/main/img/using_contraste_decoding.PNG">
</div>

### Environment

```bash
pip install transformers>=4.25.1
pip install accelerate>=0.13.2
pip install datasets>=2.6.1
pip install evaluate>=0.3.0
pip install pyext==0.7
pip install mosestokenizer==1.0.0
pip install huggingface_hub>=0.11.1
pip install fsspec<2023.10.0
```

### Code

The main framework of our code is based on [bigcode](https://github.com/bigcode-project/bigcode-evaluation-harness). Following example shows you how to perform USCD on code generation dataset.

```bash
cd ./CodeGen-USCD/model_slurm
bash codellama_7b_weight_10_-07_pass@1_1.sh
```

For detailed usage instructions, please refer to the [bigcode documentation](https://github.com/bigcode-project/bigcode-evaluation-harness?tab=readme-ov-file#documentation).

### Data

We provide the following data used in our experimentsm

- Evaluation benchmarks:
    - [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval): The HumanEval benchmark consists of 164 handwritten Python programming problems and primarily focuses on language comprehension, algorithms, and basic mathematics. Additionally, the HumanEval benchmark mainly evaluates the function completion capability of LLMs.
    - [MBPP](https://huggingface.co/datasets/nus-yam/mbpp): The MBPP benchmark primarily evaluates the function generation capability of LLMs. The test set for the MBPP benchmark consists of 500 samples of Python language programs.
    - [MultiPL-E](https://huggingface.co/datasets/nuprl/MultiPL-E): MultiPL-E translates the HumanEval and MBPP benchmarks into eighteen other programming languages, e.g., C++, C#, JAVA, PHP, and Bash.

### Model

We provide the following model used in our experimentsm

- Models:
    - [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b): The Llama2-7b model, released by the Meta research team in July 2023, is pre-trained with a parameter architecture of 70 billion.
    - [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf): The CodeLlama-7b model is fine-tuned based on the Llama model, primarily designed for tasks, e.g., code generation and code understanding.
    - [StarCoder](https://huggingface.co/bigcode/starcoder): The StarCoder model is a 15.5 billion parameter model trained using over 80 programming languages from Stack (v1.2).
    - [WizardCoder](https://huggingface.co/WizardLMTeam/WizardCoder-15B-V1.0): The WizardCoder is fine-tuned by applying the Evol-Instruct to Code LLMs.
    - [StarCoder](https://huggingface.co/facebook/incoder-6B):  The Incoder-6b is trained on code using a causal-masked objective.

## Cite as

Please cite the paper and star this repo if you use USCD and find it helpful. Feel free to contact wangshuai123@whu.edu.cn or open an issue if you have any questions.
```
@misc{wang2024mathbb,
  author = {Shuai Wang, Liang Ding, Li Shen, Yong Luo, Zheng He, Wei Yu, Dacheng Tao},
  title = {USCD: Improving Code Generation of LLMs by Uncertainty-Aware Selective Contrastive Decoding},
  journal = {arXiv preprint arXiv:2409.05923},
  year = 2024,
}
```
