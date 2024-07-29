# SGP-Bench

Here is the official evaluation code of SGP-Bench (Paper: Can Large Language Models Understand Symbolic Graphics Programs?)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Installation

Step-by-step instructions on how to get the development environment running.


```bash
git clone https://github.com/sgp-bench/sgp-bench.git
cd sgp-bench
conda env create -f environment.yml
```


## Usage

### Evaluate closed-sourced model
We provide examples of evaluating OpenAI api and Claude api:

```bash
python -m sgp-bench.demo --api openai-4o
```

```bash
python -m sgp-bench.demo --api claude-3.5-sonnet
```

### Evaluate open-sourced model with vLLM
We use [vLLM](https://github.com/vllm-project/vllm) to evaluate open-source LLMs:


```bash
python -m sgp-bench.demo --base_url $BASE_URL --api $API --model_path $MODEL_PATH --eval $EVAL

# example usage Llama 3 8B
python -m sgp-bench.demo --base_url http://172.22.8.4:8000/v1 --api llama3-8B --model meta-llama/Meta-Llama-3-8B --eval svg cad


# 1: python -m sgp-bench.demo --base_url http://172.22.8.8:8000/v1 --api Phi-3-mini-128k-instruct --model microsoft/Phi-3-mini-128k-instruct --eval svg cad inv
# 2: 
# 3: python -m sgp-bench.demo --base_url http://172.17.0.1:8000/v1 --api Phi-3-small-128k-instruct --model microsoft/Phi-3-small-128k-instruct --eval svg cad inv
# python -m sgp-bench.demo --base_url http://172.22.8.1:8000/v1 --api Phi-3-small-128k-instruct --model microsoft/Phi-3-small-128k-instruct --eval svg cad inv
# python -m sgp-bench.demo --base_url http://172.22.8.1:8000/v1 --api Phi-3-mini-128k-instruct --model microsoft/Phi-3-mini-128k-instruct --eval svg cad inv
```

Note, the 
* **$BASE_URL**: `hostname -i`
* **$API**: This variable likely holds the API key or token required to access the service specified by the base URL. It is used for authentication and authorization purposes.
* **$MODEL_PATH**: This variable points to the directory path where the machine learning model is located. This model is presumably used by the `sgp-bench.demo` script for its processing tasks.
* **$EVAL**: This variable points to the directory path where the machine learning model is located. This model is presumably used by the `sgp-bench.demo` script for its processing tasks.


```bash
# llama3-8B
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# llama3-70B
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-70B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# llama3.1-8B
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# llama3.1-70B
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8



# gemma-1.1-2b
python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-2b-it --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# gemma-1.1-7b
python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-7b-it --dtype auto --api-key token-abc123 --tensor-parallel-size 8




# mistral-7B-v0.3
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.3 --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-codestral-22b-v0.1
python -m vllm.entrypoints.openai.api_server --model mistralai/Codestral-22B-v0.1 --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-large
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-Large-Instruct-2407 --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# mistral-nemo
python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-Nemo-Instruct-2407 --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# c4ai-command-r-v01
python -m vllm.entrypoints.openai.api_server --model CohereForAI/c4ai-command-r-v01 --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# starcoder2-15b-instruct-v0.1
python -m vllm.entrypoints.openai.api_server --model bigcode/starcoder2-15b-instruct-v0.1 --dtype auto --api-key token-abc123 --tensor-parallel-size 8



# internlm2_5-7b-chat
python -m vllm.entrypoints.openai.api_server --model internlm/internlm2_5-7b-chat --trust-remote-code --dtype auto --api-key token-abc123  --tensor-parallel-size 8

# internlm2-chat-20b
python -m vllm.entrypoints.openai.api_server --model internlm/internlm2-chat-20b --trust-remote-code --dtype auto --api-key token-abc123  --tensor-parallel-size 8

# internlm2-chat-7b
python -m vllm.entrypoints.openai.api_server --model internlm/internlm2-chat-7b --trust-remote-code --dtype auto --api-key token-abc123  --tensor-parallel-size 8



# qwen-1.5-7B
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-7B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-1.5-32B
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-32B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-1.5-72B
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-72B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-1.5-110B
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen1.5-110B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# codeqwen-1.5-7B
python -m vllm.entrypoints.openai.api_server --model Qwen/CodeQwen1.5-7B-Chat --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# qwen-2-7B
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# qwen-2-72B
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-72B-Instruct --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# Yi-1.5-9B
python -m vllm.entrypoints.openai.api_server --model 01-ai/Yi-1.5-9B-Chat-16K --dtype auto --api-key token-abc123 --tensor-parallel-size 8

# Yi-1.5-34B
python -m vllm.entrypoints.openai.api_server --model 01-ai/Yi-1.5-34B-Chat-16K --dtype auto --api-key token-abc123 --tensor-parallel-size 8


# DeepSeek-Coder-V2-16B
python -m vllm.entrypoints.openai.api_server --model deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --trust-remote-code  --dtype auto --api-key token-abc123 --tensor-parallel-size 8
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Zeju Qiu - [your-email@example.com](mailto:your-email@example.com)

Project Link: [https://github.com/your-username/your-repo](https://github.com/your-username/your-repo)

## Acknowledgements

This project is based on the open-source repository available at https://github.com/openai/simple-evals. We are thankful to OpenAI for providing the base implementation.