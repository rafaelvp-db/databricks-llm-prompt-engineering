# Prompt Engineering with Hugging Face, Databricks and MLflow

<img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/header.png?raw=true" />

## Getting Started

To start using this repo on Databricks, there are a few pre-requirements:

1. Create a GPU Cluster, minimally with Databricks Machine Learning Runtime 13.2 and an NVIDIA T4 GPU (A100 is required for the steps involving VLLM).
2. Configure the init script for the cluster. Once you clone this repo to your workspace, you can do so by pointing to the following path in the Init Script configuration: `/Repos/your_name@email.com/databricks-llm-prompt-engineering/init/init.sh`
3. Install the following Python packages in your cluster:
```bash
accelerate==0.21.0
einops==0.6.1
flash-attn==v1.0.5
ninja
tokenizers==0.13.3
transformers==4.30.2
xformers==0.0.20
```
4. Once all dependencies finish installing, you should be good to go.

## Contents

The repo is structured per different use cases. As of 18/08/2023, you will find the following examples in the `notebooks` folder:

* `customer_service`
  * For this use case, there are 5 different notebooks:
    * `00_hf_mlflow_crash_course``: provides a basic example using Hugging Face for training an intent classification model using `distilbert-qa`. Also showcases foundational concepts of MLflow, such as experiment tracking, artifact logging and model registration.
    * `01_primer`: mostly conceptual notebook. Contains explanations around Prompt Engineering, and foundational concepts such as **Top K** sampling, **Top p** sampling and **Temperature**.
    * `02_basic_prompt_evaluation`: demonstrates basic Prompt Engineeering with lightweight LLM models. In addition to this, showcases MLflow's newest LLM features, such as `mlflow.evaluate()`.
    * `03_few_shot_learning`: here we explore Few Shot Learning for a sequence classification use case using MPT-Instruct-7b.
    * `04_active_prompting`: in this notebook, we explore active learning techniques. Additionally, we demonstrate how to leverage VLLM in order to achieve 7X - 10X inference latency improvements.
   
## Coming soon

* Retrieval Augmented Generation (RAG)
* Model Deployment and Real Time Inference


## Credits & Reference

* [Rafael Pierre](https://github.com/rafaelvp-db)
* [Daniel Liden](https://github.com/djliden)
* [Getting Started with NLP using Hugging Face Transformers](https://www.databricks.com/blog/2023/02/06/getting-started-nlp-using-hugging-face-transformers-pipelines.html)
* [DAIR.ai - Prompt Engineering Guide](https://www.promptingguide.ai/)
* [Peter Cheng - Token Selection Strategies: Top-K, Top-p and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/)
