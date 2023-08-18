# Prompt Engineering with Hugging Face, Databricks and MLflow

<img src="https://github.com/rafaelvp-db/databricks-llm-workshop/blob/main/img/header.png?raw=true" />

## Contents

The repo is structured per different use cases related to Prompt Engineering and LLMs. As of 18/08/2023, you will find the following examples in the `notebooks` folder:

🙋🏻‍♂️ `customer_service`

For this use case, there are 5 different notebooks:

🤓 `00_hf_mlflow_crash_course`: provides a basic example using [Hugging Face](https://huggingface.co/) for training an [intent classification model](https://research.aimultiple.com/intent-classification/) using `distilbert-qa`. Also showcases foundational concepts of MLflow, such as [experiment tracking](https://mlflow.org/docs/latest/tracking.html), [artifact logging](https://mlflow.org/docs/latest/python_api/mlflow.artifacts.html) and [model registration](https://mlflow.org/docs/latest/model-registry.html).

🎬 `01_primer`: mostly conceptual notebook. Contains explanations around Prompt Engineering, and foundational concepts such as **Top K** sampling, **Top p** sampling and **Temperature**.

🧪 `02_basic_prompt_evaluation`: demonstrates basic Prompt Engineeering with lightweight LLM models. In addition to this, showcases [MLflow's newest LLM features](https://www.databricks.com/blog/announcing-mlflow-24-llmops-tools-robust-model-evaluation), such as `mlflow.evaluate()`.

💉 `03_few_shot_learning`: here we explore Few Shot Learning for a sequence classification use case using [mpt-7b-instruct](https://huggingface.co/mosaicml/mpt-7b-instruct).

🏃🏻‍♂️ `04_active_prompting`: in this notebook, we explore active learning techniques. Additionally, we demonstrate how to leverage [VLLM](https://vllm.readthedocs.io/en/latest/) in order to achieve 7X - 10X inference latency improvements.

## Getting Started

To start using this repo on Databricks, there are a few pre-requirements:

1. Create a [GPU Cluster](https://learn.microsoft.com/en-us/azure/databricks/clusters/gpu), minimally with [Databricks Machine Learning Runtime 13.2 GPU](https://docs.databricks.com/en/release-notes/runtime/13.2ml.html) and an [NVIDIA T4 GPU](https://www.nvidia.com/en-us/data-center/tesla-t4/) ([A100](https://www.nvidia.com/en-us/data-center/a100/) is required for the steps involving VLLM).
2. Configure the [init script](https://docs.databricks.com/en/init-scripts/index.html) for the cluster. Once you [clone this repo to your workspace](https://docs.databricks.com/en/repos/index.html), you can configure a init script by pointing to the following path in the Init Script configuration: `/Repos/your_name@email.com/databricks-llm-prompt-engineering/init/init.sh`
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
4. Once all dependencies finish installing and your cluster has successfully started, you should be good to go.


   
## Coming soon

🔎 [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag)
🚀 [Model Deployment and Real Time Inference](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
🛣️ [MLflow AI Gateway](https://mlflow.org/docs/latest/gateway/index.html)


## Credits & Reference

* [Rafael Pierre](https://github.com/rafaelvp-db)
* [Daniel Liden](https://github.com/djliden)
* [Getting Started with NLP using Hugging Face Transformers](https://www.databricks.com/blog/2023/02/06/getting-started-nlp-using-hugging-face-transformers-pipelines.html)
* [DAIR.ai - Prompt Engineering Guide](https://www.promptingguide.ai/)
* [Peter Cheng - Token Selection Strategies: Top-K, Top-p and Temperature](https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature/)
