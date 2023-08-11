# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Few Shot Learning
# MAGIC
# MAGIC TODO: add more text / explanations

# COMMAND ----------

# DBTITLE 1,Installing Dependencies
# MAGIC %pip install xformers==0.0.20 einops==0.6.1 flash-attn==v1.0.3.post0 triton-pre-mlir@git+https://github.com/vchiley/triton.git@triton_pre_mlir#subdirectory=python

# COMMAND ----------

# DBTITLE 1,Declaring and Configuring MPT-7b
import transformers
import torch

# it is suggested to pin the revision commit hash and not change it for reproducibility because the uploader might change the model afterwards; you can find the commmit history of mpt-7b-instruct in https://huggingface.co/mosaicml/mpt-7b-instruct/commits/main

name = "mosaicml/mpt-7b-instruct"
config = transformers.AutoConfig.from_pretrained(
  name,
  trust_remote_code=True
)
config.attn_config['attn_impl'] = 'triton'
config.init_device = 'cuda'

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  cache_dir="/local_disk0/.cache/huggingface/",
  revision="bbe7a55d70215e16c00c1825805b81e4badb57d7"
)

tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", padding_side="left")

generator = transformers.pipeline(
  "text-generation",
  model=model, 
  config=config, 
  tokenizer=tokenizer,
  torch_dtype=torch.bfloat16,
  device=0
)

# COMMAND ----------

# DBTITLE 1,Declaring our Generation Wrapper Function
import re

def generate_text(prompt, **kwargs):
  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 512
  
  kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
  
  template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n###Instruction\n{instruction}\n\n### Response\n"
  if isinstance(prompt, str):
    full_prompt = template.format(instruction=prompt)
    generated_text = generator(full_prompt, **kwargs)
    generated_text = generated_text[0]["generated_text"]
  elif isinstance(prompt, list):
    full_prompts = list(map(lambda promp: template.format(instruction=promp), prompt))
    outputs = generator(full_prompts, **kwargs)
    generated_text = [out[0]["generated_text"] for out in outputs]
  
  response = generated_text.split("### Response\n")[1]

  return prompt, response

def postprocess(response: str):

  matches = re.search(r"(\'.+\')|(\".+\")", response)
  print(matches)

# COMMAND ----------

# DBTITLE 1,Downloading our Dataset from Hugging Face
import datasets

ds = datasets.load_dataset("bitext/customer-support-intent-dataset")
df = ds["train"].to_pandas()
display(df)

# COMMAND ----------

# DBTITLE 1,Creating a sampled dataset for Few Shot Examples
import pandas as pd

df_arr = []

for intent in df.intent.unique()[:10]:
  sample = df.loc[df.intent == intent, :].sample(n = 3)
  df_arr.append(sample)

df_sampled = pd.concat(df_arr, axis = 0)
display(df_sampled)

# COMMAND ----------

# DBTITLE 1,Creating Prompts using Few Shot Learning
prompt = """
  In a dataset, we have 10 different intents.
  Below you will find an array containing three examples of utterances for each of these intents. Each example is in JSON format:
  {examples}
  Based on the examples above, return the right intent for the utterance below. Write your answer as short as possible and use double quotes (").
  {utterance}
"""

examples = sample_dict_arr
utterance = "I need to cancel my order"

formatted_prompt = prompt.format(
  examples = examples,
  utterance = utterance
)

response = generate_text(prompt = formatted_prompt)
print(response)
