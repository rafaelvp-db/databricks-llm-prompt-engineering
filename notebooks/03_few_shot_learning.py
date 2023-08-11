# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Few Shot Learning
# MAGIC
# MAGIC TODO: add more text / explanations

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
config.max_seq_len = 4000

model = transformers.AutoModelForCausalLM.from_pretrained(
  name,
  config=config,
  torch_dtype=torch.bfloat16,
  trust_remote_code=True,
  cache_dir="/local_disk0/.cache/huggingface/",
  revision="bbe7a55d70215e16c00c1825805b81e4badb57d7"
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
  "EleutherAI/gpt-neox-20b",
  padding_side="left"
)

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
from jsonformer import Jsonformer

def generate_text(prompt, **kwargs):

  json_schema = {
    "type": "object",
    "properties": {
      "utterance": {"type": "string"},
      "intent": {"type": "string"}
    }
  }

  if "max_new_tokens" not in kwargs:
    kwargs["max_new_tokens"] = 100
  
  kwargs.update(
    {
      "pad_token_id": tokenizer.eos_token_id,
      "eos_token_id": tokenizer.eos_token_id,
    }
  )
  
  if isinstance(prompt, str):
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    response = jsonformer()

  elif isinstance(prompt, list):
    response = []
    for prompt_ in prompt:
      jsonformer = Jsonformer(model, tokenizer, json_schema, prompt_)
      generated_text = jsonformer()
      response.append(generated_text)

  return response

# COMMAND ----------

# DBTITLE 1,Downloading our Dataset from Hugging Face and Saving to Dclta
import datasets

ds = datasets.load_dataset("bitext/customer-support-intent-dataset")
df = ds["train"].to_pandas()
display(df.sample(frac=0.2, random_state = 123).head(10))

# COMMAND ----------

# DBTITLE 1,Creating a sampled dataset for Few Shot Examples
import pandas as pd
import numpy as np

def get_top_intents_samples(df, k = 10, random_state = 123, n_samples = 5):
  
  df_arr = []

  for intent in df.intent.unique()[:k]:
    sample = df.loc[df.intent == intent, :].sample(
      n = n_samples,
      random_state = random_state
    )
    df_arr.append(sample)

  df_sampled = pd.concat(df_arr, axis = 0)
  return df_sampled.sample(frac = 1.0)

df_train = get_top_intents_samples(
  df = df,
  random_state = 123
)

df_test = get_top_intents_samples(
  df = df[~np.isin(df.index, df_train.index)],
  random_state = 234,
  n_samples = 3
)

# COMMAND ----------

# DBTITLE 1,Creating Prompts using Few Shot Learning
import tqdm

prompt = """
  In a dataset, we have 10 different intents.
  Below you will find an array containing three examples of utterances for each of these intents. Each example is in JSON format:
  {examples}
  Based on the examples above, return a JSON object with the actual utterance, and the right intent for that utterance. Please include only one of the intents listed above.
  {utterance}
"""

sample_dict_arr = df_train.to_dict(orient="records")
examples = sample_dict_arr

utterances = df_test.utterance.values
intents = df_test.intent.values
utterances_intents = list(zip(utterances, intents))
result = []

for utterance_intent in tqdm.tqdm(utterances_intents):

  formatted_prompt = prompt.format(
    examples = examples,
    utterance = utterance_intent[0]
  )

  response = generate_text(prompt = formatted_prompt)
  output = f"""\n
    utterance: {utterance}\n
    predicted_intent: {response}\n
    actual_intent: {utterance_intent[1]}
  """
  result.append(output)

# COMMAND ----------

result
