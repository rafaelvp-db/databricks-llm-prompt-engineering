# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Prompt Engineering Techniques
# MAGIC ### Active Prompting & Prompting Ensemble
# MAGIC <hr/>
# MAGIC
# MAGIC ### Recap
# MAGIC
# MAGIC * In the last notebook, we achieved good results with Few Shot Learning
# MAGIC * However, for some of the intents, performance was really bad:
# MAGIC   * `change_delivery_address`
# MAGIC   * `contact_customer_service`
# MAGIC * Let's increase the number of samples for these two particular intents and see how that affects our overall performance

# COMMAND ----------

sdf = spark.read.table("prompt_engineering.customer_support_intent")
df = sdf.toPandas()

# COMMAND ----------

import pandas as pd
import numpy as np

def get_top_intents_samples(
  df,
  k = 10,
  random_state = 123,
  n_samples = 5,
  active_intents = [],
  active_prompt_factor = 2
):
  
  df_arr = []

  for intent in df.intent.unique()[:k]:
    if intent not in active_intents:
      active_prompt_factor = 1
    
    final_n_samples = n_samples * active_prompt_factor
    sample = df.loc[df.intent == intent, :].sample(
      n = final_n_samples,
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
  n_samples = 3,
)

print(f"Training Set - Unique Intents:\n{sorted(df_train.intent.unique())}")
print(f"\nTesting Set - Unique Intents:\n{sorted(df_test.intent.unique())}")

# COMMAND ----------

# DBTITLE 1,Declaring and Configuring MPT-7b-Instruct
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
config.max_seq_len = 6000

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

import tqdm

def train_predict(prompt_template, df_train, df_test):

  sample_dict_arr = df_train.to_dict(orient="records")
  examples = sample_dict_arr

  utterances = df_test.utterance.values
  intents = df_test.intent.values
  utterances_intents = list(zip(utterances, intents))
  result = []

  for utterance_intent in tqdm.tqdm(utterances_intents):

    formatted_prompt = prompt_template.format(
      intents = df_train.intent.unique(),
      examples = examples,
      utterance = utterance_intent[0]
    )

    response = generate_text(prompt = formatted_prompt)
    response["actual_intent"] = utterance_intent[1]
    result.append(response)

  return result

prompt_template = """
    Below you will find an array containing three examples of utterances for each of these intents. Each example is in JSON format:
    {examples}
    Based on the examples above, return a JSON object with the actual utterance, and the right intent for that utterance. Please include only one of the intents listed above.
    {utterance}
  """

result = train_predict(prompt_template, df_train, df_test)

# COMMAND ----------

from sklearn.metrics import classification_report

result_df = pd.DataFrame.from_dict(result)
report = classification_report(result_df.actual_intent, result_df.intent)
print(report)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Chain-of-Thought
import tqdm

def train_predict(df_train, df_test):

  prompt = """
    In a dataset, we have 10 different intents:
    {intents}
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
      intents = df_train.intent.unique(),
      examples = examples,
      utterance = utterance_intent[0]
    )

    response = generate_text(prompt = formatted_prompt)
    if response["intent"] not in df_train.unique():
      prompt = """You predicted the utterance '{utterance}' as having
      intent = '{predicted_intent}', but this intent doesn't exist. Pick one of the following ones that relates the closest:\n{actual_intent}"""
      response = generate_text(
        prompt = prompt.format(
          utterance = utterance_intent[0],
          predicted_intent = response["intent"],
          actual_intent = utterance_intent[1]
        )
      )
    response["actual_intent"] = utterance_intent[1]
    result.append(response)

  return result

result = train_predict(df_train, df_test)

# COMMAND ----------

df_train = get_top_intents_samples(
  df = df,
  random_state = 123,
  active_intents = [
    "change_delivery_address",
    "contact_customer_service",
    "find_account_deletion",
    "sign_up",
    "change_order",
    "contact_human_agent",
    "create_account",
    "delete_account"
  ]
)

df_test = get_top_intents_samples(
  df = df[~np.isin(df.index, df_train.index)],
  random_state = 234,
  n_samples = 3,
)

# COMMAND ----------

result = train_predict(df_train, df_test)

# COMMAND ----------

result_df = pd.DataFrame.from_dict(result)
report = classification_report(result_df.actual_intent, result_df.intent)
print(classification_report)

# COMMAND ----------

df_train = get_top_intents_samples(
  df = df,
  random_state = 123,
  active_intents = [
    "change_delivery_address",
    "contact_customer_service",
    "find_account_deletion",
    "sign_up",
    "change_order",
    "contact_human_agent",
    "create_account",
    "delete_account",
    "remove_item"
  ]
)

df_test = get_top_intents_samples(
  df = df[~np.isin(df.index, df_train.index)],
  random_state = 234,
  n_samples = 3,
)

# COMMAND ----------

result = train_predict(df_train, df_test)

# COMMAND ----------

from sklearn.metrics import classification_report

result_df = pd.DataFrame.from_dict(result)
report = classification_report(result_df.actual_intent, result_df.intent)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Evaluation
# MAGIC
# MAGIC * By adding more examples we were able to sequentially improve individual performance per intent, as well as overall F1 score.
# MAGIC * We will probably get even better results if we simply increase the amount of samples globally.
# MAGIC * Of course, here we have the extra samples at our disposal. In a real-life setting, we would have to manually label these extra samples, which can be costly and time consuming. At the same time, improving individual performance of intents can already push F1 score up quite significantly - specially if the intents in question have bad performance and are quite frequent.
# MAGIC * Just for confirming our assumption about extra samples, let's increase the number of examples for all intents and see the results we get.

# COMMAND ----------

df_train = get_top_intents_samples(
  df = df,
  random_state = 123,
  n_samples = 7
)

df_test = get_top_intents_samples(
  df = df[~np.isin(df.index, df_train.index)],
  random_state = 234,
  n_samples = 3,
)

# COMMAND ----------

result = train_predict(df_train, df_test)

# COMMAND ----------

result_df = pd.DataFrame.from_dict(result)
report = classification_report(result_df.actual_intent, result_df.intent)
print(report)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC * By adding more few show examples we improved our metrics significantly 🚀
# MAGIC * However, that came at the cost of increased inference latency - which makes sense, since we also increased the number of tokens for each of the prompts - due to including more examples as part of the prompts
# MAGIC * In the next notebook, we'll explore strategies for improving inference latency

# COMMAND ----------

#TODO: include MLflow logging
