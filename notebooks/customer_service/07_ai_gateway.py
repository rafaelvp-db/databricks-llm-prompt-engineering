# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Manage access to Databricks Serving Endpoint with AI Gateway
# MAGIC <hr/>
# MAGIC <img src="https://promptengineeringdbl.blob.core.windows.net/img/header.png"/>
# MAGIC <hr/>
# MAGIC
# MAGIC This example notebook demonstrates how to use MLflow AI Gateway ([see announcement blog](https://www.databricks.com/blog/announcing-mlflow-ai-gateway)) with a Databricks Serving Endpoint.
# MAGIC
# MAGIC Requirement:
# MAGIC - A Databricks serving endpoint that is in the "Ready" status. Please refer to the `02_mlflow_logging_inference` example notebook for steps to create a Databricks serving endpoint.
# MAGIC
# MAGIC Environment:
# MAGIC - MLR: 13.3 ML
# MAGIC - Instance: `i3.xlarge` on AWS, `Standard_DS3_v2` on Azure

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[gateway]>=2.7"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# TODO: Please change endpoint_name to your Databricks serving endpoint name if it's different
# The below assumes you've create an endpoint "llama2-7b-chat" according to 02_mlflow_logging_inference
endpoint_name = "optimized-mpt-7b-example"
gateway_route_name = f"{endpoint_name}_completion2"

# COMMAND ----------

# Databricks URL and token that would be used to route the Databricks serving endpoint
databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

import mlflow.gateway
mlflow.gateway.set_gateway_uri("databricks")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an AI Gateway Route

# COMMAND ----------

mlflow.gateway.create_route(
    name=gateway_route_name,
    route_type="llm/v1/completions",
    model= {
        "name": endpoint_name, 
        "provider": "databricks-model-serving",
        "databricks_model_serving_config": {
          "databricks_api_token": token,
          "databricks_workspace_url": databricks_url
        }
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query an AI Gateway Route
# MAGIC The below code uses `mlflow.gateway.query` to query the `Route` created in the above cell.
# MAGIC
# MAGIC Note that `mlflow.gateway.query` doesn't need to be run in the same notebook nor the same cluster, and it doesn't require the Databricks URL or API token to query it, which makes it convenient for multiple users within the same organization to access a served model.

# COMMAND ----------

import mlflow

with mlflow.start_run():

  prompt_template = "Get databricks documentation to answer all the questions: {question}"
  prompt = {"prompt": prompt_template.format(question = "What is MLflow?")}
  params = {
    "temperature": 0.1,
    "max_tokens": 300
  }
  for key in params.keys():
    prompt[key] = params[key]

  mlflow.log_params(params)

  response = mlflow.gateway.query(
      route=gateway_route_name,
      data=prompt
  )

  prompts = [prompt_template]
  inputs = [prompt["prompt"]]
  outputs = [response['candidates'][0]['text']]

  mlflow.llm.log_predictions(inputs, outputs, prompts)

# COMMAND ----------


