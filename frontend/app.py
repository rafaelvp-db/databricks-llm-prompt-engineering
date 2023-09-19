import gradio as gr
from dotenv import load_dotenv
import requests
import json
import os
import logging

load_dotenv()

logging.basicConfig(level = "DEBUG")

def generate(
    message: str = "what is ML?",
    max_new_tokens: int = 1000,
    temperature: float = 0.3
):
    
    payload = {
        "inputs":
            {
                "prompt": [message],
                "temperature": [temperature]
            }
    }

    token = os.environ["DATABRICKS_TOKEN"]
    host = os.environ["DATABRICKS_HOST"]
    endpoint = os.environ["ENDPOINT_NAME"]

    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }

    url = f'{host}/serving-endpoints/{endpoint}/invocations'
    response = requests.post(
        url = url,
        headers = headers,
        data = json.dumps(payload)
    )
    logging.info(f"Response: {response.text}")
    return response.json()["predictions"][0]["candidates"][0]["text"]


def predict(message, history):
 
    logging.info(f"Message: {message}")
    final_message = f"""
    ### INSTRUCTION: {message}
    ### RESPONSE:
    """
    response = generate(final_message)
    logging.info(f"Response: {response}")
    response = response.split("INSTRUCTION")[0].split("\n")[0]
    return response

gr.ChatInterface(predict).launch()