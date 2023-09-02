## Databricks LLMs with Model Serving
### Gradio Frontend App

#### Instructions

1. In your laptop, clone the repo https://github.com/rafaelvp-db/databricks-llm-prompt-engineering
2. Inside the repo, go to the folder `frontend`
3. In a terminal window, create virtual environment by running:
```bash
python -m venv .venv
```
4. Activate the virtual environment - on Linux/Mac: `source .venv/bin/activate`
5. Install the necessary requirements by running `pip install -r requirements.txt`
6. In the `frontend` folder:
    * Copy the `.env.example` file to the same folder
    * Rename the copied file to `.env`
    * Fill in the necessary parameters in this file
7. From the terminal in the `frontend` folder, run:
```bash
gradio app.py
```
8. If every went well, you should be able to load the Gradio app by firing up a browser window and entering the following URL: http://localhost:7861

##### Sample Screenshot

<img src="https://github.com/rafaelvp-db/databricks-llm-prompt-engineering/blob/main/frontend/img/gradio.png?raw=true" />

#### Reference

* [Gradio](https://www.gradio.app/)
* [Databricks Model Serving](https://docs.databricks.com/en/machine-learning/model-serving/index.html)
