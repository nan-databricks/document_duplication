# Databricks notebook source
# MAGIC %md
# MAGIC ### Load, register, deploy OpenAI Whisper model from the HuggingFace

# COMMAND ----------

# MAGIC %pip install torchvision torch transformers databricks-automl-runtime ffmpeg-python ffmpeg-binaries==1.0.1
# MAGIC %restart_python 

# COMMAND ----------

import mlflow
from transformers import pipeline

# Set the registry URI to Unity Catalog (optional if already default)
mlflow.set_registry_uri("databricks-uc")

# Define catalog and schema
CATALOG = "nan_catalog"
SCHEMA = "default"
MODEL_NAME = "openai-whisper-small"
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small"
)

with mlflow.start_run():
    mlflow.transformers.log_model(
        transformers_model=pipe,
        name="nan-whisper-asr",
        registered_model_name=FULL_MODEL_NAME
    )

# COMMAND ----------

import urllib.request
urllib.request.urlretrieve("http://www.moviesoundclips.net/movies1/darkknightrises/darkness.mp3", "/tmp/audio.mp3")

# COMMAND ----------

displayHTML(
    '<audio controls src="/tmp/audio.mp3"></audio>'
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Inference

# COMMAND ----------

# Load the model from Unity Catalog using FULL_MODEL_NAME
model = mlflow.pyfunc.load_model(f"models:/{FULL_MODEL_NAME}/1")

# Pass the file path as input (use a list for batch inference)
inputs = ["/tmp/audio.mp3"]
predictions = model.predict(inputs)

display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Real-time Inference

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput,
    EndpointCoreConfigInput
)
from databricks.sdk.errors import ResourceDoesNotExist

# Extract model name and version from FULL_MODEL_NAME
model_name = FULL_MODEL_NAME
model_version = "1"  # Default to version 1; update if needed

# Generate endpoint name
serving_endpoint_name = f"nan-whisper-endpoint"

w = WorkspaceClient()  # Assumes DATABRICKS_HOST and DATABRICKS_TOKEN are set

served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=model_version,
        workload_size="Small",
        scale_to_zero_enabled=True
    )
]

# Update serving endpoint if it already exists, otherwise create the serving endpoint
try:
    w.serving_endpoints.update_config(
        name=serving_endpoint_name,
        served_entities=served_entities
    )
except ResourceDoesNotExist:
    w.serving_endpoints.create(
        name=serving_endpoint_name,
        config=EndpointCoreConfigInput(served_entities=served_entities)
    )

# COMMAND ----------

import requests
import base64

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

with open("/tmp/audio.mp3", "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

endpoint_url = f"{w.config.host}/serving-endpoints/{serving_endpoint_name}/invocations"

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# The model expects a single n-dimensional array as input, so send the base64 string directly in a list
payload = {
    "inputs": [audio_b64]
}

response = requests.post(
    endpoint_url,
    headers=headers,
    json=payload
)

if response.ok:
    try:
        result = response.json()
    except Exception:
        result = {"error": "Response is not valid JSON", "content": response.text}
else:
    result = {
        "error": f"Request failed with status {response.status_code}",
        "content": response.text
    }

display(result)

# COMMAND ----------


