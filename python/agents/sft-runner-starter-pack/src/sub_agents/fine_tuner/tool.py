# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard imports
import json
import time
import logging

# Google/ADK Imports
import vertexai
from google.cloud import aiplatform
from vertexai.preview.tuning import sft

# configuration import
from ...config import config

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Utility: Run supervised fine tuning
def fine_tune_model(
    training_data_gcs_path: str,
) -> str:
    """Starts a supervised fine-tuning job on Vertex AI and returns the new model endpoint."""
    # Set project and location from config
    project_id = config.project_id
    location = config.location
    base_model = config.flash_model

    logger.info(f"--- Executing Tool: fine_tune_model ---")
    logger.info(
        f"Received parameters: training_data_gcs_path={training_data_gcs_path}, base_model={base_model}, project_id={project_id}, location={location}"
    )

    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    tuned_model_name = f"sft-adk-agent-tuned-model-{int(time.time())}"

    logger.info(
        f"Starting fine-tuning job with model {base_model} and training data {training_data_gcs_path}..."
    )
    sft_tuning_job = sft.train(
        source_model=base_model,
        train_dataset=training_data_gcs_path,
        tuned_model_display_name=tuned_model_name,
        epochs=4,  # TODO: change to config item
        adapter_size=4,  # TODO: change to config item
    )
    logger.info("Fine-tuning job started.")

    job_id = sft_tuning_job.resource_name.split("/")[-1]

    # It can take some time for the endpoint to be ready, so we poll for it.
    while True:
        endpoints = aiplatform.Endpoint.list(
            filter=f"labels.google-vertex-llm-tuning-job-id={job_id}"
        )
        if endpoints:
            tuned_model_endpoint = endpoints[0]
            logger.info(f"Model is deployed to endpoint: {tuned_model_endpoint.name}")
            break
        logger.info("Waiting for endpoint to be created and model to be deployed...")
        time.sleep(5)

    output = {"model_endpoint": tuned_model_endpoint.resource_name}
    logger.info(f"Returning from tool: {json.dumps(output)}")
    return json.dumps(output)
