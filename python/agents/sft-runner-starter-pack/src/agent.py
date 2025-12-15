# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Root Agent definition."""

# Import standard libraries
import logging

# ADK/Google Imports
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

# Sub agents
from .sub_agents.data_generator.tool import generate_synthetic_data
from .sub_agents.fine_tuner.tool import fine_tune_model
from .sub_agents.evaluator.tool import evaluate_model

# Load configuration
from .config import config

# Filter unwanted warnings
import warnings

warnings.filterwarnings("ignore")

# import prompts
from .prompts import (
    ROOT_PROMPT,
    DATA_GENERATOR_PROMPT,
    FINE_TUNER_PROMPT,
    EVALUATOR_PROMPT,
)

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Agent Definitions
def create_data_generator_agent():
    return LlmAgent(
        name="data_generator_agent",
        model=config.pro_model,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        description="Generates a synthetic dataset for fine-tuning.",
        instruction=DATA_GENERATOR_PROMPT,
        global_instruction="Greet the user first. Tell the user about your role and capabilities. For each step, take confirmation from user.",
        tools=[generate_synthetic_data],
    )


def create_fine_tuner_agent():
    return LlmAgent(
        name="fine_tuner_agent",
        model=config.flash_model,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        description="Fine-tunes a model on Vertex AI.",
        instruction=FINE_TUNER_PROMPT,
        tools=[fine_tune_model],
    )


def create_evaluator_agent():
    return LlmAgent(
        name="evaluator_agent",
        model=config.pro_model,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        description="Evaluates a fine-tuned model.",
        instruction=EVALUATOR_PROMPT,
        tools=[evaluate_model],
    )


full_pipeline_agent = SequentialAgent(
    name="full_pipeline_agent",
    description="Runs the full fine-tuning pipeline: Data Generation -> Fine-Tuning -> Evaluation.",
    sub_agents=[
        create_data_generator_agent(),
        create_fine_tuner_agent(),
        create_evaluator_agent(),
    ],
)


# --- State Setup ---
def setup_before_agent_call(callback_context: CallbackContext):
    """Sets up the initial state for the agent from environment variables."""
    print("--- Setting up initial state from .config file ---")

    callback_context.state["seed_data_path"] = (
        config.seed_queries
    )  # "data/seed_queries.csv"
    callback_context.state["eval_dataset_path"] = (
        config.eval_dataset
    )  # "data/sample_eval_queries.csv"
    callback_context.state["project_id"] = config.project_id
    callback_context.state["gcp_location"] = config.location
    callback_context.state["gcs_bucket_name"] = config.bucket_name
    callback_context.state["base_model"] = config.flash_model
    callback_context.state["target_examples"] = config.initial_target_examples
    callback_context.state["bq_dataset_id"] = config.DATASET
    callback_context.state["GCP_PROJECT_ID"] = config.project_id
    callback_context.state["GCP_LOCATION"] = config.location
    callback_context.state["GCS_BUCKET_NAME"] = config.bucket_name
    callback_context.state["BASE_MODEL_FOR_TUNING"] = config.flash_model
    callback_context.state["INITIAL_TARGET_EXAMPLES"] = config.initial_target_examples
    callback_context.state["BQ_DATASET_ID"] = config.DATASET
    print(f"Initial state setup complete.")


# Root Agent
root_agent = LlmAgent(
    name="FineTuningCoordinator",
    model=config.flash_model,
    description="A master coordinator for a machine learning fine-tuning pipeline.",
    instruction=ROOT_PROMPT,
    sub_agents=[
        full_pipeline_agent,
        create_data_generator_agent(),
        create_fine_tuner_agent(),
        create_evaluator_agent(),
    ],
    before_agent_callback=setup_before_agent_call,
)
