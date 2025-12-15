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
import os
import time
import pandas as pd
import logging

# Google/ADK imports
import vertexai
from google.cloud import storage
from vertexai.evaluation import EvalTask, PointwiseMetric
from vertexai.generative_models import GenerativeModel

# import configuration
from ...config import config

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Utility: upload to GCS
# TODO: make this tool common across all agents
def _upload_to_gcs(local_path: str, bucket_name: str, folder_name: str) -> str:
    """Uploads a local file to a specific folder in Google Cloud Storage."""
    logger.info(
        f"Uploading file {local_path} to GCS bucket {bucket_name} in folder {folder_name}..."
    )
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_name = f"{folder_name}/{os.path.basename(local_path)}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    gcs_path = f"gs://{bucket_name}/{blob_name}"
    logger.info(f"Successfully uploaded file to {gcs_path}")
    return gcs_path


# MAIN Utility: Evaluate fine tuned model
def evaluate_model(
    model_endpoint: str,
) -> str:
    """Evaluates a fine-tuned Text-to-SQL model using the Vertex AI Evaluation service."""
    # set configurations
    project_id = config.project_id
    location = config.location
    eval_dataset_path = config.eval_dataset
    gcs_bucket_name = config.bucket_name

    logger.info(f"--- Executing Tool: evaluate_model ---")
    logger.info(
        f"Received parameters: model_endpoint={model_endpoint}, eval_dataset_path={eval_dataset_path}, project_id={project_id}, location={location}, gcs_bucket_name={gcs_bucket_name}"
    )
    vertexai.init(project=project_id, location=location)

    # TODO: Improve to read Seed data from BQ rather than files at all
    # Resolve seed_data_path for local files if it's not a GCS path
    if not eval_dataset_path.startswith("gs://"):
        # If the path is relative, resolve it relative to the current script's directory
        if not os.path.isabs(eval_dataset_path):
            # Get the absolute path of the current script's directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            eval_dataset_path = os.path.join(current_script_dir, eval_dataset_path)
        logger.info(f"Resolved local eval_dataset_path: {eval_dataset_path}")

    # Load the evaluation dataset
    eval_df = pd.read_csv(eval_dataset_path)
    tuned_model = GenerativeModel(model_endpoint)

    # Generate predictions
    logger.info("Generating predictions from the fine-tuned model...")
    predictions = [
        tuned_model.generate_content(
            f"Question: {row['question']}\n\nSQL:"
        ).text.strip()
        for _, row in eval_df.iterrows()
    ]
    eval_df["response"] = predictions
    logger.info("Successfully generated predictions.")
    logger.info(f"Sample predictions:\n{eval_df.head()}\n")

    # Rename columns to match the expected format for the evaluation task
    eval_df = eval_df.rename(
        columns={"question": "instruction", "ground_truth_sql": "context"}
    )

    # Define the evaluation task
    logger.info("Defining the evaluation task...")
    eval_task = EvalTask(
        dataset=eval_df,
        metrics=[
            PointwiseMetric(
                metric="relevance",
                metric_prompt_template="""You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria. You will be assessing question answering relevance, which measures the ability to respond with relevant information when asked a question. You will assign the writing response a score from 5, 4, 3, 2, 1, following the INDIVIDUAL RATING RUBRIC and EVALUATION STEPS.

# Evaluation
## Criteria
Relevance: The response should be relevant to the instruction and directly address the instruction.

## Rating Rubric
5 (completely relevant): Response is entirely relevant to the instruction and provides clearly defined information that addresses the instruction's core needs directly.
4 (mostly relevant): Response is mostly relevant to the instruction and addresses the instruction mostly directly.
3 (somewhat relevant): Response is somewhat relevant to the instruction and may address the instruction indirectly, but could be more relevant and more direct.
2 (somewhat irrelevant): Response is minimally relevant to the instruction and does not address the instruction directly.
1 (irrelevant): Response is completely irrelevant to the instruction.

## Evaluation Steps
STEP 1: Assess relevance: is response relevant to the instruction and directly address the instruction?
STEP 2: Score based on the criteria and rubrics.

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

# User Inputs and AI-generated Response
## User Inputs
### INSTRUCTION
{instruction}

### CONTEXT
{context}

## AI-generated Response
{response}""",
            ),
            PointwiseMetric(
                metric="helpfulness",
                metric_prompt_template="""You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria. You will be assessing question answering helpfulness, which measures the ability to provide important details when answering a question. You will assign the writing response a score from 5, 4, 3, 2, 1, following the INDIVIDUAL RATING RUBRIC and EVALUATION STEPS.

# Evaluation
## Criteria
Helpfulness: The response is comprehensive with well-defined key details. The user would feel very satisfied with the content in a good response.

## Rating Rubric
5 (completely helpful): Response is useful and very comprehensive with well-defined key details to address the needs in the question and usually beyond what explicitly asked. The user would feel very satisfied with the content in the response.
4 (mostly helpful): Response is very relevant to the question, providing clearly defined information that addresses the question's core needs.  It may include additional insights that go slightly beyond the immediate question.  The user would feel quite satisfied with the content in the response.
3 (somewhat helpful): Response is relevant to the question and provides some useful content, but could be more relevant, well-defined, comprehensive, and/or detailed. The user would feel somewhat satisfied with the content in the response.
2 (somewhat unhelpful): Response is minimally relevant to the question and may provide some vaguely useful information, but it lacks clarity and detail. It might contain minor inaccuracies. The user would feel only slightly satisfied with the content in the response.
1 (unhelpful): Response is useless/irrelevant, contains inaccurate/deceptive/misleading information, and/or contains harmful/offensive content. The user would feel not at all satisfied with the content in the response.

## Evaluation Steps
STEP 1: Assess comprehensiveness: does the response provide specific, comprehensive, and clearly defined information for the user needs expressed in the question?
STEP 2: Assess relevance: When appropriate for the question, does the response exceed the question by providing relevant details and related information to contextualize content and help the user better understand the response.
STEP 3: Assess accuracy: Is the response free of inaccurate, deceptive, or misleading information?
STEP 4: Assess safety: Is the response free of harmful or offensive content?

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

# User Inputs and AI-generated Response
## User Inputs
### INSTRUCTION
{instruction}

### CONTEXT
{context}

## AI-generated Response
{response}""",
            ),
            PointwiseMetric(
                metric="fulfillment",
                metric_prompt_template="""You are a professional writing evaluator. Your job is to score writing responses according to pre-defined evaluation criteria. You will be assessing fulfillment, which measures the ability to follow instructions. You will assign the writing response a score from 5, 4, 3, 2, 1, following the INDIVIDUAL RATING RUBRIC and EVALUATION STEPS.

# Evaluation
## Criteria
Instruction following: The response demonstrates a clear understanding of the instructions, satisfying all of the instruction's requirements.

## Rating Rubric
5 (complete fulfillment): Response addresses all aspects and adheres to all requirements of the instruction. The user would feel like their instruction was completely understood.
4 (good fulfillment): Response addresses most aspects and requirements of the instruction. It might miss very minor details or have slight deviations from requirements. The user would feel like their instruction was well understood.
3 (some fulfillment): Response does not address some minor aspects and/or ignores some requirements of the instruction. The user would feel like their instruction was partially understood.
2 (poor fulfillment): Response addresses some aspects of the instruction but misses key requirements or major components. The user would feel like their instruction was misunderstood in significant ways.
1 (no fulfillment): Response does not address the most important aspects of the instruction. The user would feel like their request was not at all understood.

## Evaluation Steps
STEP 1: Assess instruction understanding: Does the response address the intent of the instruction such that a user would not feel the instruction was ignored or misinterpreted by the response?
STEP 2: Assess requirements adherence: Does the response adhere to any requirements indicated in the instruction such as an explicitly specified word length, tone, format, or information that the response should include?

Give step by step explanations for your scoring, and only choose scores from 5, 4, 3, 2, 1.

# User Inputs and AI-generated Response
## User Inputs
### INSTRUCTION
{instruction}

## AI-generated Response
{response}""",
            ),
        ],
    )

    logger.info("Starting evaluation...")
    eval_result = eval_task.evaluate()
    logger.info("Evaluation completed.")

    results_filename = f"evaluation_results_{int(time.time())}.json"
    eval_result.metrics_table.to_json(results_filename, orient="records", lines=True)
    logger.info(f"Saved detailed evaluation results to {results_filename}")

    # Upload the results to GCS
    results_gcs_path = _upload_to_gcs(
        results_filename, gcs_bucket_name, "evaluation_results"
    )

    # Calculate average scores
    avg_relevance = eval_result.metrics_table["relevance/score"].mean()
    avg_helpfulness = eval_result.metrics_table["helpfulness/score"].mean()
    avg_fulfillment = eval_result.metrics_table["fulfillment/score"].mean()

    # Remove local temp files
    os.remove(results_filename)

    output = {
        "average_relevance": avg_relevance,
        "average_helpfulness": avg_helpfulness,
        "average_fulfillment": avg_fulfillment,
        "detailed_results_gcs_path": results_gcs_path,
    }
    logger.info(f"Returning from tool: {json.dumps(output)}")
    return json.dumps(output)
