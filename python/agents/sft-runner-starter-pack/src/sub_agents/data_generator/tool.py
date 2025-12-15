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

# standard imports
import json
import os
import time
import pandas as pd
import logging

# GCP imports
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, GenerationConfig

# Utility imports
from ...utils.mcptoolbox_client import MCPToolboxClient
from ...config import config

# Define logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Utility function: Format schema for Prompt injection
def _format_schemas_for_prompt(
    mcptoolbox_client: MCPToolboxClient, bq_dataset_id: str, project_id: str
) -> str:
    """Fetch DB Schema from BQ and Format for use with prompt for data generation"""
    logger.info(f"Fetching schemas for dataset: {project_id}.{bq_dataset_id}")
    tables_df = mcptoolbox_client.execute_tool(
        "list_tables", {"dataset_id": bq_dataset_id}
    )
    if tables_df.empty:
        raise ValueError(f"No tables found in dataset {bq_dataset_id}.")

    all_schemas_string = (
        f"Dataset: `{project_id}.{bq_dataset_id}`\n\n--- Table Schemas ---\n"
    )
    for table_name in tables_df["table_name"]:
        schema_df = mcptoolbox_client.execute_tool(
            "get_table_schema", {"dataset_id": bq_dataset_id, "table_id": table_name}
        )
        full_table_name = f"`{project_id}.{bq_dataset_id}.{table_name}`"
        schema_string = f"Table Name: {full_table_name}\nColumns:\n"
        for _, row in schema_df.iterrows():
            schema_string += f"- {row['column_name']} ({row['data_type']})\n"
        all_schemas_string += schema_string + "\n---\n"
    logger.info("Successfully formatted schemas.")
    return all_schemas_string


# Utility function: Generate synthetic data examples
def _generate_examples_batch(
    model, schemas_str, num_examples, existing_questions, sample_questions_df
) -> str:
    """Generate X number of synthetic examples"""
    # TODO: Improve: Make the use case generic
    prompt = f"""
    You are an expert system designed to generate synthetic training data for fine-tuning text-to-SQL models.
    Your task is to generate diverse pairs of natural language questions and their corresponding valid GoogleSQL queries based on the provided table schemas and sample questions.

    **Instructions:**
    1.  Generate exactly {num_examples} unique question-query pairs.
    2.  Ensure the generated questions are different from these existing ones: {", ".join(existing_questions[-20:]) if existing_questions else "None"}
    3.  Format the output STRICTLY as a valid JSON list of objects, where each object has two keys: "question" and "query".
    4.  Do NOT include any text, explanations, or markdown formatting outside the JSON list itself.

    **Table Schemas:**
    {schemas_str}

    **Sample Questions for reference:**
    {sample_questions_df.to_string()}

    **Generate the JSON list now:**
    """

    generation_config = GenerationConfig(
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        response_mime_type="application/json",
    )
    try:
        logger.info("Sending request to Gemini to generate examples...")
        response = model.generate_content(prompt, generation_config=generation_config)
        logger.info("Received response from Gemini.")
        return response.text
    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}")
        return None


# Utility funciton: Upload generated data in GCS Bucket
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


# MAIN Utility: Generates synthetic data - calls other Utilities
def generate_synthetic_data(
    project_id: str = config.project_id,
    bq_dataset_id: str = config.DATASET,
    seed_data_path: str = config.seed_queries,
    target_examples: int = config.initial_target_examples,
    gcs_bucket_name: str = config.bucket_name,
) -> str:
    """Generates a synthetic dataset for fine-tuning a Text-to-SQL model."""
    logger.info(f"--- Executing Tool: generate_synthetic_data ---")
    logger.info(
        f"Received parameters: project_id={config.project_id}, \
            bq_dataset_id={config.DATASET}, seed_data_path={config.seed_queries}, \
                target_examples={config.initial_target_examples}, \
                    gcs_bucket_name={config.bucket_name}"
    )

    mcptoolbox_client = MCPToolboxClient(project_id=config.project_id)
    gemini_model = GenerativeModel(config.pro_model)

    # TODO: Improve to read Seed data from BQ rather than files at all
    # Resolve seed_data_path for local files if it's not a GCS path
    if not seed_data_path.startswith("gs://"):
        # If the path is relative, resolve it relative to the current script's directory
        if not os.path.isabs(seed_data_path):
            # Get the absolute path of the current script's directory
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            seed_data_path = os.path.join(current_script_dir, seed_data_path)
        logger.info(f"Resolved local seed_data_path: {seed_data_path}")

    schemas_prompt_string = _format_schemas_for_prompt(
        mcptoolbox_client, bq_dataset_id, project_id
    )
    sample_questions_df = pd.read_csv(seed_data_path)

    all_generated_examples = []
    generated_questions_set = set(sample_questions_df["question"].tolist())

    while len(all_generated_examples) < target_examples:
        batch_size = min(100, target_examples - len(all_generated_examples))
        logger.info(f"Requesting a batch of {batch_size} examples...")

        raw_response_text = _generate_examples_batch(
            model=gemini_model,
            schemas_str=schemas_prompt_string,
            num_examples=batch_size,
            existing_questions=list(generated_questions_set),
            sample_questions_df=sample_questions_df,
        )

        if raw_response_text:
            try:
                batch_examples = json.loads(raw_response_text)
                for item in batch_examples:
                    if (
                        isinstance(item, dict)
                        and "question" in item
                        and "query" in item
                    ):
                        if (
                            item["question"]
                            and item["query"]
                            and item["question"] not in generated_questions_set
                        ):
                            all_generated_examples.append(item)
                            generated_questions_set.add(item["question"])
            except json.JSONDecodeError as e:
                logger.error(f"Warning: Failed to decode JSON from model response: {e}")

        logger.info(f"Total examples generated so far: {len(all_generated_examples)}")
        if len(all_generated_examples) < target_examples:
            time.sleep(5)

    # Create a unique folder for this run
    folder_name = f"sft_data_{int(time.time())}"

    raw_output_filename = f"synthetic_sql_data.jsonl"
    with open(raw_output_filename, "w") as f:
        for example in all_generated_examples:
            f.write(json.dumps(example) + "\n")
    logger.info(f"Successfully generated raw data file: {raw_output_filename}")
    raw_gcs_path = _upload_to_gcs(raw_output_filename, gcs_bucket_name, folder_name)
    logger.info(
        f"Sample from raw data file:\n{pd.read_json(raw_output_filename, lines=True).head()}\n"
    )

    # Prepare the data in the final chat format
    formatted_output_filename = f"formatted_synthetic_sql_data.jsonl"
    _prepare_finetuning_data_chat_format(
        input_jsonl_path=raw_output_filename,
        output_jsonl_path=formatted_output_filename,
        all_schemas_str=schemas_prompt_string,
    )
    formatted_gcs_path = _upload_to_gcs(
        formatted_output_filename, gcs_bucket_name, folder_name
    )

    # Delete local files
    os.remove(raw_output_filename)
    os.remove(formatted_output_filename)

    output = {
        "raw_data_gcs_path": raw_gcs_path,
        "generated_data_gcs_path": formatted_gcs_path,
    }
    logger.info(f"Returning from tool: {json.dumps(output)}")
    return json.dumps(output)


# Utility function: Prepare data format for SFT
def _prepare_finetuning_data_chat_format(
    input_jsonl_path: str, output_jsonl_path: str, all_schemas_str: str
):
    """Reads generated question-query data and formats it into the multi-turn chat format."""
    logger.info(
        f"Preparing data from '{input_jsonl_path}' into multi-turn chat format..."
    )
    count = 0
    with open(input_jsonl_path, "r") as infile, open(output_jsonl_path, "w") as outfile:
        for line in infile:
            original_data = json.loads(line)
            question = original_data.get("question")
            query = original_data.get("query")

            if question and query:
                prepared_data = {
                    "contents": [
                        {"role": "user", "parts": [{"text": question}]},
                        {"role": "model", "parts": [{"text": query}]},
                    ]
                }
                outfile.write(json.dumps(prepared_data) + "\n")
                count += 1
    logger.info(
        f"Successfully prepared {count} records in multi-turn chat format into '{output_jsonl_path}'."
    )
    logger.info(
        f"Sample from formatted data file:\n{pd.read_json(output_jsonl_path, lines=True).head()}\n"
    )
