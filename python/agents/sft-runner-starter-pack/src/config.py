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

"""Configuration mapper"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


# Configuration skeleton Class
class Config:
    def __init__(self):
        # Google Cloud Configuration
        self.project_id: Optional[str] = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv(
            "GCP_PROJECT_ID"
        )
        self.location: Optional[str] = os.getenv(
            "GOOGLE_CLOUD_LOCATION", "us-central1"
        ).lower()
        self.use_vertex_ai: bool = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "0") == "1"

        # Bigquery configurations
        self.BQ_LOCATION: Optional[str] = os.getenv("BQ_LOCATION") or os.getenv(
            "BQ_LOCATION"
        )
        self.PROJECT: Optional[str] = os.getenv("PROJECT") or os.getenv("PROJECT")
        self.DATASET: Optional[str] = os.getenv("DATASET") or os.getenv("DATASET")

        # Model and Service Configuration
        self.flash_model: str = os.getenv("FLASH_MODEL")
        self.pro_model: str = os.getenv("PRO_MODEL")
        self.bucket_name: str = os.getenv("GCS_BUCKET_NAME")

        # Parameters for overall fine tuning process
        self.initial_target_examples: int = int(os.getenv("INITIAL_TARGET_EXAMPLES"))
        self.seed_queries: str = os.getenv("SEED_QUERIES")
        self.eval_dataset: str = os.getenv("EVAL_DATASET")
        # TODO: Refactor to GCS/BigQuery Table

        # LLM Retry Parameters
        # TODO: Add in utilities/sub-agents
        self.genai_max_retries: int = int(os.getenv("MAX_RETRIES"))
        self.genai_initial_delay: int = int(os.getenv("INITIAL_DELAY"))

        # LLM Generation Parameters
        self.temperature: float = float(os.getenv("TEMPERATURE", 0.0))
        self.top_p: float = float(os.getenv("TOP_P", 1))
        self.max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", 65535))
        self.seed: int = int(os.getenv("SEED", 42))

    def validate(self) -> bool:
        """Validate that all required configuration is present."""
        if not self.project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required")
        return True

    @property
    def project_location(self) -> str:
        """Get the project location in the format required by BigQuery and Dataform."""
        return f"{self.project_id}.{self.location}"

    @property
    def vertex_project_location(self) -> str:
        """Get the project location in the format required by Vertex AI."""
        return f"{self.project_id}.{self.location}"


# Create a global config instance
config = Config()
