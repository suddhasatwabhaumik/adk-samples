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

# Google/ADK import
from google.adk.agents import LlmAgent
from google.genai import types

# Tool import
from . import prompt
from . import tool

# Configuration import
from ...config import config

# CONSTANTS
_MODEL = config.flash_model


# Agent definition
def get_agent():
    return LlmAgent(
        name="EvaluationAgent",
        model=_MODEL,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        description="Evaluates a fine-tuned model.",
        instruction=prompt.PROMPT,
        tools=[tool.evaluate_model],
    )
