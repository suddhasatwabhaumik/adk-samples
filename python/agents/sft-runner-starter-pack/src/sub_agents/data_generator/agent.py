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

# import config library
from ...config import config

# Google/ADK imports
from google.adk.agents import LlmAgent
from . import prompt
from . import tool
from google.genai import types


# Agent definition
def get_agent():
    return LlmAgent(
        name="DataGeneratorAgent",
        model=config.pro_model,
        generate_content_config=types.GenerateContentConfig(
            temperature=config.temperature,
            top_p=config.top_p,
        ),
        description="Generates a synthetic dataset for fine-tuning.",
        instruction=prompt.PROMPT,
        tools=[tool.generate_synthetic_data],
    )
