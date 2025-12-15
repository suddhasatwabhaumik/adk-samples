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
"""
Simulated MCPToolbox Client.

This file contains a client that simulates the behavior of the MCPToolbox.
It loads the `tools.yaml` file and provides an interface to execute the tools
defined within it. This allows the agent to be developed against a consistent,
tool-based interface for database interactions.

This is a more accurate representation of how the agent will interact with the
real MCPToolbox library.
"""

# Standard imports
import yaml
import pandas as pd

# Google improts
from google.cloud import bigquery

# import configurations
from ..config import config

""" Toolbox outline """


class MCPToolboxClient:
    """A simulated client for interacting with tools defined in tools.yaml."""

    def __init__(self, project_id: str, tools_yaml_path: str = "tools.yaml"):
        """
        Initializes the client by loading the tools.yaml configuration
        and setting up a BigQuery client for the backend.

        Args:
            project_id: The Google Cloud project ID.
            tools_yaml_path: The path to the tools.yaml configuration file.
        """
        print("--- MCPToolboxClient: Initializing... ---")
        try:
            with open(tools_yaml_path, "r") as f:
                self.config = yaml.safe_load(f)
                print(
                    f"--- MCPToolboxClient: Loaded configuration from {tools_yaml_path} ---"
                )
        except FileNotFoundError:
            print(f"FATAL: tools.yaml not found at path: {tools_yaml_path}")
            raise
        except yaml.YAMLError as e:
            print(f"FATAL: Error parsing YAML from {tools_yaml_path}: {e}")
            raise

        self.project_id = config.project_id
        self.bq_client = bigquery.Client(project=self.project_id)
        print("--- MCPToolboxClient: BigQuery client initialized. ---")

    def execute_tool(self, tool_name: str, parameters: dict = None) -> pd.DataFrame:
        """
        Executes a tool defined in tools.yaml.

        Args:
            tool_name: The name of the tool to execute (e.g., 'execute_sql').
            parameters: A dictionary of parameters to pass to the tool.

        Returns:
            A pandas DataFrame containing the result of the tool's execution.
        """
        if parameters is None:
            parameters = {}

        print(
            f"--- MCPToolboxClient: Executing tool '{tool_name}' with params: {parameters} ---"
        )

        tool_config = self.config.get("tools", {}).get(tool_name)
        if not tool_config:
            raise ValueError(f"Tool '{tool_name}' not found in tools.yaml")

        kind = tool_config.get("kind")
        query = ""

        if kind == "bigquery-execute-sql":
            if "query" not in parameters:
                raise ValueError(
                    f"Missing required parameter 'query' for tool '{tool_name}'."
                )
            query = parameters["query"]
        elif kind == "bigquery-sql":
            statement = tool_config.get("statement", "")
            # Substitute parameters into the statement
            query = statement.format(**parameters)
        else:
            raise ValueError(f"Unsupported tool kind: {kind}")

        print(f"--- MCPToolboxClient: Executing generated query: {query[:200]}... ---")
        try:
            df = self.bq_client.query(query).to_dataframe()
            return df
        except Exception as e:
            print(
                f"--- MCPToolboxClient Error: Query execution failed for tool '{tool_name}': {e} ---"
            )
            # Return an empty DataFrame to maintain type consistency
            return pd.DataFrame()
