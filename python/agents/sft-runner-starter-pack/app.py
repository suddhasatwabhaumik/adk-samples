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
"""Streamlit UI with an in-process ADK Runner."""

# Import libraries
import os
import logging
import uuid
import time
import asyncio
import warnings

# 3rd Party imports
import streamlit as st

# Google and Vertex AI Imports
import vertexai
from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

# from .src.agent import root_agent
from src.agent import root_agent

# Filter unwanted warnings
warnings.filterwarnings("ignore")

# Global Env variables
_PROJECT_ID = os.environ.get("PROJECT_ID")
_REGION = os.environ.get("GCP_LOCATION", "us-central1")
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Core Agent Logic & Initialization
@st.cache_resource
def initialize_agent():
    """
    Initializes the Vertex AI SDK, session service, and the ADK runner.
    This function is cached to prevent re-initialization on every user interaction.
    """
    logger.info("Application starting up... Initializing Vertex AI and ADK Runner.")
    try:
        vertexai.init(project=_PROJECT_ID, location=_REGION)
        logger.info("Vertex AI SDK initialized.")

        session_service = InMemorySessionService()
        logger.info("Session service created using InMemorySessionService.")

        app_name = "finetunagent-streamlit"
        runner = Runner(
            agent=root_agent, app_name=app_name, session_service=session_service
        )
        logger.info(f"Runner created for app: '{app_name}'")

        return runner, session_service
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}", exc_info=True)
        st.error(
            f"Fatal Error: Could not initialize the AI Agent. Please check the logs. Error: {e}"
        )
        st.stop()


async def _run_agent_async_task(
    runner: Runner, user_id: str, session_id: str, query: str
) -> str:
    """
    Internal asynchronous function to run the agent.
    Handles the case where runner.run returns a synchronous generator.
    """
    logger.info(
        f"Executing async agent task for user '{user_id}' in session '{session_id}' with query: '{query[:50]}...'"
    )
    content = types.Content(role="user", parts=[types.Part(text=query)])

    try:
        # Ensure session exists
        session = await runner.session_service.get_session(
            app_name=runner.app_name, user_id=user_id, session_id=session_id
        )

        if not session:
            logger.info(f"Session '{session_id}' not found. Creating a new one.")
            await runner.session_service.create_session(
                app_name=runner.app_name, user_id=user_id, session_id=session_id
            )

        # Run the agent logic
        events_generator = runner.run(
            user_id=user_id, session_id=session_id, new_message=content
        )

        final_answer = None

        # Iterate over the synchronous generator
        for event in events_generator:
            if event.is_final_response() and event.content:
                if event.content.parts and event.content.parts[0].text:
                    final_answer = event.content.parts[0].text.strip()
                    logger.info("Final answer extracted.")
                else:
                    logger.warning("Received final response with empty content parts.")
                break  # Stop after the final response

        return final_answer

    except Exception as e:
        logger.error("An error occurred during agent execution", exc_info=True)
        raise e


# Synchronous Wrapper for Agent Calls
def call_agent_sync(runner: Runner, user_id: str, session_id: str, query: str) -> str:
    """
    Synchronously calls the agent runner by managing the asyncio event loop.
    This function is responsible for ensuring the event loop is correctly
    managed for each call, especially in a Streamlit rerun context.
    """
    loop = None
    try:
        # Get the existing event loop
        loop = get_or_create_managed_loop()

        # Run the asynchronous task
        logger.info(f"Running agent task on managed loop for session '{session_id}'")
        final_answer = loop.run_until_complete(
            _run_agent_async_task(runner, user_id, session_id, query)
        )
        return final_answer

    except Exception as e:
        logger.error("Exception in call_agent_sync", exc_info=True)
        raise e


# Event Loop Management using Streamlit Session State
def get_or_create_managed_loop():
    """
    Retrieves or creates an asyncio event loop and stores it in st.session_state.
    This ensures a consistent loop is used across reruns within a single user session
    and helps in managing its lifecycle.
    """
    if (
        "asyncio_event_loop" not in st.session_state
        or st.session_state.asyncio_event_loop is None
        or st.session_state.asyncio_event_loop.is_closed()
    ):
        logger.info("Creating or resetting asyncio event loop in session state.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state.asyncio_event_loop = loop

        # Store a flag for managing this loop
        st.session_state.loop_is_managed = True

    return st.session_state.asyncio_event_loop


def close_managed_loop():
    """
    Safely closes the asyncio event loop stored in session state if it's being managed.
    """
    if "loop_is_managed" in st.session_state and st.session_state.loop_is_managed:
        if (
            "asyncio_event_loop" in st.session_state
            and st.session_state.asyncio_event_loop
        ):
            loop = st.session_state.asyncio_event_loop
            if not loop.is_closed():
                logger.info("Closing managed asyncio event loop.")
                try:
                    loop.close()
                except Exception as e:
                    logger.error(f"Error closing asyncio loop: {e}", exc_info=True)
            st.session_state.asyncio_event_loop = None  # Clear the reference
            st.session_state.loop_is_managed = False  # Reset flag


# Streamlit Page Configuration ---
st.set_page_config(
    page_title="SFT Runner Starter Pack",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Initialize Agent (runs once and is cached)
runner, session_service = initialize_agent()

# Custom CSS
st.markdown("""<style>...</style>""", unsafe_allow_html=True)  # Your CSS here


# Helper Function for Typing Animation
def stream_text(text):
    for word in text.split():
        yield word + " "
        time.sleep(0.05)
    yield ""


# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{str(uuid.uuid4())}"
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{str(uuid.uuid4())}"
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit App UI
st.title("🤖 SFT Runner Starter Pack")

# Sidebar for Configuration and Session Management
with st.sidebar:
    st.header("Chat Settings")
    st.info("Your unique identifiers help the AI agent maintain context.")
    st.markdown(f"**User ID:** `{st.session_state.user_id}`")
    st.markdown(f"**Session ID:** `{st.session_state.session_id}`")

    if st.button("🔄 Start New Chat Session"):
        # Close the current managed loop before starting a new session
        close_managed_loop()

        # Reset session state for a new chat
        st.session_state.session_id = f"session_{str(uuid.uuid4())}"  # New session ID
        st.session_state.messages = []
        st.success("New chat session started!")
        st.rerun()  # Rerun the app

    st.markdown("---")
    st.caption("Powered by Streamlit & Google ADK")

# Main Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input and Agent Interaction
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # Call the agent logic
            final_answer = call_agent_sync(
                runner=runner,
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                query=prompt,
            )

            if final_answer:
                full_response = ""
                stream_generator = stream_text(final_answer)
                for chunk in stream_generator:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)  # Final message

                # Store the final response
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            else:
                warning_message = "The agent did not provide a final answer."
                message_placeholder.warning(warning_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": warning_message}
                )

        except Exception as e:
            # Log the detailed error
            logger.error(
                f"Error in Streamlit chat interface during agent call: {e}",
                exc_info=True,
            )

            # Display error
            error_message = f"❌ An error occurred while processing your request. Please check the logs for details."
            message_placeholder.error(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )
