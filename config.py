import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configurations
PLANNING_MODEL = "gemini-2.5-pro-exp-03-25"
EXECUTION_MODEL = "gemini-2.0-flash"
VERIFIER_MODEL = "gemini-2.0-flash"

# Playwright configurations
HEADLESS = False  # Set to True for production, False for development/debugging

# Agent configurations
MAX_RETRIES = 3
MAX_CONSECUTIVE_TOOL_CALLS = 20
HUMAN_IN_LOOP = True  # Set to False to run without human intervention 