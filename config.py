import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configurations
# PLANNING_MODEL = "gemini-2.0-flash"#"gemini-2.5-pro-exp-03-25"
# EXECUTION_MODEL = "gemini-2.0-flash"
# VERIFIER_MODEL = "gemini-2.0-flash"

PLANNING_MODEL = "gemini-2.5-pro-exp-03-25"#"gemini-2.5-pro-exp-03-25"
EXECUTION_MODEL = "gemini-2.0-flash"
VERIFIER_MODEL = "gemini-2.0-flash"

# Model rate Limits - Requests per minute
RATE_LIMITS = {
    "gemini-2.5-pro-exp-03-25": 5,
    "gemini-2.0-flash": 15,
    "gemini-2.0-flash-lite": 15,
    "gemini-2.0-flash-thinking-exp-01-21": 10,
    "gemini-1.5-flash": 15,
    "gemini-1.5-pro": 2
}

# Model Token Limits - Tokens per minute for each model
TOKEN_LIMITS = {
    "gemini-2.5-pro-exp-03-25": 1000000,
    "gemini-2.0-flash": 1000000,
    "gemini-2.0-flash-lite": 1000000,
    "gemini-2.0-flash-thinking-exp-01-21": 1000000,
    "gemini-1.5-flash": 1000000,
    "gemini-1.5-pro": 32000
}

# Playwright configurations
HEADLESS = False  # Set to True for production, False for development/debugging

# Agent configurations
MAX_RETRIES = 3
MAX_CONSECUTIVE_TOOL_CALLS = 20
HUMAN_IN_LOOP = True  # Set to False to run without human intervention 

# Database configurations
DB_PATH = "logs/chat_history.db"  # Can be overridden by environment variable
if os.getenv("CHAT_DB_PATH"):
    DB_PATH = os.getenv("CHAT_DB_PATH") 


# Extracted data Path
RESULTS_DIR = "results"  # Can be overridden by environment variable