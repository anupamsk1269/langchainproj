import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Dynamically locate the absolute path to the project root
ROOT_DIR = Path(__file__).resolve().parent if "__file__" in locals() else Path(os.getcwd()).resolve()
if ROOT_DIR.name == "notebooks":
    ROOT_DIR = ROOT_DIR.parent

ENV_PATH = ROOT_DIR / ".env"

# 2. Load the sensitive keys from .env into environment variables
load_dotenv(dotenv_path=ENV_PATH)

# 3. Pull sensitive keys from environment memory
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Inside your config.py file
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_ANTHROPIC_MODEL = "claude-3-sonnet-20241022"

# 4. Define non-sensitive default configurations right here in code
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Quick safeguard checks
if not OPENAI_API_KEY:
    raise ValueError(f"CRITICAL: OPENAI_API_KEY is missing from {ENV_PATH}")
if not GEMINI_API_KEY:
    raise ValueError(f"CRITICAL: GEMINI_API_KEY is missing from {ENV_PATH}")