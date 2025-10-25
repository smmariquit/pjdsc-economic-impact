"""Application settings and configuration"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / 'src'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'

# Model paths
PERSONS_MODEL_DIR = MODELS_DIR / 'persons'
HOUSES_MODEL_DIR = MODELS_DIR / 'houses'

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Application settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Constants
USD_TO_PHP = 56.0  # USD to PHP conversion rate
