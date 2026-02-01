import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5-mini-2025-08-07")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 2000))

# Path Configuration
# Get the project root directory (assuming this file is in src/)
PROJECT_ROOT = Path(__file__).parent.parent

# Default paths - can be overridden by environment variables
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for path in [RESULTS_DIR, LOG_DIR]:
    path.mkdir(parents=True, exist_ok=True)

DATA_PATH = os.getenv("DATA_PATH", str(DATA_DIR / "merged_data.csv"))
RESULTS_PATH = os.getenv("RESULTS_PATH", str(RESULTS_DIR))
LOG_PATH = os.getenv("LOG_PATH", str(LOG_DIR))

# Nature Paper Phenotypes
NATURE_PHENOTYPES = [
    'LVM (g)', 'LVEDV (mL)', 'LVEF (%)',
    'RVEDV (mL)', 'RVEF (%)',
    'LAV max (mL)', 'LAEF (%)',
    'RAV max (mL)', 'RAEF (%)',
    'AAo max area (mm2)', 'AAo distensibility (10-3 mmHg-1)',
    'DAo max area (mm2)', 'DAo distensibility (10-3 mmHg-1)'
]

# Organ Phenotype Definitions
ORGAN_PHENOTYPES = {
    'LV': [
        'LVEDV (mL)', 'LVESV (mL)', 'LVSV (mL)',
        'LVEF (%)', 'LVCO (L/min)', 'LVM (g)'
    ],
    'RV': [
        'RVEDV (mL)', 'RVESV (mL)', 'RVSV (mL)', 'RVEF (%)'
    ],
    'LA': [
        'LAV max (mL)', 'LAV min (mL)', 'LASV (mL)', 'LAEF (%)'
    ],
    'RA': [
        'RAV max (mL)', 'RAV min (mL)', 'RASV (mL)', 'RAEF (%)'
    ],
    'Aorta': [
        'AAo max area (mm2)', 'AAo min area (mm2)', 'AAo distensibility (10-3 mmHg-1)',
        'DAo max area (mm2)', 'DAo min area (mm2)', 'DAo distensibility (10-3 mmHg-1)'
    ],
    'Strain': [
        'WT_Global (mm)', 'Ecc_Global (%)', 'Err_Global (%)', 'Ell_Global (%)'
    ]
}

# Normal Ranges
NORMAL_RANGES = {
    'LVEDV (mL)': (65, 240),
    'LVEF (%)': (52, 72),
    'LVM (g)': (88, 224),
    'RVEDV (mL)': (60, 180),
    'RVEF (%)': (47, 63),
    'LAV max (mL)': (16, 64),
    'LAEF (%)': (50, 70),
    'RAV max (mL)': (25, 58),
    'RAEF (%)': (46, 68),
    'AAo distensibility (10-3 mmHg-1)': (2.0, 5.0),
    'DAo distensibility (10-3 mmHg-1)': (2.5, 5.5)
}

# Disease Related
DISEASES = [
    'Hypertension', 'High cholesterol', 'Cardiac disease', 'PVD',
    'Diabetes', 'Stroke', 'Asthma', 'COPD', 'Bronchitis',
    'Parkinson\'s', 'Dementia', 'Depression'
]

# Statistical Configuration
STATISTICAL_CONFIG = {
    'significance_level': 0.05,
    'bonferroni_correction': True,
    'fdr_correction': True,
    'min_correlation': 0.3,
    'min_sample_size': 30
}
