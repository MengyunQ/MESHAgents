import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Any
from config import (
    OPENAI_API_KEY, DATA_PATH, RESULTS_PATH, LOG_PATH, DISEASES, GPT_MODEL
)
from agents import ChiefAgent


# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(LOG_PATH) / 'analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def json_serializer(obj):
    """Custom JSON serializer"""
    if isinstance(obj, (np.integer, np.floating)):
        return int(obj) if isinstance(obj, np.integer) else float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

async def save_results(results: Any, filepath: Path) -> None:
    """Async save analysis results"""
    try:
        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath)
        else:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=4, default=json_serializer)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results to {filepath}: {e}")
        raise

def prepare_clinical_outcomes(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Prepare clinical outcome data"""
    return {
        disease: df[disease] for disease in DISEASES if disease in df.columns
    }

async def main():
    try:
        # Create necessary directories
        for path in [RESULTS_PATH, LOG_PATH]:
            Path(path).mkdir(parents=True, exist_ok=True)

        # 1. Load data
        logger.info("Loading data...")
        data = pd.read_csv(DATA_PATH)
        logger.info(f"Data loaded: {data.shape}")
        
        # 2. Prepare clinical data
        clinical_outcomes = prepare_clinical_outcomes(data)
        logger.info(f"Clinical outcomes prepared for {len(clinical_outcomes)} diseases")

        # 3. Initialize Chief Agent
        logger.info(f"Initializing Chief Agent with model {GPT_MODEL}")
        chief_agent = ChiefAgent(OPENAI_API_KEY)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 4. Run analysis
        logger.info("Running analysis pipeline...")
        results = await chief_agent.run_analysis(data)

        # 5. Save results
        tasks = []
        # Save raw analysis results
        tasks.append(save_results(
            results, 
            Path(RESULTS_PATH) / f'analysis_results_{timestamp}.json'
        ))

        # Save phenotype importance scores
        phenotype_scores = pd.DataFrame(
            results['phenotype_importance']['top_phenotypes'], 
            columns=['Phenotype', 'Score']
        )
        tasks.append(save_results(
            phenotype_scores, 
            Path(RESULTS_PATH) / f'phenotype_scores_{timestamp}.csv'
        ))

        # Save organ-specific analysis
        for organ, organ_results in results['organ_specific'].items():
            tasks.append(save_results(
                organ_results, 
                Path(RESULTS_PATH) / f'{organ}_analysis_{timestamp}.json'
            ))

        # Save GPT analysis results
        if 'gpt_analysis' in results:
            tasks.append(save_results(
                {'interpretation': results['gpt_analysis']}, 
                Path(RESULTS_PATH) / f'gpt_analysis_{timestamp}.json'
            ))

        # Wait for all save tasks to complete
        await asyncio.gather(*tasks)

        # 6. Print main findings
        logger.info("\nMain Findings:")
        logger.info(f"Analyzed phenotypes: {len(results['phenotype_importance']['scores'])}")
        logger.info("\nTop 5 most important phenotypes:")
        logger.info(phenotype_scores.head().to_string())
        logger.info("Analysis pipeline completed successfully")
        logger.info("\nTop 30 most important phenotypes:")
        logger.info(phenotype_scores.head(30).to_string())

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())