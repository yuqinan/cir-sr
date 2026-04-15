#!/usr/bin/env python3
"""
Simplified Teacher Evaluation Script

Clean, focused script for running teacher model evaluation.
Reduced from 726 lines to ~300 lines by extracting orchestration logic.
"""

import os
import sys
import logging
from pprint import pprint
from datetime import datetime
import hydra
from omegaconf import OmegaConf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate.teacher_evaluator import TeacherEvaluator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('teacher_eval.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="mini_sudoku", version_base=None)
def main(config):
    """Main entry point for teacher evaluation."""
    # Print resolved configuration
    print(f"MAIN CALLED AT: {datetime.now()}")  # Add this line

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    config.evaluation.teacher_model.preappend_token = "<" + config.evaluation.teacher_model.preappend_token + ">"
    
    # Initialize evaluator and run evaluation
    evaluator = TeacherEvaluator(config)
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()