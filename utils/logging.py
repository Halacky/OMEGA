import logging
from pathlib import Path
from datetime import datetime

def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('NinaProPipeline')
    logger.info(f"Logger setup complete. Log file: {log_file}")
    return logger

def seed_everything(seed: int = 42):
    import numpy as np
    import torch
    import os
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False