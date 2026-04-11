import logging
import os
import random
from pathlib import Path
from datetime import datetime
import numpy as np
import torch


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


def seed_everything(seed: int = 42, verbose: bool = True):
    """
    Set random seeds for reproducibility across all libraries and frameworks.
    
    Args:
        seed: Random seed value
        verbose: Whether to print seed information
    
    Note:
        This function sets seeds for:
        - Python's random module
        - NumPy
        - PyTorch (CPU and CUDA)
        - CuPy (if available)
        - PYTHONHASHSEED environment variable
        
        It also configures PyTorch for deterministic behavior, which may
        impact performance but ensures reproducibility.
    """
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    if verbose:
        print(f"[SEED] Setting global random seed to: {seed}")
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        if verbose:
            print(f"[SEED] Warning: could not enable full deterministic algorithms: {e}")
 
    # Python random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Python hash seed (for hash-based operations)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set worker seed for DataLoader reproducibility
    def worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Store worker_init_fn for later use
    torch.utils.data._worker_init_fn = worker_init_fn
    
    # CuPy (if available)
    try:
        import cupy as cp
        cp.random.seed(seed)
        if verbose:
            print(f"[SEED] CuPy seed set to: {seed}")
    except ImportError:
        pass
    
    if verbose:
        print(f"[SEED] All random seeds successfully set to: {seed}")
        print(f"[SEED] PyTorch deterministic mode: enabled")
        print(f"[SEED] PyTorch cudnn benchmark: disabled")


def get_worker_init_fn(base_seed: int = 42):
    """
    Get a worker initialization function for PyTorch DataLoader.
    This ensures each worker has a different but reproducible seed.
    
    Args:
        base_seed: Base seed value
    
    Returns:
        Worker initialization function
    
    Example:
        >>> dataloader = DataLoader(
        ...     dataset, 
        ...     batch_size=32,
        ...     num_workers=4,
        ...     worker_init_fn=get_worker_init_fn(42)
        ... )
    """
    def worker_init_fn(worker_id):
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    
    return worker_init_fn