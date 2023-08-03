from pathlib import Path
import yaml

LOG_DIR = "logs"

def ensure_exists(p: Path) -> Path:
    """
    Helper to ensure a directory exists.
    """
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def arch_path(config_file) -> Path:
    """
    Construct a path based on the name of a configuration file eg. 'saved/EfficientNet'
    """
    config = yaml.safe_load(open(config_file))
    p = Path(config['results_dir']) / config['name']
    return ensure_exists(p)

def log_path(config_file) -> Path:
    p = arch_path(config_file) / LOG_DIR
    return ensure_exists(p)
