import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Load YAML config."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config {config_path} not found. Using defaults.")
        return {}  # Fallback to module defaults

def mock_frame(shape=(480, 640, 3)):
    """Mock BGR frame for testing."""
    return np.zeros(shape, dtype=np.uint8)