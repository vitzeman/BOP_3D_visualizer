# Author: Vit Zeman
# Czech Technical University in Prague, Czech Institute of Informatics, Robotics and Cybernetics, Testbed for Industry 4.0

"""
Main script to run this application, it uses suboprocesses to run
both 3D window and 2D headless renderer and socket to handle the communication.

Usage:
    python main.py -c config/example_config.json

Need to specify the `config/example_config.json` file and fill it with dataset paths.
"""

import os
import json
import logging
from pathlib import Path
import subprocess
import argparse

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="a"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser(description="Run the prediction visualizer scripts.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    return parser.parse_args()

def check_config(config_path:Path) -> None:
    """Check if the configuration file contains all the required parameters to run the visualizer scripts.

    Args:
        config_path (Path): Path to the configuration file. Example in `config/example_config.json`.

    Raises:
        ValueError: If a required parameter is missing from the configuration file.
        AssertionError: If a path does not exist.

    """    
    with open(config_path, "r") as f:
        config = json.load(f)

    required_params = ["models_path", "split_scenes_path"]
    for param in required_params:
        if param not in config:
            raise ValueError(f"Parameter {param} is missing from the configuration file.")
        
        if not Path(config[param]).exists():
            raise AssertionError(f"Path {config[param]} does not exist.")
        

    LOGGER.info("Configuration file is valid.")
    return


if __name__ == "__main__":
    args = parse_arguments()
    config_path = Path(args.config)
    # TODO: ADD TO CONFIG HOST AND PORT POSSIBIILITIES
    check_config(config_path)


    conda_env = os.environ.get("CONDA_PREFIX")
    venv_env = os.environ.get("VIRTUAL_ENV")

    if conda_env:
        python_executable = os.path.join(conda_env, "bin", "python")
    elif venv_env:
        python_executable = os.path.join(venv_env, "bin", "python")
    else:
        python_executable = "python"  # fallback to system python

    if str(config_path).startswith("/"):
        adjusted_config_path = config_path
    else:
        adjusted_config_path = Path().resolve() / config_path
    
    adjusted_config_path = str(adjusted_config_path)
    with open(adjusted_config_path, "r") as f:
        config = json.load(f)

    script1 = subprocess.Popen([python_executable, "src/predictionApp.py", "-c", adjusted_config_path])
    script2 = subprocess.Popen([python_executable, "src/predictions_2D_renderer.py", "-c", adjusted_config_path])

    script1.wait()
    script2.wait()
