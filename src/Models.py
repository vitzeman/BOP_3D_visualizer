# Author: Vit Zeman
# Czech Technical University in Prague, Czech Institute of Informatics, Robotics and Cybernetics, Testbed for Industry 4.0
""" 
This file contains the Models class which is used to handle the models in the dataset

"""

# Native imports
import os
import logging
import glob
from typing import Union
from pathlib import Path
import copy

# Third party imports
import open3d as o3d 
from tqdm import tqdm 

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="w"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

class Models:
    """Handles the models in the dataset"""

    def __init__(self, models_path: Union[str, Path]) -> None:
        """Initializes the models class, loads the models from the path

        Args:
            models_path (Union[str, Path]): Path to the models in the dataset
        """
        assert os.path.isdir(
            models_path
        ), f"Path {models_path} is not a valid directory"
        self.models_path = Path(models_path)

        self.models = {}
        self.load_models()

    def load_models(self) -> None:
        # o3d.io.read_triangle_model(model_path) # SHOULD BE USED FOR THIS
        """Loads the models from the path"""
        models_paths = sorted(glob.glob(str(self.models_path / "*.ply")))
        for model_path in tqdm(
            models_paths, desc="Loading models", leave=False, ncols=100
        ):
            model_name = os.path.basename(model_path).split(".")[0]
            if model_name.isdigit():
                model_name = str(int(model_name))
            else:
                model_name = model_name.split("_")[1]
                model_name = str(int(model_name))

            # TODO: maybe add scaling of the model to the meter scale here so it is done once IDK
            model = o3d.io.read_triangle_mesh(model_path)
            model.compute_vertex_normals()
            self.models[model_name] = model

    def get_model(self, model_name: str) -> o3d.geometry.TriangleMesh:
        """Returns the model with the given name from the dataset

        Args:
            model_name (str): Name of the model

        Returns:
            o3d.geometry.TriangleMesh: The model
        """
        assert (
            model_name in self.models.keys()
        ), f"Model {model_name} not found in the dataset"
        model = copy.deepcopy(self.models[model_name])
        return model