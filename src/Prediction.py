# Author: Vit Zeman
# Czech Technical University in Prague, Czech Institute of Informatics, Robotics and Cybernetics, Testbed for Industry 4.0
""" 
Prediction class for handling the prediction from the BOP csv result file
Additionaly controls the 2D visualization window for each prediction
"""

# Native imports
import os
import json
import logging
from typing import List, Tuple, Union
from pathlib import Path

import socket

# Third party imports
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import pandas as pd


from Models import Models

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="a"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("PredictionClass")

class Prediction:
    """Handles the prediction files in csv format"""

    def __init__(
        self,
        csv_path: Union[str, Path],
        models: Models,
        connection: socket.socket = None,
        color: Tuple[float] = (0.5, 0.5, 0.5),
    ) -> None:
        """Initializes the prediction class, Loads the csv file and the default settings

        Args:
            csv_path (Union[str, Path]): Path to the csv file in BOP format, ie. methondName_datasetName-splitName.csv
            models (Models): Models class with the models in the dataset
            color (Tuple[float]): Color of the annotation objects in the scene (R,G,B) in range 0-1
        """

        # INFO: Connection used for the 2D visualization request
        if connection is not None:
            self.conn = connection

        csv_path = Path(csv_path)
        assert csv_path.exists(), f"Path {csv_path} does not exist"

        LOGGER.info(f"Loading the csv file: {csv_path}")
        df = pd.read_csv(csv_path)
        self.dataframe: pd.DataFrame = df

        file_name = csv_path.name
        self.method_name: str = file_name.split("_")[0]
        self.dataset_name: str = file_name.split("_")[1].split("-")[0]

        self.color: Tuple[float] = color

        self.models_set: Models = models

        self._is_window2D_initialized: bool = False

        self._material = o3d.visualization.rendering.MaterialRecord()
        self._material.base_color = [1.0, 1.0, 1.0, 1.0]
        self._material.shader = "defaultLit"
        self.window2D = None

        self.overlay = None
        self.contour = None

        self.contour_img = None
        self.overlay_img = None

    def _init_window2D(self, image: np.ndarray) -> None:
        """Initializes the 2D visualization window

        Args:
            image (np.ndarray): Image for the 2D visualization [HxWx3]
        """
        resolution = (image.shape[1], image.shape[0])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_o3d = o3d.geometry.Image(rgb)

        self.window2D = gui.Application.instance.create_window(
            f"2D visualization {self.method_name}", *resolution
        )
        self.window2D_widget = gui.ImageWidget(img_o3d)
        self.window2D.add_child(self.window2D_widget)
        self.window2D.show(self._checkbox.checked)

        self.window2D.set_on_close(self._igonore_close)

        self._is_window2D_initialized = True

    def _igonore_close(self) -> None:
        """Ignores the close event of the 2D visualization window"""
        pass

    def _update_window2D(self, image: np.ndarray) -> None:
        """Updates the image in the 2D visualization window

        Args:
            image (np.ndarray): Image for the 2D visualization [HxWx3]
        """
        # self.window2D.show(True)
        img_o3d = o3d.geometry.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.window2D_widget.update_image(img_o3d)
        self.window2D.post_redraw()

    def _hide_window2D(self) -> None:
        """Hides the 2D visualization if 2D visualization is not required"""
        self.window2D.show(False)

    def _show_window2D(self) -> None:
        """Shows the 2D visualization if 2D visualization is required"""
        self.window2D.show(True)

    def set_color(self, color: Tuple[float]) -> None:
        """Sets the color of the annotation objects in the scene

        Args:
            color (Tuple[float]): Color of the annotation objects in the scene (R,G,B) in range 0-1
        """
        self.color = color

    def _update_color(self, color: o3d.visualization.gui.Color) -> None:
        """Updates the color of the annotation objects in the scene

        Args:
            color (o3d.visualization.gui.Color): Color of the annotation objects in the scene (R,G,B) in range 0-1
        """
        # print(f"Updating the color to {color.red, color.green, color.blue}")
        self.color = (color.red, color.green, color.blue)

    def _load_predictions(self, scene_id: int, image_id: int) -> pd.DataFrame:
        """Loads the predictions for the given scene and image

        Args:
            scene_id (int): Index of the scene
            image_id (int): Index of the image
        """
        predictions = self.dataframe.loc[
            (self.dataframe["scene_id"] == scene_id)
            & (self.dataframe["im_id"] == int(image_id))
        ]

        return predictions


    def draw_curr_window2D(self):
        """Draws the current overlay and contour images in the 2D window, also should be fasetr for the change of color"""
        assert self.mask is not None, "Mask must be provided for the 2D visualization"
        assert self.contours is not None, "Contours must be provided for the 2D visualization"
        
        img = cv2.imread(str(self.image_path))
        color = [255 * x for x in self.color][::-1]
        masked_img = np.zeros_like(img)

        masked_img[self.mask] = color
        overlay_img = img.copy()
        overlay = cv2.addWeighted(overlay_img, 1, masked_img, 1, 0)
        
        contour_img = img.copy()
        cv2.drawContours(contour_img, self.contours, -1, color, 2)
        together = np.concatenate((overlay, contour_img), axis=0) 
        if self._is_window2D_initialized:
            self._update_window2D(together)
        else:
            self._init_window2D(together)

        self.contour_img = contour_img
        self.overlay_img = overlay


    def _save_images(self, direcory_path: Union[str, Path] = "") -> None:
        """Saves the contour and overlay images to the given directory

        Args:
            direcory_path (Union[str, Path]): Path to the directory where the images will be saved
        """
        if self.contour_img is None or self.overlay_img is None:
            LOGGER.warning("No overlay and contour to save")
            return

        dir_path = Path(direcory_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        contour_name = (
            dir_path / f"{self.dataset_name}_{self.method_name}_s{str(self.scene_id).zfill(6)}_i{str(self.image_id).zfill(6)}_contour.png"
        )
        overlay_name = (
            dir_path / f"{self.dataset_name}_{self.method_name}_s{str(self.scene_id).zfill(6)}_i{str(self.image_id).zfill(6)}_overlay.png"
        )
        if not contour_name.exists():
            cv2.imwrite(str(contour_name), self.contour_img)
        if not overlay_name.exists():    
            cv2.imwrite(str(overlay_name), self.overlay_img)

    def get_predictions(
        self,
        scene_id: int,
        image_id: int,
        plot2D: bool = False,
        Kmx: np.ndarray = None,
        image_path: Union[Path, str] = None,
    ) -> Union[
        List[o3d.geometry.TriangleMesh],
        Tuple[List[o3d.geometry.TriangleMesh], np.ndarray, np.ndarray],
    ]:
        """Returns the predictions for the given scene and image

        Args:
            scene_id (int): Index of the scene
            image_id (int): Index of the image
            plot2D (bool): If True, the 2D visualization is returned with the overlay and contour
            Kmx (np.ndarray): Camera matrix for the 2D visualization [3x3]
            image (np.ndarray): Image for the 2D visualization [HxWx3]

        Returns:
            Union[List[o3d.geometry.TriangleMesh],Tuple[List[o3d.geometry.TriangleMesh], np.ndarray, np.ndarray]: List of the annotation objects in the scene or Tuple of the annotation objects, overlay and contour if plot2D is True
        """
        self.image_path = image_path
        self.scene_id = scene_id
        self.image_id = image_id
        predictions = self._load_predictions(scene_id, image_id)

        geom_list = [] # List with open3d geometry objects used for the visualization
        objects_poses = []
        for e, row in predictions.iterrows():
            obj_id = str(row["obj_id"])
            Rmx = (
                np.array([x for x in row["R"].split(" ") if x != ""])
                .astype(np.float32)
                .reshape(3, 3)
            )
            tv = (
                np.array([x for x in row["t"].split(" ") if x != ""])
                .astype(np.float32)
                .reshape(3, 1)
            ) / 1000  # Convert to meters
            pred_model = self.models_set.get_model(obj_id)

            Tmx = np.eye(4)
            Tmx[:3, :3] = Rmx
            Tmx[:3, 3] = tv.flatten()

            pred_model.scale(1 / 1000, center=(0, 0, 0))
            pred_model.transform(Tmx)
            pred_model.paint_uniform_color(self.color)

            objects_poses.append({"obj_id": obj_id, "Tmx": Tmx.tolist()})

            geom_list.append(pred_model)

        if plot2D:
            assert Kmx is not None, "K matrix must be provided for 2D visualization"
            assert Kmx.shape == (
                3,
                3,
            ), f"Intrinsic matrix K must be 3x3 not {Kmx.shape}"
            assert image_path is not None, "Image must be provided for 2D visualization"
            assert os.path.isfile(
                image_path
            ), f"Image path {image_path} is not a valid file"

            if self.conn is not None:
                d = {
                    "image_path": str(image_path),
                    "Kmx": Kmx.tolist(),
                    "color": self.color,
                    "objects_poses": objects_poses,
                }
                data = json.dumps(d).encode() + b"\n"

                LOGGER.debug("Sending the data to the slave")
                self.conn.sendall(data)
                
                response = ""
                while True:
                    b = self.conn.recv(1024).decode()
                    response += b
                    last = b[-1]
                    if last == "\n":
                        break

                response = json.loads(response)

                mask = response.get("mask", None)
                contours = response.get("contours", None)
                contours = [np.array(c) for c in contours]

                self.mask = mask
                self.contours = contours

                if self.mask is not None and self.contours is not None:
                    self.draw_curr_window2D()
                else:
                    LOGGER.error("No overlay and contour received")
            else:
                LOGGER.error("No connection to the 2D visualization server")
                self.mask = None
                self.contours = None
        else:
            if self._is_window2D_initialized:
                self.window2D.show(False)

        return geom_list