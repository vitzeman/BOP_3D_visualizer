# Author: Vit Zeman
# Czech Technical University in Prague, Czech Institute of Informatics, Robotics and Cybernetics, Testbed for Industry 4.0

"""
Contains the prediction visualizer file for the 2D projections of the predictions.
"""
from typing import  Tuple, Union
from pathlib import Path
import socket
import copy
import os
import logging
import time
import argparse

# Third party imports
import cv2
import numpy as np
import open3d as o3d
import json

# Custom imports
from Models import Models


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="a"), logging.StreamHandler()],
)
LOGGER = logging.getLogger("2DPrediction")

LOCAL_HOST_IP = "127.0.0.1"


class PredictionVisualizerTwoD:
    """Class which renders the 2D predictions and sends them back to the client."""

    def __init__(
        self,
        path2models: Union[str, Path],
        path2split: Union[str, Path],
        s: socket.socket = None,
    ):

        self.mtl = o3d.visualization.rendering.MaterialRecord()
        self.mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        self.mtl.shader = "defaultUnlit"

        self.models = Models(path2models)

        self.image_paths = Path(path2split)
        self.s = s

        self.last_img_path:str = ""

    def render2D(
        self,
        img: np.ndarray,
        Kmx: np.ndarray,
        objects_poses: dict,
        color: tuple = (0, 255, 0),
    ) -> Tuple[np.ndarray]:
        """Renders the images with 2D overlay and contour highlighting the object instances.

        Args:
            img (np.ndarray): Input image
            Kmx (np.ndarray): Intrinsics matrix
            objects_poses (dict): Dictionary with object ids as keys and poses as values.
            color (tuple, optional): Color of the contour and overlay. Defaults to (0, 255, 0).
        """

        height, width = img.shape[:2]
        pinhole = o3d.camera.PinholeCameraIntrinsic(
            width, height, Kmx[0, 0], Kmx[1, 1], Kmx[0, 2], Kmx[1, 2]
        )
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([0.0, 0.0, 0.0, 0.0])
        renderer.scene.scene.set_sun_light([-1, -1, -1], [1.0, 1.0, 1.0], 100000)

        contour_img = copy.deepcopy(img)
        masked_img = np.zeros_like(img, dtype=np.uint8)

        all_contours = ()
        bin_mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        for e, object_pose in enumerate(objects_poses):
            # print(object_pose)
            obj_id = object_pose["obj_id"]
            Tmx = object_pose["Tmx"]

            model = self.models.get_model(obj_id)
            model.scale(1 / 1000, center=(0, 0, 0))
            model.transform(Tmx)

            name = f"model_{obj_id}_{e}"
            model.paint_uniform_color((1, 1, 1))  # white color
            renderer.scene.add_geometry(name, model, self.mtl)
            renderer.setup_camera(pinhole, np.eye(4))

            img_o3d = renderer.render_to_image()
            bgr = np.array(img_o3d)[:, :, ::-1]  # to BGR
            img_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(img_gray, 50, 255, 0)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            all_contours += contours
            cv2.drawContours(contour_img, contours, -1, color, 2)
            renderer.scene.remove_geometry(name)
            mask = (bgr > 50)[:, :, 0]
            bin_mask = np.logical_or(bin_mask, mask)
            # Paint the colored mask into the masked image
            masked_img[mask] = color

    
        return bin_mask, all_contours

    def communication_loop(self, port: int):
        """Starts the communication loop for the prediction visualizer.

        Args:
            port (int): Port number for the communication.
        """
        while True:
            bytes = ""

            while True:
                b = self.s.recv(1024).decode()
                bytes += b
                if b == "":
                    LOGGER.info("Connection closed by the client.")
                    return
                last = b[-1]
                if last == "\n":
                    break

            data = bytes
            data = json.loads(data)
            if "exit" in data.keys():
                LOGGER.info("Exiting the 2D visualizer. ")
                break

            img_path = data["image_path"]
            Kmx = np.array(data["Kmx"])
            objects_poses = data["objects_poses"]
            color = data["color"]
            color = [255 * x for x in color][::-1]
            # if self.last_img_path != img_path: # THIS IS PROBLEMATIC DUE TO THE MULTIPLE PREDICTION METHODS USED on the same image
            
            img = cv2.imread(img_path) 
            self.overlay, contours = self.render2D(img, Kmx, objects_poses, color)
            self.contours = tuple(c.tolist() for c in contours)

                    
            # else:
            #     LOGGER.info("Image path is the same as the last one. Skipping the rendering.")

            self.last_img_path = img_path
            overlay = self.overlay.tolist()
            contour_img = self.contours

            images = {"mask": overlay, "contours": contour_img}
            back = json.dumps(images).encode() + b"\n"
            LOGGER.debug("Sending back the overlay and contour images.")
            self.s.sendall(back)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the prediction visualizer scripts.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the configuration file.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        config = json.load(f)

    host = LOCAL_HOST_IP
    port = 65432 

    host = config.get("host", host)
    port = config.get("port", port)
    
    split_scene_path = config.get("split_scenes_path", None)
    models_path = config.get("models_path", None)

    assert split_scene_path is not None, "Split scene path is not provided."
    assert models_path is not None, "Models path is not provided."

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    LOGGER.info(f"Trying to connect on {host}:{port}")
    while True:
        try:
            s.connect((host, port))
            break
        except Exception as e:
            LOGGER.warning(f"Connection on {host}:{port} failed. Retrying...")
            time.sleep(1)
    

    LOGGER.info(f"Connection established on {host}:{port}")

    pv = PredictionVisualizerTwoD(models_path, split_scene_path, s=s)
    pv.communication_loop(port)
