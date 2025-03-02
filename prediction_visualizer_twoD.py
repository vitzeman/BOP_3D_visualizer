"""Prediction visualiyer for rendering 2D representation of the prediction results."""

from typing import List, Tuple, Union
from pathlib import Path
import socket
import copy
import os
import logging
import time

import cv2
import numpy as np
import open3d as o3d
import json


from prediction_visualizer import Models


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="w"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)

LOCAL_HOST_IP = "127.0.0.1"


class PredictionVisualizerTwoD:

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

        geom_list = []
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
            cv2.drawContours(contour_img, contours, -1, color, 2)
            renderer.scene.remove_geometry(name)

            mask = (bgr > 50)[:, :, 0]
            # Paint the colored mask into the masked image
            masked_img[mask] = color

            # model.paint_uniform_color(color)
            # geom_list.append(model)

        # for model in geom_list:
        #     renderer.scene.add_geometry(model)

        # renderer.scene.set_camera(pinhole, np.eye(4))
        # img_o3d = renderer.render_to_image()

        # renderer.scene.clear_geometry() # IDK IF IT EXISTS

        overlay = cv2.addWeighted(img, 1, masked_img, 1, 0)

        return overlay, contour_img

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
                # print(b)
                last = b[-1]
                # print(last)
                if last == "\n":
                    break

            data = bytes
            # LOGGER.debug(f"Received data: {type(data)}")
            data = json.loads(data)
            # print(type(data))
            if "exit" in data.keys():
                LOGGER.info("Exiting the 2D visualizer. ")
                break

            img_path = data["image_path"]
            Kmx = np.array(data["Kmx"])
            objects_poses = data["objects_poses"]
            color = data["color"]
            color = [255 * x for x in color]

            img = cv2.imread(img_path)
            # cv2.imshow("Input", img)

            overlay, contour_img = self.render2D(img, Kmx, objects_poses, color)
            # cv2.imshow("Overlay", overlay)
            # cv2.imshow("Contour", contour_img)

            # cv2.waitKey(0)
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
            images = {"overlay": overlay.tolist(), "contour": contour_img.tolist()}
            back = json.dumps(images).encode() + b"\n"
            LOGGER.debug(f"Sending back the overlay and contour images.")
            self.s.sendall(back)


def demo_from_computer():
    split_scene_path = "/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/test"
    models_path = "/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/models"

    pv = PredictionVisualizerTwoD(models_path, split_scene_path)
    print("Visualizer initialized.")
    data_path = "/home/vit/CIIRC/bop_toolkit/data.json"
    data = json.load(open(data_path, "r"))

    img_path = data["img_path"]
    Kmx = np.array(data["Kmx"])
    objects_poses = data["object_poses"]
    color = data["color"]
    color = [255 * x for x in color]

    img = cv2.imread(img_path)
    cv2.imshow("Input", img)

    overlay, contour_img = pv.render2D(img, Kmx, objects_poses, color)

    cv2.imshow("Overlay", overlay)
    cv2.imshow("Contour", contour_img)

    cv2.waitKey(0)


if __name__ == "__main__":
    # demo_from_computer()
    host = LOCAL_HOST_IP
    port = 65432  # Determine somehow

    # TODO: THINK ABOUT HOW TO GET THIS AUTOMATICALLY
    split_scene_path = "/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/test"
    models_path = "/home/vit/CIIRC/bop_toolkit/clearpose_downsample_100_bop/models"

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    LOGGER.info(f"Trying to connect on {host}:{port}")
    while True:
        try:
            s.connect((host, port))
            break
        except:
            LOGGER.warning(f"Connection on {host}:{port} failed. Retrying...")
            time.sleep(1)

    LOGGER.info(f"Connection established on {host}:{port}")

    pv = PredictionVisualizerTwoD(models_path, split_scene_path, s=s)
    pv.communication_loop(port)
