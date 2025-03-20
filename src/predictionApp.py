# Author: Vit Zeman
# Czech Technical University in Prague, Czech Institute of Informatics, Robotics and Cybernetics, Testbed for Industry 4.0

"""
Visualization of the prediction results from the csv file
Shows the ground truth and the predicted poses of the objects in 3D scene
Additionally allows to visualize the 2D images with the overlay of the predictions
"""

# Native imports
import os
import json
import warnings
import argparse
import logging
import sys
import glob
from dataclasses import dataclass
from typing import List, Tuple, Union
from pathlib import Path
import copy
import threading
import socket


# Third party imports
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import open3d.visualization.gui as gui
import pandas as pd
from tqdm import tqdm

# Custom imports
from Models import Models
from Prediction import Prediction

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | l: %(lineno)s | %(message)s",
    handlers=[logging.FileHandler("logs/log.log", mode="w"), logging.StreamHandler()],
)
LOGGER = logging.getLogger(__name__)


# Define these colors form https://sashamaps.net/docs/resources/20-colors/
COLORS_RGB_U8 = [
    # (230, 25, 75), # RED # Excluding due to to usage as the GT
    # (60, 180, 75), # GREEN # Excludint due to to usage as the GT
    (255, 225, 25),  # YELLOW
    (0, 130, 200),  # BLUE
    (245, 130, 48),  # ORANGE
    (145, 30, 180),  # PURPLE
    (70, 240, 240),  # CYAN
    (240, 50, 230),  # MAGENTA
    (210, 245, 60),  # LIME
    (250, 190, 212),  # PINK
    (0, 128, 128),  # TEAL
    (220, 190, 255),  # LAVENDER
    (170, 110, 40),  # BROWN
    (255, 250, 200),  # BEIGE
    (128, 0, 0),  # MAROON
    (170, 255, 195),  # MINT
    (128, 128, 0),  # OLIVE
    (255, 215, 180),  # APRICOT
    (0, 0, 128),  # NAVY
    # (128, 128, 128),# GREY # Excluding due the basic
    # (255, 255, 255), # WHITE # Excluding due the basic
    # (0, 0, 0), # BLACK # Excluding due the basic
]

def random_color() -> Tuple[float, float, float]:
    """Generates a random color

    Returns:
        Tuple[float, float, float]: Random color in RGB format [0-1]
    """
    return (np.random.rand(), np.random.rand(), np.random.rand())


COLORS_RGB_F = [(x[0] / 255, x[1] / 255, x[2] / 255) for x in COLORS_RGB_U8]
LOCAL_HOST_IP = "127.0.0.1"


class Dataset:
    """Handles the dataset paths and the csv file and provides the access to the data"""

    def __init__(
        self,
        dataset_name: str,
        scenes_path: Union[str, Path],
        models_path: Union[str, Path],
        csv_paths: List[Union[str, Path]] = [],
    ) -> None:
        assert os.path.isdir(
            scenes_path
        ), f"Path {scenes_path} is not a valid directory"
        assert os.path.isdir(
            models_path
        ), f"Path {models_path} is not a valid directory"
        assert type(csv_paths) == list, "The csv_paths must be a list of paths"

        self.dataset_name:str = dataset_name
        self.scenes_path = scenes_path
        self.models_path = models_path
        # self.csv_path = csv_path
        self.csv_paths = csv_paths

        self.data_frames = []
        self.names = []
        for csv_path in csv_paths:
            assert os.path.isfile(csv_path) and os.path.basename(csv_path).endswith(
                ".csv"
            ), f"Path {csv_path} is not a valid csv file"
            self.data_frames.append(pd.read_csv(csv_path))
            self.names.append(os.path.basename(csv_path).split(".")[0].split("_")[0])

class Settings:
    UNLIT = "defaultUnlit"
    LIT = "defaultLit"

    def __init__(self):
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.highlight_obj = True

        self.apply_material = True  # clear to False after processing

        self.scene_material = rendering.MaterialRecord()
        self.scene_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.scene_material.shader = Settings.LIT

        self.annotation_obj_material = rendering.MaterialRecord()
        self.annotation_obj_material.base_color = [1.0, 1.0, 1.0, 1.0]
        self.annotation_obj_material.shader = Settings.LIT


class AppWindow:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    MATERIAL_NAMES = ["Unlit"]
    MATERIAL_SHADERS = [Settings.UNLIT]

    def __init__(
        self,
        scene: Dataset,
        connection: socket.socket = None,
        resolution: Tuple[int, int] = (int(1080/3*4), 1080),
        save_image_path: Union[str, Path] = Path("images"),
        camera_name: str = None,
    ) -> None:
        """Initializes the main application window and the settings

        Args:
            scene (Dataset): Dataset class with the dataset paths and the csv files
            connection (socket.socket, optional): Connection to the 2D visualization server. Defaults to None.
            resolution (Tuple[int, int], optional): Resolution of the main window. Defaults to (int(1080/3*4), 1080).
            save_image_path (Union[str, Path], optional): Path to the directory where the images will be saved. Defaults to Path("images").
            camera_name (str, optional): Camera name used in the bop dataset such as xyzibd, where there are multiple test image sets, named gray_xyz, rgb_realsense. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
        """        
        if connection is not None:
            self.connection = connection

        self.camera_name = camera_name
        self._depth_dir_name = "depth"
        if camera_name:
            self._depth_dir_name = f"{self._depth_dir_name}_{camera_name}" 

        self.scene: Dataset = scene
        self._reset_camera_view: bool = True
        # >>> Load the models >>>
        models = Models(scene.models_path)
        self.models_set: Models = models
        # <<< Load the models <<<

        # >>> Load the predictions >>>
        csv_paths = scene.csv_paths
        predictions = []
        for e, csv_path in enumerate(csv_paths):
            prediction = Prediction(csv_path, models, connection)
            prediction.set_color(COLORS_RGB_F[e] if e < len(COLORS_RGB_F) else random_color())
            predictions.append(prediction)
        self.predictions: list[Prediction] = predictions

        # <<< Load the predictions <<<

        # >>> Path preparation >>>
        self.scenes_path = Path(self.scene.scenes_path)
        self.scenes_names_l = sorted(os.listdir(self.scenes_path))
        self.max_scene_num = len(self.scenes_names_l)
        self.cur_scene_id = 0
        self.cur_scene_name = self.scenes_names_l[self.cur_scene_id]

        self._image_modality = "rgb" # TODO: ADD CHECKING BASED ON THE CAMERA NAME
        path2images = self.scenes_path / self.cur_scene_name / self._image_modality
        

        # INFO: Check if the RGB images are present else try to find the gray images
        if not path2images.exists():
            LOGGER.warning(f"No images found in {path2images} - trying to find the gray images")
            self._image_modality = "gray" # TODO: ADD CHECKING BASED ON THE CAMERA NAME
            path2images = self.scenes_path / self.cur_scene_name / self._image_modality
            if not path2images.exists():
                raise ValueError(f"No images found in {path2images}")

        self.cur_scene_img_names_l = sorted(
            os.listdir(path2images)
        )
        self.cur_image_id = 0

        if len(self.scenes_names_l) == 0:
            raise ValueError(f"No scenes found in {self.scenes_path}")
        # <<< Path preparation <<<

        self.settings = Settings()

        self.main_window = gui.Application.instance.create_window(
            "Prediction Visualizer", *resolution,0,0
        )
        print(type(self.main_window))
        mw = self.main_window
        mw.set_on_close(self._on_close_mw)

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(mw.renderer)

        em = mw.theme.font_size

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        )

        # >>> View control >>>
        view_ctrls = gui.CollapsableVert("View control", 0.33*em, gui.Margins(em, 0, 0, 0))
        view_ctrls.set_is_open(True)

        self._reset_camera_view_button = gui.Button("Reset camera view")
        self._reset_camera_view_button.horizontal_padding_em = 0.8
        self._reset_camera_view_button.vertical_padding_em = 0.1
        self._reset_camera_view_button.set_on_clicked(self._on_reset_camera_view)
        view_ctrls.add_child(self._reset_camera_view_button)

        self._show_pointcloud = gui.Checkbox("Show point cloud")
        self._show_pointcloud.set_on_checked(self._on_show_pointcloud)
        self._show_pointcloud.checked = True
        view_ctrls.add_child(self._show_pointcloud)

        self._show_axes = gui.Checkbox("Show camera axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._show_2D = gui.Checkbox("Show 2D overlay")
        self._show_2D.set_on_checked(self._on_show_2D)
        view_ctrls.add_child(self._show_2D)

        self._highlight_obj = gui.Checkbox("Highligh annotation objects")
        self._highlight_obj.set_on_checked(self._on_highlight_obj)
        # view_ctrls.add_child(self._highlight_obj) #

        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 5)
        self._point_size.set_on_value_changed(self._on_point_size)

        grid = gui.VGrid(2, 0.25 * em)
        grid.add_child(gui.Label("Point size"))
        grid.add_child(self._point_size)
        view_ctrls.add_child(grid)

        self._settings_panel.add_child(view_ctrls)
        # <<< View control <<<

        # This I do not understand
        mw.set_on_layout(self._on_layout)
        mw.add_child(self._scene)
        mw.add_child(self._settings_panel)

        # >>> Scene control >>>
        self._scene_control = gui.CollapsableVert(
            "Scene Control", 0.33 * em, gui.Margins(em, 0, 0, 0)
        )
        self._scene_control.set_is_open(True)

        # >>> Scene conroll >>>
        self._samples_buttons_label = gui.Label("Scene:")
        self._pre_sample_button = gui.Button("<")
        self._pre_sample_button.horizontal_padding_em = 0.4
        self._pre_sample_button.vertical_padding_em = 0
        self._pre_sample_button.set_on_clicked(self._on_previous_scene)

        self._next_sample_button = gui.Button(">")
        self._next_sample_button.horizontal_padding_em = 0.4
        self._next_sample_button.vertical_padding_em = 0
        self._next_sample_button.set_on_clicked(self._on_next_scene)

        self._scene_number = gui.Label("Scene:")
        self._scene_combox = gui.Combobox()
        self._init_scene_combox()
        self._scene_combox.set_on_selection_changed(self._on_scene_changed)

        h = gui.Horiz(1 * em)  # row 1
        h.add_child(self._scene_number)
        h.add_child(self._pre_sample_button)
        h.add_child(self._scene_combox)
        h.add_child(self._next_sample_button)
        self._scene_control.add_child(h)

        # <<< Scene control <<<

        # >>> Image control >>>
        self._images_buttons_label = gui.Label("Images:")
        self._pre_image_button = gui.Button("<")
        self._pre_image_button.horizontal_padding_em = 0.4
        self._pre_image_button.vertical_padding_em = 0
        self._pre_image_button.set_on_clicked(self._on_previous_image)

        self._next_image_button = gui.Button(">")
        self._next_image_button.horizontal_padding_em = 0.4
        self._next_image_button.vertical_padding_em = 0
        # self._next_image_button.background_color = gui.Color(0, 0.8, 0) # Will make the button green
        self._next_image_button.set_on_clicked(self._on_next_image)

        self._image_number = gui.Label("Image:")
        self._image_combox = gui.Combobox()
        self._init_image_combox()
        self._image_combox.set_on_selection_changed(self._on_image_changed)

        h = gui.Horiz(1 * em)  # row 1
        h.add_child(self._image_number)
        h.add_child(self._pre_image_button)
        h.add_child(self._image_combox)
        h.add_child(self._next_image_button)
        self._scene_control.add_child(h)
        # <<< Image control <<<

        # Adds the control panel
        self._settings_panel.add_child(self._scene_control)

        # >>> Visualization control >>>
        visualization_control = gui.CollapsableVert(
            "Visualization Control", 0.33 * em, gui.Margins(em, 0, 0, 0)
        )
        visualization_control.set_is_open(True)

        self._reload_button = gui.Button("Update colors")
        self._reload_button.horizontal_padding_em = 0.8
        self._reload_button.vertical_padding_em = 0
        self._reload_button.set_on_clicked(self._on_reload_colors)
        visualization_control.add_child(self._reload_button)

        self._show_ground_truth = gui.Checkbox("Ground truth")
        self._show_ground_truth.checked = True 
        self._show_ground_truth.set_on_checked(self._on_show_ground_truth)

        # self._GT_color = (60 / 255, 180 / 255, 75 / 255)  # GREEN default color
        self._GT_color = (0.,1.,0.)
        self._GT_color_picker = gui.ColorEdit()
        self._GT_color_picker.color_value = gui.Color(*self._GT_color)
        self._GT_color_picker.set_on_value_changed(self._gt_color_changed)
        h = gui.Horiz(1 * em)
        h.add_child(self._show_ground_truth)
        h.add_child(self._GT_color_picker)

        visualization_control.add_child(h)
        # >>> Prediction visualization control>>>
        for prediction in self.predictions:
            prediction._checkbox = gui.Checkbox(prediction.method_name)
            prediction._checkbox.checked = True
            prediction._checkbox.set_on_checked(self._on_show_predictions)

            # TODO: Maybe find some way how to make width it consistant in GUI
            prediction._color_picker = gui.ColorEdit()
            prediction._color_picker.color_value = gui.Color(*prediction.color)
            prediction._color_picker.set_on_value_changed(prediction._update_color)

            h = gui.Horiz(1 * em)
            h.add_child(prediction._checkbox)
            h.add_child(prediction._color_picker)
            visualization_control.add_child(h)
        # <<< Prediction visualization cotrol<<<

        # >>> save images >>>
        self.save_image_path = Path(save_image_path)
        self.save_image_path.mkdir(parents=True, exist_ok=True)
        self.viewpoits_captured = 0
        self._save_images_button = gui.Button("Save images")
        self._save_images_button.horizontal_padding_em = 0.8
        self._save_images_button.vertical_padding_em = 0
        self._save_images_button.set_on_clicked(self._on_save_images)
        visualization_control.add_child(self._save_images_button)
        # <<< save images <<<

        self._settings_panel.add_child(visualization_control)

        # <<< Visualization control <<<

        # >>> Menu >>>
        if gui.Application.instance.menubar is None:
            file_menu = gui.Menu()
            file_menu.add_separator()
            file_menu.add_item("Quit", AppWindow.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.set_checked(AppWindow.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", AppWindow.MENU_ABOUT)

            menu = gui.Menu()
            menu.add_menu("File", file_menu)
            menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        mw.set_on_menu_item_activated(AppWindow.MENU_QUIT, self._on_menu_quit)
        mw.set_on_menu_item_activated(AppWindow.MENU_ABOUT, self._on_menu_about)
        # <<< Menu <<<

        self._on_point_size(1)  # set default size to 1

        self._apply_settings()

        self._annotation_scene = None

        self._left_shift_modifier = False

        self._img_window_initialized = False

    def _on_reset_camera_view(self):
        """Resets the camera view"""
        center = np.array([0, 0, 0])
        eye = center + np.array([0, 0, -0.5])
        up = np.array([0, -1, 0])
        self._scene.look_at(center, eye, up)

    def _init_window2D(self, image: np.ndarray) -> None:
        """ Initializes the 2D visualization window

        Args:
            image (np.ndarray): Image for the 2D visualization [HxWx3]
        """
        resolution = (image.shape[1], image.shape[0])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_o3d = o3d.geometry.Image(rgb)
        self.rgb_window = gui.Application.instance.create_window(
            "RGB Image", *resolution
        )
        self.rgb_window_widget = gui.ImageWidget(img_o3d)
        self.rgb_window.add_child(self.rgb_window_widget)
        self.rgb_window.show(self._show_2D.checked)

        self.rgb_window.set_on_close(self._igonore_close)
        
        self._img_window_initialized = True

    def _igonore_close(self) -> None:
        """Ignores the close event of the 2D visualization window"""
        pass

    def _on_show_pointcloud(self, show):
        print("TODO Implement the point cloud visualization")

        self._update_scene()
        pass

    def _init_scene_combox(self):
        """Initializes the scene combobox with the scene names"""
        for name in self.scenes_names_l:
            self._scene_combox.add_item(name)

    def _update_scene_combox(self):
        """Updates the selected"""
        self._scene_combox.selected_index = self.cur_scene_id
        self._scene_combox.selected_text = self.scenes_names_l[self.cur_scene_id]

        # NEED TO ALSO UPDATE THE IMAGE COMBOX WITH THE NEW IMAGES
        self.cur_scene_name = self.scenes_names_l[self.cur_scene_id]
        self.cur_scene_img_names_l = sorted(
            os.listdir(self.scenes_path / self.cur_scene_name / "rgb")
        )
        self._update_image_combox()

    def _init_image_combox(self):
        """Initializes the image combobox with the image names"""
        for name in self.cur_scene_img_names_l:
            self._image_combox.add_item(name.split(".")[0])

    def _update_image_combox(self):
        """Updates the image combobox with the image names for the new scene"""
        self._image_combox.clear_items()
        for name in self.cur_scene_img_names_l:
            self._image_combox.add_item(name.split(".")[0])

        self._image_combox.selected_index = 0

    def _update_image_combox_selection(self):
        """Updates the selected image in the combobox"""
        self._image_combox.selected_index = self.cur_image_id
        self._image_combox.selected_text = self.cur_scene_img_names_l[
            self.cur_image_id
        ].split(".")[0]

    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red,
            self.settings.bg_color.green,
            self.settings.bg_color.blue,
            self.settings.bg_color.alpha,
        ]
        # print(bg_color)
        # bg_color = [1,1,1,1]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            self._scene.scene.modify_geometry_material(
                "point_cloud", self.settings.scene_material
            )
            self.settings.apply_material = False

        self._show_axes.checked = self.settings.show_axes
        self._highlight_obj.checked = self.settings.highlight_obj
        self._point_size.double_value = self.settings.scene_material.point_size

    def _on_menu_quit(self):
        """Quits the application"""
        gui.Application.instance.quit()

    def _on_close_mw(self):
        """Close the main window and all others
        """      
        gui.Application.instance.quit()

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.main_window.theme.font_size
        dlg = gui.Dialog("Prediction visualization")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Prediction visualization"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_show_2D(self, show):
        print("TODO Implement the 2D visualization")
        self.rgb_window.show(show)

        self._update_scene()

    # THESE NEED TO BE REWRITEN BASED on the indexes not just 1,2,3 etc
    def _on_next_scene(self):
        candidate_scene_id = self.cur_scene_id + 1
        if candidate_scene_id >= self.max_scene_num:
            print("Already at the last scene")
            return
        elif candidate_scene_id < 0:
            print("Already at the first scene")
            return

        self.cur_scene_id = candidate_scene_id
        self.cur_image_id = 0  # Reset the image index

        self.cur_scene_name = self.scenes_names_l[self.cur_scene_id]
        self.cur_scene_img_names_l = sorted(
            os.listdir(self.scenes_path / self.cur_scene_name / "rgb")
        )

        self._reset_camera_view = True
        self._update_scene_combox()
        self._update_scene()

    def _on_previous_scene(self):
        candidate_scene_id = self.cur_scene_id - 1
        if candidate_scene_id < 0:
            print("Already at the first scene")
            return
        elif candidate_scene_id >= self.max_scene_num:  # Should not happen
            print("Already at the last scene")
            return

        self.cur_scene_id = candidate_scene_id
        self.cur_image_id = 0

        self.cur_scene_name = self.scenes_names_l[self.cur_scene_id]
        self.cur_scene_img_names_l = sorted(
            os.listdir(self.scenes_path / self.cur_scene_name / "rgb")
        )

        self._reset_camera_view = True
        self._update_scene_combox()
        self._update_scene()

    def _on_scene_changed(self, name, index):
        print(index, name)
        self.cur_scene_id = index
        self.cur_scene_name = self.scenes_names_l[self.cur_scene_id]

        self.cur_scene_img_names_l = sorted(
            os.listdir(self.scenes_path / self.cur_scene_name / "rgb")
        )
        self.cur_image_id = 0

        self._image_combox.clear_items()
        for name in self.cur_scene_img_names_l:
            self._image_combox.add_item(name.split(".")[0])

        self._image_combox.selected_index = self.cur_image_id
        self._reset_camera_view = True

        self.scene_load(self.cur_scene_id, self.cur_image_id)

    def _on_next_image(self):
        candidate_image_id = self.cur_image_id + 1
        if candidate_image_id >= len(self.cur_scene_img_names_l):
            print("Already at the last image")
            return
        elif candidate_image_id < 0:
            print("Already at the first image")
            return

        self._reset_camera_view = True
        self.cur_image_id = candidate_image_id
        self._update_image_combox_selection()
        self._update_scene()

    def _on_previous_image(self):
        candidate_image_id = self.cur_image_id - 1
        if candidate_image_id < 0:
            print("Already at the first image")
            return
        elif candidate_image_id >= len(self.cur_scene_img_names_l):
            print("Already at the last image")
            return

        self._reset_camera_view = True
        self.cur_image_id = candidate_image_id
        self._update_image_combox_selection()
        self._update_scene()

    def _on_image_changed(self, name, index):
        self.cur_image_id = index
        self._reset_camera_view = True
        self.scene_load(self.cur_scene_id, self.cur_image_id)

    def _on_reload_colors(self): 
        """Reloads the visualization colors"""

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultLit"

        for prediction in self.predictions:
            if prediction._checkbox.checked: # Draw only if the checkbox is checked
                if self._show_2D.checked:
                    prediction.draw_curr_window2D()

                for pred_name, pred_model in zip(prediction._current_pred_names, prediction._current_pred_models):
                    self._scene.scene.remove_geometry(pred_name)
                    pred_model.paint_uniform_color(prediction.color)
                    self._scene.scene.add_geometry(pred_name, pred_model, mtl)

        if self._show_ground_truth.checked:
            for gt_name, gt_model in zip(self._current_gt_names, self._current_gt_models):
                self._scene.scene.remove_geometry(gt_name)
                gt_model.paint_uniform_color(self._GT_color)
                self._scene.scene.add_geometry(gt_name, gt_model, mtl)

    def _gt_color_changed(self, color: o3d.visualization.gui.Color):
        """Updates the ground truth color based on the user input. 

        To redraw the scene press the "Update colors" button in the visualization control

        Args:
            color (o3d.visualization.gui.Color): Color in the format (R,G,B)
        """
        rgb = (color.red, color.green, color.blue)
        self._GT_color = rgb

    def _on_show_ground_truth(self, show):
        if show:
            mtl = o3d.visualization.rendering.MaterialRecord()
            mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
            mtl.shader = "defaultLit"
            for gt_name, gt_model in zip(self._current_gt_names, self._current_gt_models):
                self._scene.scene.remove_geometry(gt_name)
                gt_model.paint_uniform_color(self._GT_color)
                self._scene.scene.add_geometry(gt_name, gt_model, mtl)
        else:
            for gt_name in self._current_gt_names:
                self._scene.scene.remove_geometry(gt_name)
            

    def _on_show_predictions(self, show):
        """Serves as a callback for the prediction checkboxes, Removes and shows the predictions based on the checkbox state"""
        print("update the scene")
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultLit"
        for prediction in self.predictions:
            if prediction._checkbox.checked:
                for pred_name, pred_model in zip(prediction._current_pred_names, prediction._current_pred_models):
                    self._scene.scene.remove_geometry(pred_name)
                    pred_model.paint_uniform_color(prediction.color)
                    self._scene.scene.add_geometry(pred_name, pred_model, mtl)
                if prediction.window2D is not None and self._show_2D.checked:
                    prediction._show_window2D()
                
            else:
                if prediction.window2D is not None:
                    prediction._hide_window2D()
                for pred_name in prediction._current_pred_names:
                    self._scene.scene.remove_geometry(pred_name)

    def _update_scene(self):
        self.scene_load(self.cur_scene_id, self.cur_image_id)

    def _on_layout(self, layout_context):
        r = self.main_window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height,
        )
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _on_point_size(self, size):
        self.settings.scene_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_highlight_obj(self, light):
        self.settings.highlight_obj = light
        if light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
        elif not light:
            self.settings.annotation_obj_material.base_color = [0.9, 0.9, 0.9, 1.0]

        self._apply_settings()

        # update current object visualization
        meshes = self._annotation_scene.get_objects()
        for mesh in meshes:
            self._scene.scene.modify_geometry_material(
                mesh.obj_name, self.settings.annotation_obj_material
            )

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()
        pass

    def _on_save_images(self):
        LOGGER.info(f"Saving images for {self.scene.dataset_name} scene: {self._bop_scene_id} image: {self._bop_img_id}")
        image_name = f"{self.scene.dataset_name}_3Dvis_s{str(self._bop_scene_id).zfill(6)}_i{str(self._bop_img_id).zfill(6)}_v{str(self.viewpoits_captured).zfill(2)}"
        scene_img_path = self.save_image_path / f"{image_name}.png"
        def save_image_callback(image):
            img = np.array(image)[:, :, ::-1]  # Converts to BGR
            white_mask = (img[:, :, 0] > 230) & (img[:, :, 1] > 230) & (img[:, :, 2] > 230)
            img[white_mask] = 255
            cv2.imwrite(str(scene_img_path), img)

        self._scene.scene.scene.render_to_image(save_image_callback)
        self.viewpoits_captured += 1

        if self._show_2D.checked:
            for prediction in self.predictions:
                prediction._save_images(self.save_image_path)

        inference_image_name = f"{self.scene.dataset_name}_2Dimg_s{str(self._bop_scene_id).zfill(6)}_i{str(self._bop_img_id).zfill(6)}"
        inference_img_path = self.save_image_path / f"{inference_image_name}.png"
        if not inference_img_path.exists():
            cv2.imwrite(str(inference_img_path), self.rgb)


    def _make_point_cloud(self, rgb_img, depth_img, cam_K):
        # convert images to open3d types
        rgb_img_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))
        depth_img_o3d = o3d.geometry.Image(depth_img)

        # convert image to point cloud
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            rgb_img.shape[0],
            rgb_img.shape[1],
            cam_K[0, 0],
            cam_K[1, 1],
            cam_K[0, 2],
            cam_K[1, 2],
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img_o3d, depth_img_o3d, depth_scale=1, convert_rgb_to_intensity=False
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        return pcd

    def _add_gt3D_to_scene(self, gt: list) -> None:
        """Adds the ground truth mesh model to the scene including the transformation

        Args:
            gt (list): List with the ground truth annotations of the BOP format
        """

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultLit"
        self._current_gt_count = {}
        self._current_gt_names = []
        self._current_gt_models = []
        for annotation in tqdm(
            gt, desc="Adding GT to the scene", leave=False, ncols=100
        ):
            obj_id = str(annotation["obj_id"])
            Rmx = np.array(annotation["cam_R_m2c"]).reshape(3, 3)
            tmx = np.array(annotation["cam_t_m2c"]).reshape(3, 1) / 1000

            Tmx = np.eye(4)
            Tmx[:3, :3] = Rmx
            Tmx[:3, 3] = tmx.flatten()

            if obj_id not in self.models_set.models.keys():
                LOGGER.error(f"Model {obj_id} not found in the dataset - Skipping it")
                continue

            model = self.models_set.get_model(obj_id)

            model.scale(1 / 1000, center=(0, 0, 0))
            model.transform(Tmx)
            model.paint_uniform_color(self._GT_color)

            num = self._current_gt_count.get(obj_id, 0)
            self._current_gt_count[obj_id] = num + 1
            model_name = f"GT_{obj_id}-{num}"
            self._scene.scene.add_geometry(model_name, model, mtl)
            self._current_gt_names.append(model_name)
            self._current_gt_models.append(model)

    def _add_predictions_to_scene(self, scene_idx: int, image_idx: int) -> None:
        """Adds the predictions to the scene

        Args:
            scene_idx (int): Index of the scene
            image_idx (int): Index of the image
        """

        for prediction in self.predictions:
            if not self._show_2D.checked:
                prediction_objs = prediction.get_predictions(scene_idx, image_idx)

            else:
                prediction_objs = prediction.get_predictions(
                    scene_idx,
                    image_idx,
                    plot2D=True,
                    Kmx=self.Kmx,
                    image_path=self.rgb_path,
                )

            LOGGER.info(
                f"Adding {prediction.method_name} predictions to the scene, {len(prediction_objs)} objects"
            )

            prediction._current_pred_models = prediction_objs
            prediction._current_pred_names = []

            for e, obj in enumerate(prediction_objs):
                model_name = f"{prediction.method_name}_{e}"
                prediction._current_pred_names.append(model_name)
                LOGGER.debug(f"Adding {model_name} to the scene")
                # TODO: Figure out why sometimes the model is not able to be givent to the scene
                try:
                    if prediction._checkbox.checked:
                        self._scene.scene.add_geometry(
                            model_name, obj, prediction._material
                        )

                except Exception as e:
                    LOGGER.warning(
                        f"Error adding {model_name} to the scene - Skipping it"
                    )
                    LOGGER.warning(e)
                    continue

    def scene_load(self, scene_idx: int, image_idx: int) -> None:
        """Function to load the certain scene and image from the dataset

        Args:
            scene_idx (int): Index of the scene
            image_idx (int): Index of the image

        Raises:
            ValueError: If the scene or image index is out of bounds
        """ """"""
        self._annotation_changed = False

        self._scene.scene.clear_geometry()

        # >>> Limiting the scene index >>>
        if scene_idx < 0:
            scene_idx = 0
        elif scene_idx >= self.max_scene_num:
            scene_idx = self.max_scene_num - 1
        else: 
            self.viewpoits_captured = 0 # Reset the viewpoits count
        # <<< Limiting the scene index <<<
        
        scene_path = self.scenes_path / self.scenes_names_l[scene_idx]

        image_name = int(self.cur_scene_img_names_l[image_idx].split(".")[0])
        LOGGER.info(
            f"Loading scene_idx {scene_idx} and image_idx {image_idx} ~ {image_name}"
        )

        # >>> CAMERA PARAMETERS >>>
        camera_params_path = scene_path / "scene_camera.json"
        if self.camera_name is not None:
            # FOR XYZIBD da
            camera_params_path = scene_path / f"scene_camera_{self.camera_name}.json"
        # camera_params_path = scene_path / "scene_camera_xyz.json"
        if not camera_params_path.exists():
            LOGGER.error(
                f"Camera parameters not found in {camera_params_path} - Skipping the scene"
            )
            return

        with open(camera_params_path) as f:
            camera_params = json.load(f)
            key = str(int(image_name))
            LOGGER.debug(f"Loading camera parameters for image {key}")
            cur_camera_params = camera_params.get(key, None)
            if cur_camera_params is None:
                cam_K = None    
                depth_scale = None  
            else:
                cam_K = np.array(camera_params[key]["cam_K"]).reshape(3, 3)
                depth_scale = camera_params[key].get("depth_scale", 1)
            self.Kmx = cam_K

        # <<< CAMERA PARAMETERS <<<


        # >>> RGB IMAGE >>>
        rgbs_path = scene_path / self._image_modality
        rgbs_names = sorted(os.listdir(rgbs_path))

        if len(rgbs_names) == 0:
            raise ValueError(f"No images found in {rgbs_path}")

        rgb_path = rgbs_path / rgbs_names[image_idx]
        self.rgb_path = rgb_path
        rgb = cv2.imread(str(rgb_path))
        self.rgb = rgb

        if not self._img_window_initialized:
            self._init_window2D(rgb)
        else:
            o3d_image = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            self.rgb_window_widget.update_image(o3d_image)  # Update the image in the window
            self.rgb_window.post_redraw()  # Forces to redraw the window
        # <<< RGB IMAGE <<<

        # >>> DEPTH IMAGE  + PCD >>>
        depth_path = scene_path / self._depth_dir_name / rgbs_names[image_idx]
        depth_exists = depth_path.exists()
        if depth_exists and self._show_pointcloud.checked:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
            depth = depth.astype(np.float32) * depth_scale / 1000
        else:
            depth = None
            pcd = None
            LOGGER.warning(
                f"Depth image not found for {depth_path} - No point cloud will be shown"
            )
            self._show_pointcloud.checked = False

        if depth is not None and self._show_pointcloud.checked:
            pcd = self._make_point_cloud(rgb, depth, cam_K)
            mtl = o3d.visualization.rendering.MaterialRecord()
            mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
            mtl.shader = "defaultUnlit"
            self._scene.scene.add_geometry(
                "point_cloud",
                pcd,
                mtl,
                # self.settings.scene_material,
                add_downsampled_copy_for_fast_rendering=True,
            )

            bounds = pcd.get_axis_aligned_bounding_box()

            if self._reset_camera_view:
                self._scene.setup_camera(60, bounds, bounds.get_center())
                center = np.array([0, 0, 0])
                eye = center + np.array([0, 0, -0.5])
                up = np.array([0, -1, 0])
                self._scene.look_at(center, eye, up)
                self._reset_camera_view = False
        
        # <<< DEPTH IMAGE  + PCD <<<

        # >>> VISUALIZATION OF THE GT >>>
        if self._show_ground_truth.checked:
            gt_path = scene_path / "scene_gt.json"
            self.gt_path = gt_path

            if not gt_path.exists():
                LOGGER.warning(
                    f"Ground truth file not found in {gt_path} - Skipping the GT visualization"
                )
                gt = []
            else:
                with open(gt_path) as f:
                    gt = json.load(f)
                    key = str(int(image_name))
                    gt = gt[key]

                self.gt_scene_annot = gt
                self._add_gt3D_to_scene(gt)
        else:
            gt = []
        # <<< VISUALIZATION OF THE GT <<<

        # >>> VISUALIZATION OF THE PREDICTIONS >>>
        bop_img_id = int(image_name)
        bop_scene_id = int(self.cur_scene_name)
        self._bop_scene_id = bop_scene_id
        self._bop_img_id = bop_img_id
        self._add_predictions_to_scene(bop_scene_id, bop_img_id)
        # <<< VISUALIZATION OF THE PREDICTIONS <<<

def run_app(config: dict, connection: socket.socket):
    """Runs the application with the given connection for 2D visualization

    Args:
        config (dict): Configuration dictionary
        connection (socket.socket, optional):   Connection to the 2D renderer. Defaults to None.
    """
    split_scene_path = config.get("split_scenes_path", None)
    models_path = config.get("models_path", None)
    csv_paths = config.get("csv_paths", None)
    save_image_path = config.get("save_image_path", "images")
    dataset_name = config.get("dataset_name", "BOP")

    scenes = Dataset(
        dataset_name=dataset_name,
        scenes_path=split_scene_path, models_path=models_path, csv_paths=csv_paths
    )

    gui.Application.instance.initialize()
    app = AppWindow(scenes, connection, save_image_path=save_image_path)  # Initializes the window

    app.scene_load(0, 0)
    # app.update_obj_list()

    gui.Application.instance.run()


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


    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        LOGGER.info(
            f"Server is running on {host}:{port} - Waiting for 2D renderer to connect"
        )
        conn, addr = s.accept()
        LOGGER.info(f"Connected by {addr}")
        run_app(config, conn)
