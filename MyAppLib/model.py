import json
import os
from typing import Tuple

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFileDialog, QGraphicsItem, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap, QPen

from MyAppLib.irect import InteractiveRectangle
from MyAppLib.iscene import ImageScene
from MyAppLib.onnx_inference import InferenceYOLOv5
from MyAppLib.settings import (cell_type, WIDTH_THRESHOLD, SCALE_FACTOR,
                               GRID_CELL_SIZE, COLOR_GRID)


class ModelLayer:
    """ Contains all the program data; opens and saves files. """
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._image_to_analyze_path = ''
        self._image_to_analyze_size = 0
        self._image_to_analyze = None
        self._image_scene      = None
        self._active_class     = None
        self._has_image        = False
        self._annotations_path = ''

    # ------------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------------
    def get_model_path(self) -> str:
        """ Returns the path to the ONNX file of the model. """
        return self._model_path

    def get_annotations_path(self) -> str:
        """ Returns the path to the annotations file. """
        return self._annotations_path

    def get_counter_values(self) -> Tuple[int, int]:
        """ Returns the number of dividing and non-dividing cells. """
        return self._image_scene.dividing, self._image_scene.not_dividing

    def get_image_to_analyze(self) -> np.ndarray:
        """ Returns the loaded image. """
        return self._image_to_analyze

    def has_image_to_analyze(self) -> bool:
        """ Checks if an ImageScene has been created. """
        return self._has_image

    def get_image_to_analyze_name(self) -> np.ndarray:
        """ Returns the file name of the loaded image. """
        return os.path.basename(self._image_to_analyze_path)

    def get_image_scene(self) -> ImageScene:
        """ Returns the created ImageScene. """
        return self._image_scene

    def open_image(self) -> None:
        """ Loads an image from the specified file. """
        self._image_to_analyze_path = QFileDialog.getOpenFileName(
                None, 'Open file', 'C:/', 'Image files (*.jpg *.png)')[0]

        if self._image_to_analyze_path == '':
            return

        self._image_to_analyze_size = os.path.getsize(
                self._image_to_analyze_path)

        self._image_to_analyze = cv2.imread(self._image_to_analyze_path,
                                            cv2.IMREAD_GRAYSCALE)

        self._image_scene = None
        self._has_image = False

    def clear_selection(self) -> None:
        """ Deselects the rectangles. """
        self._image_scene.clear_selection()

    def delete_rect_selected(self) -> None:
        """ Returns a reference to the rectangle selected in the scene. """
        rect_selected = self.get_rect_selected()

        if rect_selected.get_label() == cell_type.DIV:
            self._image_scene._update_div_cell_counter(-1)
        else:
            self._image_scene._update_not_div_cell_counter(-1)

        self._delete_rect_selected()
        self.clear_selection()

    def change_class_rect_selected(self) -> None:
        """ Changes the class of the selected rectangle. """
        rect_selected = self.get_rect_selected()

        rect_selected.change_class()

        if rect_selected.get_label() == cell_type.DIV:
            self._image_scene._update_div_cell_counter(1)
            self._image_scene._update_not_div_cell_counter(-1)
        else:
            self._image_scene._update_div_cell_counter(-1)
            self._image_scene._update_not_div_cell_counter(1)

    def has_rect_selected(self) -> bool:
        """ Checks if any rectangle is selected. """
        return False if self.get_rect_selected() is None else True

    def has_image_scene(self) -> bool:
        """ Checks if any ImageScene created already exists. """
        return False if self._image_scene is None else True

    def get_rect_selected(self) -> "InteractiveRectangle":
        """ Returns a reference to the rectangle selected in the scene. """
        return self._image_scene.get_rect_selected()

    def update_scene(self) -> None:
        """ Calls the 'update' method of the created ImageScene object. """
        self._image_scene.update()

    def parse_json_annotations(self, annotations: dict) -> list:
        """ Parses a JSON file with annotations in the VIA format.

        This function must conform to the interface of the
        '_add_predictions_to_scene' fucntion.

        Parameters
        ----------
        annotations : dict
            Loaded data from JSON file.

        Returns
        -------
        list
            Bounding boxes from the data. They are in the form:
                [x0, y0, x1, y1, conf, div_probability, not_div_probability]
        """
        boxes = []

        for _, v in annotations.items():
            for box in v['regions']:
                x0 = box['shape_attributes']['x']
                y0 = box['shape_attributes']['y']
                x1 = box['shape_attributes']['x'] +\
                    box['shape_attributes']['width']
                y1 = box['shape_attributes']['y'] +\
                    box['shape_attributes']['height']

                if box['region_attributes']['cell_type'] == 'dividing':
                    labels = [1, 0]
                else:
                    labels = [0, 1]

                # '1' is for the confidence value
                boxes.append([x0, y0, x1, y1, 1, *labels])

        return [boxes]

    def load_annotations(self) -> None:
        """ Loads annotations from a JSON file. """
        image = self._image_to_analyze
        height, width = image.shape

        annotations_path = QFileDialog.getOpenFileName(
            None, 'Open file', '.', 'JSON (*.json)')[0]
        self._annotations_path = annotations_path

        if self._annotations_path == '':
            return

        with open(annotations_path, 'r') as annotations_file:
            data = json.load(annotations_file)

        predictions = self.parse_json_annotations(data)

        self._set_image_scene()
        self._add_predictions_to_scene(predictions)

        # Clearing memory
        self._image_to_analyze = None

        if width > WIDTH_THRESHOLD:
            width //= SCALE_FACTOR
            height //= SCALE_FACTOR

        self._create_grid(width, height)

    def save_annotations(self) -> None:
        """ Saves annotations to a JSON file. """
        filename = os.path.basename(self._image_to_analyze_path)
        size = self._image_to_analyze_size

        save_path = QFileDialog.getSaveFileName(None, 'Save file',
                                                '.', 'JSON (*.json)')[0]

        data = {
            filename + str(size): {
                    'filename': filename,
                    'size': size,
                    'regions': [],
                    'file_attributes': {},
                }
        }

        rects = [rect for rect in self._image_scene.items()
                 if rect.zValue() == 0]

        for it in rects:

            x = round(it.sceneBoundingRect().left() * SCALE_FACTOR, 0)
            y = round(it.sceneBoundingRect().top() * SCALE_FACTOR, 0)
            w = round(it.sceneBoundingRect().width() * SCALE_FACTOR, 0)
            h = round(it.sceneBoundingRect().height() * SCALE_FACTOR, 0)
            label = 'dividing' if (it._annotation_label == cell_type.DIV)\
                    else 'not_dividing'

            box = {
                'shape_attributes': {
                    'name': 'rect',
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                },
                'region_attributes': {
                    'cell_type': label
                }
            }

            data[filename + str(size)]['regions'].append(box)

        with open(save_path, 'w', encoding='utf-8') as wf:
            json.dump(data, wf, indent=2)

    def make_predictions(self) -> None:
        """ Uses a neural network to detect cells in an image.

        This function must conform to the interface of the
        '_add_predictions_to_scene' fucntion.
        """
        image = self._image_to_analyze
        height, width = image.shape

        if image is not None:
            # Make predictions
            model = InferenceYOLOv5(self.get_model_path())
            predictions = model.inference_big_image(image)

            # Clearing memory 1
            del model

            self._set_image_scene()
            self._add_predictions_to_scene(predictions)

            # Clearing memory 2
            self._image_to_analyze = None

            if width > WIDTH_THRESHOLD:
                width //= SCALE_FACTOR
                height //= SCALE_FACTOR

            self._create_grid(width, height)

    def set_active_class(self, active_class: cell_type) -> None:
        """ Gets the current active class.

        Parameters
        ----------
        active_class : cell_type
            Class to set active.
        """
        self._active_class = active_class
        if self.has_image_scene():
            self._image_scene.set_active_class(active_class)

    # ------------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------------
    def _add_horizontal_line(self, y: int, image_width: int) -> None:
        """ Adds a horizontal line to to the current ImageScene.

        Parameters
        ----------
        y : int
            The position of a new horizontal line.

        image_width : int
            The width of the analyzed image.
        """
        pen = QPen(COLOR_GRID)
        pen.setCosmetic(True)

        new_line = self._image_scene.addLine(0, y, image_width - 1, y,
                                             pen=pen)
        new_line.setZValue(1)

    def _add_vertical_line(self, x: int, image_height: int) -> None:
        """ Adds a vertical line to to the current ImageScene.

        Parameters
        ----------
        x : int
            The position of a new vertical line.

        image_height : int
            The height of the analyzed image.
        """
        pen = QPen(COLOR_GRID)
        pen.setCosmetic(True)

        new_line = self._image_scene.addLine(x, 0, x, image_height - 1,
                                             pen=pen)
        new_line.setZValue(1)

    def _add_predictions_to_scene(self, predictions: list) -> None:
        """ Draws bounding boxes on an ImageScene.

        Parameters
        ----------
        predictions : list
            A 2D list of predictions: image-crop x boxes.
        """
        # Drawing the bounding boxes
        for crop in predictions:
            for box in crop:
                width  = box[2] - box[0]
                height = box[3] - box[1]
                label  = cell_type.DIV if box[5] > box[6]\
                    else cell_type.NOT_DIV

                # Add a new rectangle to the QtGraphicsScene
                self._image_scene.create_rectangle(
                    (box[0] / SCALE_FACTOR, box[1] / SCALE_FACTOR),
                    (width / SCALE_FACTOR,  height / SCALE_FACTOR),
                    label)

                # Update counters
                if label == cell_type.DIV:
                    self._image_scene._update_div_cell_counter(1)
                else:
                    self._image_scene._update_not_div_cell_counter(1)

    def _delete_rect_selected(self) -> None:
        """ Returns a reference to the rectangle selected in the scene. """
        rect_selected = self.get_rect_selected()
        self._image_scene.removeItem(rect_selected)

    def _set_image_scene(self) -> None:
        """ Creates an ImageScene object. """
        height, width = self._image_to_analyze.shape
        bytes_per_line = 1 * width
        qimage = QImage(self._image_to_analyze.data, width, height,
                        bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(qimage)

        if pixmap.width() > WIDTH_THRESHOLD:
            pixmap = pixmap.scaled(width  / SCALE_FACTOR,
                                   height / SCALE_FACTOR,
                                   Qt.KeepAspectRatio,
                                   Qt.FastTransformation)
            height //= SCALE_FACTOR
            width //= SCALE_FACTOR

        pixmap_graphics_item = QGraphicsPixmapItem(pixmap)
        pixmap_graphics_item.setFlag(QGraphicsItem.ItemIsFocusable, True)

        self._image_scene = ImageScene()
        self._image_scene.addItem(pixmap_graphics_item)
        pixmap_graphics_item.setZValue(-1)

        # Update counters
        self._image_scene._reset_cell_counters()

        # Update the 'has_image' variable
        self._has_image = True

    def _create_grid(self, width: int, height: int) -> None:
        """ Creates a grid over the analyzed image.

        Parameters
        ----------
        width : int
            The width of the analyzed image.
        height : int
            The height of the analyzed image.
        """
        w_num, h_num = self._get_number_of_grid_lines(width, height)

        # Horizontal lines
        for y in range(1, w_num + 1):
            self._add_horizontal_line(y * GRID_CELL_SIZE, width)

        # Vertical lines
        for x in range(1, h_num + 1):
            self._add_vertical_line(x * GRID_CELL_SIZE, height)

    def _get_number_of_grid_lines(self, width: int,
                                  height: int) -> 'tuple[int, int]':
        """ Calculates the number of grid lines to draw.

        Parameters
        ----------
        width : int
            The width of the analyzed image.
        height : int
            The height of the analyzed image.
        """
        return (width // GRID_CELL_SIZE, height // GRID_CELL_SIZE)
