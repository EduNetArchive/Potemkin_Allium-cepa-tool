""" This module contains the ImageScene class implementation. """
from typing import Optional, Tuple

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtWidgets import QGraphicsScene, QGraphicsRectItem
from PyQt5.QtGui import QMouseEvent, QPen

from MyAppLib.irect import InteractiveRectangle
from MyAppLib.settings import (cell_type, COLOR_FILL, COLOR_STROKE,
                               STROKE_WIDTH)


MIN_WIDTH  = 50
MIN_HEIGHT = 50


class ImageScene(QGraphicsScene):
    """ The main widget of this app.

    Attributes
    ----------
    rect_selected : InteractiveRectangle
        The currently selected bounding box
    begin : PyQt5.QtCore.QPoint
    end : PyQt5.QtCore.QPoint
    drawing : bool
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._begin_point = QPoint()
        self._end_point   = QPoint()

        self._active_class  = None
        self._rect_selected = None
        self._item_to_draw  = None
        self._diff = 0.0

        # Counters
        self.dividing     = 0
        self.not_dividing = 0

        # Modes
        self._selecting = False
        self._drawing   = False
        self._moving    = False
        self._scaling   = False

    # ------------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------------
    def create_rectangle(self, top_left: tuple, size: tuple,
                         label: cell_type) -> InteractiveRectangle:
        """ Creates a new rectangle and returns a reference to it.

        Parameters
        ----------
        top_left : tuple
            The coordinates of the top left vertex of the rectangle.
        size : tuple
            The width and height of the rectangle.
        label : cell_type
            The class label of the rectangle.

        Returns
        -------
        new_rect : InteractiveRectangle
            The reference to the created rectangle.
        """
        new_rect = InteractiveRectangle(*top_left, *size)
        new_rect.set_label(label)

        new_rect.setBrush(COLOR_FILL[label.value])
        new_rect.setPen(QPen(Qt.NoPen))

        self.addItem(new_rect)

        if label == cell_type.DIV:
            self._update_div_cell_counter(1)
        else:
            self._update_not_div_cell_counter(1)

        return new_rect

    def clear_selection(self) -> None:
        """ Deselects the rectangles. """
        self.clearSelection()
        self._rect_selected = None

    def get_rect_selected(self) -> InteractiveRectangle:
        """ Returns a reference to the rectangle selected in the scene. """
        return self._rect_selected

    def delete_rect_selected(self) -> None:
        """ Removes the selected rectangle from the scene. """
        if self._rect_selected is not None:
            self.removeItem(self._rect_selected)

        self.clear_selection()

    def set_rect_selected(self, rect: InteractiveRectangle) -> None:
        """ Returns the selected rectangle. """
        rect.setSelected(True)
        self._rect_selected = rect

    def set_active_class(self, active_class: cell_type) -> None:
        """ """
        self._active_class = active_class

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """ Disables double click on ImageScene (Method overriding). """
        pass

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """ Handles a mouse buttons press - Selecting the mode.

        (Method overriding)
        """
        if event.button() == Qt.LeftButton:
            self._begin_point = event.scenePos()
            self._end_point   = event.scenePos()
            # The selection mode is always possible. The final check if it is
            # the selection mode or another mode is done in the mouseMoveEvent
            # method.
            self._selecting = True

            # If a click was over the selected rectangle - disable drawing.
            if self._rect_selected in self.items(self._begin_point)[:-1]:
                self._drawing = False

                # If the click was over the resizing handle - resizing is
                # possible, otherwise, moving is possible.
                if not self._rect_selected.has_handle_under_cursor(
                        self._begin_point):
                    self._moving = True
                    self._scaling = False
                else:
                    self._scaling = True
                    self._moving = False
                    self._rect_selected.mousePressEvent(event)

                rect_scene_position = self._rect_selected.scenePos()
                self._diff = rect_scene_position - self._begin_point
            else:
                # Otherwise. the drawing mode is possible.
                self._scaling = False
                self._moving = False
                self._drawing = True

            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """ Handles mouse movement - Do drawing, moving, or scaling.

        (Method overriding)
        """
        # If this event fires - disable the selecting mode.
        self._selecting = False

        self._end_point = event.scenePos()

        if self._drawing:
            if self._item_to_draw is None:
                self._create_item_to_draw()

            width, height = self._get_area_width_and_height()

            self._item_to_draw.setRect(self._begin_point.x(),
                                       self._begin_point.y(),
                                       width, height)
        else:
            if self._rect_selected is not None:
                if self._moving:
                    self._rect_selected.setPos(self._end_point + self._diff)
                else:
                    # Do scaling.
                    self._rect_selected.mouseMoveEvent(event)

        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """ Handles a mouse buttons release.

        (Method overriding)
        """
        if event.button() == Qt.LeftButton:

            self._end_point = event.scenePos()

            if self._selecting:
                self._update_selection()
            elif self._moving:
                self._rect_selected.update_item()
            elif self._scaling:
                self._rect_selected.mouseReleaseEvent(event)
            elif self._drawing:
                self._fixate_drawing_result()
                self._clear_item_to_draw()

            # Reset modes
            self._moving    = False
            self._drawing   = False
            self._selecting = False
            self._scaling   = False

            self.update()
            self.parent()._update_count_labels()
            super().mouseReleaseEvent(event)

    # ----------------------------------------------------------------------------
    # PRIVATE
    # ----------------------------------------------------------------------------
    def _create_item_to_draw(self) -> None:
        """ Creates a new _item_to_draw - dynamic rectangle while drawing. """
        self._item_to_draw = QGraphicsRectItem()
        self.addItem(self._item_to_draw)

        label = self._active_class.value
        self._item_to_draw.setPen(QPen(COLOR_STROKE[label], STROKE_WIDTH))
        self._item_to_draw.setPos((QPoint(0, 0)))

    def _clear_item_to_draw(self) -> None:
        """ Deletes _item_to_draw - dynamic rectangle while drawing. """
        self.removeItem(self._item_to_draw)
        self._item_to_draw = None

    def _update_selection(self) -> None:
        """ Updates selection. """
        item = None
        # Get items under click (Clip the last element which is a  QPixmap).
        items = self.items(self._begin_point)[:-1]

        if self._rect_selected in items:
            if len(items) > 1:
                idx = items.index(self._rect_selected)
                if idx + 1 == len(items):
                    item = items[0]
                else:
                    item = items[idx+1]
            elif len(items) == 1:
                item = items[0]
        else:
            if len(items) >= 1:
                item = items[0]

        # Clear selection before update.
        self.clear_selection()

        # Update selection
        if item is not None:
            self.set_rect_selected(item)

    def _fixate_drawing_result(self) -> None:
        """ Creates a new rectangle when drawing is complete.

        The rectangle will be selected immediately after creation.
        """
        # Selecting doesn't work correctly with negative width/height
        width, height = self._get_area_width_and_height()
        width = abs(width)
        height = abs(height)

        # If the width and height of the area between the start and end mouse
        # positions are large enough, create a new rectangle. Otherwise, do
        # nothing.
        if width > MIN_WIDTH and height > MIN_HEIGHT:
            begin_vertex = self._get_correct_top_left_vertex()
            # Get a label selected by the user from the UI.
            label = self._active_class
            new_rect = self.create_rectangle(begin_vertex,
                                             (width, height),
                                             label)
            self.clear_selection()
            self.set_rect_selected(new_rect)

    def _get_area_width_and_height(self) -> Tuple[float, float]:
        """ Returns the width and height.

        The width and height are of the area between the start and end mouse
        positions.
        """
        width  = self._end_point.x() - self._begin_point.x()
        height = self._end_point.y() - self._begin_point.y()

        return width, height

    def _get_correct_top_left_vertex(self) -> Tuple[float, float]:
        """ Returns the coordinates of the top left vertex of a rectangle. """
        begin_vertex = (min(self._begin_point.x(), self._end_point.x()),
                        min(self._begin_point.y(), self._end_point.y()))

        return begin_vertex

    def _update_div_cell_counter(self, num: int) -> None:
        """ Adds a specified number to the dividing cell counter. """
        self.dividing += num

    def _update_not_div_cell_counter(self, num: int) -> None:
        """ Adds a specified number to the non-dividing cell counter. """
        self.not_dividing += num

    def _reset_cell_counters(self) -> None:
        """ Resets cell counters. """
        self.dividing = 0
        self.not_dividing = 0
