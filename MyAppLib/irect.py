""" This module contains the InteractiveRectangle class implementation. """
from typing import Optional

from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtWidgets import (QGraphicsRectItem, QStyleOptionGraphicsItem,
                             QWidget)
from PyQt5.QtGui import QBrush, QMouseEvent, QPainter, QPainterPath, QPen

from MyAppLib.settings import (HANDLE_SIZE, STROKE_WIDTH, cell_type,
                               COLOR_FILL, COLOR_STROKE, SELETED_STROKE_COLOR,
                               HANDLE_FILL_COLOR, HANDLE_STROKE_COLOR)


class InteractiveRectangle(QGraphicsRectItem):
    """ Describes an interactive rectangle.

    Attributes
    ----------
    _handles : dict
        All resizing handles of this instance in the format: 'name': QRectF.
        This dictionary is created and updated by calling the
        _update_handles_pos method.
    _handle_selected : str
        The resizing handle currently selected.
    _annotation_label : cell_type
        The class of this instance: cell_type.DIV or cell_type.NOT_DIV.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_flags()

        self._handles = {}
        self._update_handles_pos()

        self._handle_selected = None
        self._annotation_label = None    # cell_type.DIV or NOT_DIV

    # ------------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:
        """ Handles a mouse buttons press - handle selection.

        (Method overriding)
        """
        if self.isSelected():
            self._handle_selected = self._handle_at(event.scenePos())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """ Handles mouse movement - resizing.

        (Method overriding)

        Movement of rectangles is handled by the ImageScene class from the
        'iscene' module.
        """
        if self._handle_selected is not None:
            self._interactive_resize(event.scenePos())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """ Handles a mouse buttons release - fixing the new size.

        (Method overriding)
        """
        self._handle_selected = None

        rect = self.rect()
        self._rect_correction(rect)

        self.setRect(rect)
        self._update_handles_pos()
        self.update()

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem,
              widget: QWidget) -> None:
        """ Draws the rectangle and the handlers.

        (Method overriding)
        """
        label = self._annotation_label.value
        rect_stroke_color = SELETED_STROKE_COLOR if self.isSelected() \
            else COLOR_STROKE[label]

        # Draw rectangle
        painter.setBrush(COLOR_FILL[label])
        painter.setPen(QPen(rect_stroke_color, STROKE_WIDTH))
        painter.drawRect(self.rect())

        # Draw circle handles
        painter.setBrush(QBrush(HANDLE_FILL_COLOR))
        painter.setPen(QPen(HANDLE_STROKE_COLOR, STROKE_WIDTH))
        if self.isSelected():
            for _, rect in self._handles.items():
                painter.drawEllipse(rect)

    def shape(self) -> QPainterPath:
        """ Returns the shape of this item as a QPainterPath.

        (Method overriding)

        PyQT uses this function for collision detection. The shape is now
        based on the bounding box of this instance.

        Returns
        -------
        QPainterPath
            The shape of this instance.
        """
        path = QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def boundingRect(self) -> QRectF:
        """ Returns the bounding box of this instance.

        (Method overriding)

        For convenience, the bounding box is expanded slightly if the instance
        is selected. This allows to easily select resizing handles.

        Returns
        -------
        QRectF
            The bounding box.
        """
        if not self.isSelected():
            return self.rect()
        else:
            return self.rect().adjusted(-HANDLE_SIZE, -HANDLE_SIZE,
                                        HANDLE_SIZE, HANDLE_SIZE)

    def change_class(self) -> None:
        """ Reverses the class label of this instance.

        cell_type.DIV <--> cell_type.NOT_DIV.
        """
        if self._annotation_label == cell_type.DIV:
            self._annotation_label = cell_type.NOT_DIV
        else:
            self._annotation_label = cell_type.DIV

    def get_label(self) -> cell_type:
        """ Returns the class label of this instance.

        Returns
        -------
        cell_type
            The class label.
        """
        return self._annotation_label

    def set_label(self, label: cell_type) -> None:
        """ """
        self._annotation_label = label

    def update_item(self) -> None:
        """ """
        self._update_handles_pos()

    def has_handle_under_cursor(self, point: QPointF) -> bool:
        """ """
        return False if self._handle_at(point) is None else True

    # ------------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------------
    def _interactive_resize(self, point: QPointF) -> None:
        """ Updates the rectangle shape according to the mouse position.

        Fixates the results of the resizing process.

        Parameters
        ----------
        point : QPointF
            The position of the mouse cursor.
        """
        rect = self.rect()
        self.prepareGeometryChange()

        # Quit if no handle is selected.
        if self._handle_selected is None:
            return

        updated_x = point.x() - self.scenePos().x()
        updated_y = point.y() - self.scenePos().y()

        # --------------------------------------------------------------------
        # Top
        # --------------------------------------------------------------------
        if self._handle_selected == 'top left':
            rect.setLeft(updated_x)
            rect.setTop(updated_y)

        elif self._handle_selected == 'top':
            rect.setTop(updated_y)

        elif self._handle_selected == 'top right':
            rect.setRight(updated_x)
            rect.setTop(updated_y)

        # --------------------------------------------------------------------
        # Middle
        # --------------------------------------------------------------------
        elif self._handle_selected == 'middle left':
            rect.setLeft(updated_x)

        elif self._handle_selected == 'middle right':
            rect.setRight(updated_x)

        # --------------------------------------------------------------------
        # Bottom
        # --------------------------------------------------------------------
        elif self._handle_selected == 'bottom left':
            rect.setLeft(updated_x)
            rect.setBottom(updated_y)

        elif self._handle_selected == 'bottom':
            rect.setBottom(updated_y)

        elif self._handle_selected == 'bottom right':
            rect.setRight(updated_x)
            rect.setBottom(updated_y)

        # Update the rect and the handles.
        self.setRect(rect)
        self._update_handles_pos()

    def _handle_at(self, point: QPointF) -> Optional[str]:
        """ Returns a resizing handle under the mouse cursor, else None.

        Parameters
        ----------
        point : QPointF
            The position of the mouse cursor.

        Returns
        -------
        str or None
            The name of a resizing handle.
        """
        correct_point = point - self.scenePos()

        for handle, rect in self._handles.items():
            if rect.contains(correct_point):
                return handle

        return None

    def _rect_correction(self, rect: QRectF) -> None:
        """ Fixes negative width and/or negative height of rectangles.

        Parameters
        ----------
        rect : QRectF
            Rectangle to correct.
        """
        if rect.width() < 0:
            original_left  = rect.left()
            original_right = rect.right()
            rect.setLeft(original_right)
            rect.setRight(original_left)

        if rect.height() < 0:
            original_top    = rect.top()
            original_bottom = rect.bottom()
            rect.setTop(original_bottom)
            rect.setBottom(original_top)

    def _set_flags(self) -> None:
        """ Sets required flags for this instance. """
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsFocusable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)

    def _update_handles_pos(self) -> None:
        """ Updates the positions of the resizing handles.

        The update is performed according to the current state of the
        bounding box.
        """
        s = HANDLE_SIZE
        b = self.rect()

        left_x   = b.left() - s / 2
        center_x = b.center().x() - s / 2
        right_x  = b.right() - s / 2

        top_y    = b.top() - s / 2
        middle_y = b.center().y() - s / 2
        bottom_y = b.bottom() - s / 2

        self._handles['top left']  = QRectF(left_x, top_y, s, s)
        self._handles['top']       = QRectF(center_x, top_y, s, s)
        self._handles['top right'] = QRectF(right_x, top_y, s, s)

        self._handles['middle left']  = QRectF(left_x, middle_y, s, s)
        self._handles['middle right'] = QRectF(right_x, middle_y, s, s)

        self._handles['bottom left']  = QRectF(left_x, bottom_y, s, s)
        self._handles['bottom']       = QRectF(center_x, bottom_y, s, s)
        self._handles['bottom right'] = QRectF(right_x, bottom_y, s, s)
