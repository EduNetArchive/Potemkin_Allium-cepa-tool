import time

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (QApplication, QWidget, QMainWindow, QFrame,
                             QLabel, QHBoxLayout, QVBoxLayout, QPushButton,
                             QButtonGroup, QGraphicsView)
from PyQt5.QtGui import QPixmap, QColor, QIcon, QKeyEvent

from MyAppLib.model import ModelLayer
from MyAppLib.settings import (COLOR_FILL, WINDOW_HEIGHT, WINDOW_WIDTH,
                               BUTTON_NAMES, cell_type)


class ViewLayer(QMainWindow):
    """ Describes the UI of the application. """
    def __init__(self, model_layer: ModelLayer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Connect a model layer
        self._model = model_layer

        self.setWindowTitle('Allium Cepa Tool')
        self.setFixedSize(QSize(WINDOW_WIDTH, WINDOW_HEIGHT))

        # The left side of the UI --------------------------------------------
        self._view = QGraphicsView()
        self._customize_view()

        # The right side of the UI -------------------------------------------
        self._menu = QFrame()
        self._customize_menu()

        _menu_layout = QVBoxLayout()
        _menu_layout.setAlignment(Qt.AlignTop)

        self._image_name_label   = QLabel()
        self._dividing_label     = QLabel()
        self._not_dividing_label = QLabel()
        self._class_button_group, _buttons = self._create_class_buttons()
        self._class_button_group.buttonClicked.connect(
                self._set_active_class)

        self._set_active_class()

        self._open_image_button = QPushButton('Open Image')
        self._open_image_button.clicked.connect(self._open_image)

        self._predict_button = QPushButton('Make predictions')
        self._predict_button.clicked.connect(self._make_predictions)

        self._load_anns_button = QPushButton('Load annotations')
        self._load_anns_button.clicked.connect(self._load_annotations)

        self._save_anns_button = QPushButton('Save annotations')
        self._save_anns_button.clicked.connect(self._save_annotations)

        _hint_label = QLabel()
        _hint_label.setText(f'D - delete selected cell\nZ - change class of'
                            'selected cell\n"+" - Zoom in\n"-" - Zoom out')

        # Add widgets
        _menu_layout.addWidget(self._open_image_button)
        _menu_layout.addWidget(self._image_name_label)
        _menu_layout.addWidget(self._predict_button)
        _menu_layout.addWidget(self._load_anns_button)
        _menu_layout.addWidget(QLabel())
        for b in _buttons:
            _menu_layout.addWidget(b)
        _menu_layout.addWidget(self._dividing_label)
        _menu_layout.addWidget(self._not_dividing_label)
        _menu_layout.addWidget(self._save_anns_button)
        _menu_layout.addWidget(_hint_label)

        self._menu.setLayout(_menu_layout)

        # A central widget ---------------------------------------------------
        _central_widget = QWidget()

        _central_layout = QHBoxLayout()
        _central_layout.addWidget(self._view, 3)
        _central_layout.addWidget(self._menu, 1)
        _central_widget.setLayout(_central_layout)

        self.setCentralWidget(_central_widget)

    # ------------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------------
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """ Handles key presses.

        (Method overriding)
        """
        if not self._model.has_image_scene():
            return

        # Zoom out
        if event.key() == Qt.Key_Minus:
            self._view.scale(0.8, 0.8)
            self._view.update()

        if event.key() == Qt.Key_Equal:
            self._view.scale(1.25, 1.25)
            self._view.update()

        if self._model.has_rect_selected():

            # Clear selection
            if event.key() == Qt.Key_Escape:
                self._model.clear_selection()

            # Delete a rectangle
            if event.key() == Qt.Key_D:
                self._model.delete_rect_selected()
                self._update_count_labels()

            # Change a class
            if event.key() == Qt.Key_Z:
                self._model.change_class_rect_selected()
                self._update_count_labels()

        self._model.update_scene()

    # ------------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------------
    # The left side of the UI ------------------------------------------------
    def _customize_view(self) -> None:
        """ Sets required flags for QGraphicsView. """
        self._view.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self._view.setLineWidth(3)
        self._view.setStyleSheet('background-color: darkgray;')
        self._view.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    # The right side of the UI -----------------------------------------------
    def _open_image(self) -> None:
        """ Opens an image and loads it into memory. """
        self._model.open_image()

        if not self._model.has_image_scene():
            self._view.setScene(None)

        self._update_image_label(self._model.get_image_to_analyze_name())

        # Make beep sound
        QApplication.beep()

    def _load_annotations(self) -> None:
        """ Loads annotations from a JSON file. """
        if self._model.get_image_to_analyze() is None:
            return

        self._model.load_annotations()

        if self._model.get_annotations_path() == '':
            return

        # Display the image and annotations
        self._view.setScene(self._model.get_image_scene())
        self._view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._view.setBackgroundBrush(Qt.black)  # Change bg color
        self._model.get_image_scene().setParent(self)

        self._model.set_active_class(self._get_active_class())
        self._update_count_labels()

        self._open_image_button.setDisabled(True)
        self._predict_button.setDisabled(True)
        self._load_anns_button.setDisabled(True)

        QApplication.beep()

    def _save_annotations(self) -> None:
        """ Saves annotations to a JSON file in the VIA format. """
        if not self._model.has_image_to_analyze():
            return

        self._model.save_annotations()

        QApplication.beep()

    def _make_predictions(self) -> None:
        """ Gets predictions from the neural network. """
        if self._model.get_image_to_analyze() is None:
            return

        start_time = time.time()

        self._model.make_predictions()

        # Display the image and annotations
        self._view.setScene(self._model.get_image_scene())
        self._view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._view.setBackgroundBrush(Qt.black)  # Change bg color
        self._model.get_image_scene().setParent(self)

        self._model.set_active_class(self._get_active_class())
        self._update_count_labels()

        self._open_image_button.setDisabled(True)
        self._predict_button.setDisabled(True)
        self._load_anns_button.setDisabled(True)

        current_text = self._image_name_label.text()
        time_text = f'({(time.time() - start_time) / 60:.2f} minutes)'
        self._update_image_label(current_text + '\n' + time_text)

        # Make beep sound
        QApplication.beep()

    def _update_image_label(self, name: str) -> None:
        """ Sets the image name as text for the label.

        Parameters
        ----------
        name : str
            The filename corresponding to the image.
        """
        self._image_name_label.setText(name)

    def _create_class_buttons(self) -> QButtonGroup:
        """ Creates buttons for class selection.

        Returns
        -------
        QButtonGroup
            Buttons for class selection.
        """
        class_button_group = QButtonGroup()
        buttons = []

        for ct in range(len(cell_type)):
            button = QPushButton(BUTTON_NAMES[ct])
            button.setIcon(self._create_rectangle_icon(COLOR_FILL[ct]))

            class_button_group.addButton(button, id=ct)
            button.setCheckable(True)

            buttons.append(button)

        # Check the last button
        button.setChecked(True)

        return class_button_group, buttons

    def _create_rectangle_icon(self, color: QColor) -> QIcon:
        """ Creates colored rectangle icons for buttons.

        Parameters
        ----------
        color: QColor
            The color of a new icon.

        Returns
        -------
        QIcon
            A new icon.
        """
        pixmap = QPixmap(32, 32)
        icon_color = QColor(color.red(), color.green(), color.blue(), 255)
        pixmap.fill(icon_color)

        return QIcon(pixmap)

    def _customize_menu(self) -> None:
        """ Sets required flags for the menu. """
        self._menu.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self._menu.setLineWidth(3)

    def _get_active_class(self) -> cell_type:
        """ Returns the class selected by the user in the UI.

        Returns
        -------
        cell_type
            The selected cell type.
        """
        return cell_type(self._class_button_group.checkedId())

    def _set_active_class(self) -> None:
        """ Sets the selected class. """
        active_class = cell_type(self._class_button_group.checkedId())
        self._model.set_active_class(active_class)

    def _update_count_labels(self) -> None:
        """ Updates the counting labels. """
        div, not_div = self._model.get_counter_values()

        self._dividing_label.setText(f'DIVIDING: {div}')
        self._not_dividing_label.setText(f'NOT DIVIDING: {not_div}')
