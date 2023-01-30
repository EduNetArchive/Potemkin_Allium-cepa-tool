import enum

from PyQt5.QtGui import QColor


class cell_type(enum.Enum):
    """ Describes types of cells. """
    DIV     = 0
    NOT_DIV = 1


ALPHA_FILL   = 100
COLOR_FILL   = [QColor(  0, 200, 0, ALPHA_FILL),
                QColor(200, 150, 0, ALPHA_FILL)]
COLOR_STROKE = [QColor(  0, 120, 0, 255),
                QColor(120,  70, 0, 255)]
BUTTON_NAMES = ['DIVIDING', 'NOT DIVIDING']
SELETED_STROKE_COLOR = QColor(200,  50,  50, 255)
HANDLE_STROKE_COLOR  = QColor( 10,  10,  10, 255)
HANDLE_FILL_COLOR    = QColor(255, 255, 255, 255)

# Interactive Rectangle
HANDLE_SIZE  = 10  # The handle diameter
STROKE_WIDTH = 2

# Grid
COLOR_GRID = QColor(255, 255, 255, 255)
GRID_WIDTH = 2
GRID_CELL_SIZE = 1000

# MainWindow
SCALE_FACTOR    = 3
WIDTH_THRESHOLD = 3000
WINDOW_WIDTH    = 1280
WINDOW_HEIGHT   = 720
