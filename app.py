""" To run the program, run this module! """
import sys

from PyQt5.QtWidgets import QApplication

from MyAppLib.model import ModelLayer
from MyAppLib.view import ViewLayer

if __name__ == '__main__':
    model = ModelLayer('models/yolov5_allium.onnx')

    app = QApplication(sys.argv)
    window = ViewLayer(model)
    window.show()
    app.exec_()
