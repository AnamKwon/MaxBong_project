# coding: utf-8

import sys
import os
import glob
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
import numpy as np
import cv2

class Form(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = uic.loadUi("main_V2.ui", self)
        self.ui.show()
        self.ui.actionOpen_File.triggered.connect(lambda :self.Open(1))
        self.ui.actionOpen_Directory.triggered.connect(lambda :self.Open(2))
        self.ui.actionCOCO.triggered.connect(lambda: self.Open('COCO'))
        self.ui.actionVOC.triggered.connect(lambda: self.Open('VOC'))
        self.ui.actionYOLO.triggered.connect(lambda: self.Open('YOLO'))

    def Open(self,val):
        self.file_list = []
        if val == 1 :
            files_path= QtWidgets.QFileDialog.getOpenFileNames(filter="Image Files (*.png *.jpg *.bmp)")[0]
            self.file_list += files_path
        elif val in [2 , 'VOC' , 'YOLO']:
            options = {2:['jpg','png','bmp','jpeg'],'VOC':['xml'],'YOLO':['txt']}[val]
            files_path = QtWidgets.QFileDialog.getExistingDirectory()
            for i in options :
                self.file_list += glob.glob(files_path+f'/*.{i}')
        elif val == 'COCO' :
            files_path = QtWidgets.QFileDialog.getOpenFileNames(filter="MS COCO (*.json)")[0]
            self.file_list += files_path
        print(self.file_list)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())

#QtWidgets.QListWidget.selectedItems()
QtCore
