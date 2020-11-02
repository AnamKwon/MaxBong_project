# coding: utf-8

import sys
import os
import glob
from Data_type import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot


class Form(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = uic.loadUi("main_V2.ui", self)
        self.ui.Image_V.setMouseTracking(False)
        self.ui.show()
        self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())
        self.ui.actionOpen_File.triggered.connect(lambda :self.Open(1))
        self.ui.actionOpen_Directory.triggered.connect(lambda :self.Open(2))
        self.ui.actionCOCO.triggered.connect(lambda: self.Open('COCO'))
        self.ui.actionVOC.triggered.connect(lambda: self.Open('VOC'))
        self.ui.actionYOLO.triggered.connect(lambda: self.Open('YOLO'))

        self.ui.File_L.clicked.connect(self.Img_view)
        self.startpos = None
        self.endpos = None

    def mouseReleaseEvent(self, e):  # e ; QMouseEvent
        print('BUTTON RELEASE')
        self.endpos = (e.x(),e.y())
        print(self.startpos)
        print(self.endpos)
        print(self.ui.Image_V.size())
        print(self.ui.centralwidget.childAt(self.ui.Image_V.pos()).pos())
        print(self.ui.scrollArea.pos())


    def mousePressEvent(self, e):  # e ; QMouseEvent
        print('BUTTON PRESS')
        self.startpos = (e.x(), e.y())

    def mouseMoveEvent(self, e):

        x = e.x() - 250
        y = e.y() - 26

        text = 'x: {0}, y: {1}'.format(x, y)
        print(self.Image_V.geometry())
        print(text)

    def Img_view(self):
        img = QtGui.QPixmap()
        img.load(self.ui.File_L.currentItem().text())
        print(self.ui.Image_V.width()/img.width())
        print(self.ui.Image_V.height()/img.height())
        if self.ui.Image_V.width()/img.width() > self.ui.Image_V.height()/img.height() :
            x = self.ui.Image_V.height()/img.height()
        else :
            x = self.ui.Image_V.width()/img.width()
        img = img.scaled(int(img.width()), int(img.height()))
        self.ui.Image_V.setFixedSize(img.size())
        self.ui.Image_V.setPixmap(img)


    @pyqtSlot()
    def Open(self,val):
        self.file_list = []
        if val == 1 :
            files_path= QtWidgets.QFileDialog.getOpenFileNames(filter="Image Files (*.png *.jpg *.bmp)")[0]
            self.file_list += files_path
        elif val in [2 , 'VOC' , 'YOLO']:
            options = {2:['jpg','png','bmp','jpeg'],'VOC':['xml'],'YOLO':['txt']}[val]
            files_path = QtWidgets.QFileDialog.getExistingDirectory()
            os.chdir(files_path)
            if val == 2 :
                for i in options :
                    self.file_list += glob.glob(f'*.{i}')
            elif val == 'VOC' :
                xml_files = glob.glob('*.xml')
                self.dataset = VOC(xml_files)
                self.file_list = get_image(self.dataset)

            elif val == 'YOLO' :
                label_file = QtWidgets.QFileDialog.getOpenFileName(filter="Labels (*.txt)",directory=files_path)[0]
                txt_files = glob.glob(f'*.txt')
                self.dataset = YOLO(txt_files, label_file)
                self.file_list = get_image(self.dataset)


        elif val == 'COCO' :
            files_path = QtWidgets.QFileDialog.getOpenFileName(filter="MS COCO (*.json)")[0]
            os.chdir(files_path.rsplit('/',1)[0])
            self.dataset = COCO(files_path)
            self.file_list = get_image(self.dataset)
        self.ui.File_L.addItems(self.file_list)
        self.ui.actionData_Processing.setEnabled(True)
        self.ui.actionData_Trans_Form.setEnabled(True)
        self.ui.actionImage_duplicate_check_2.setEnabled(True)
        self.ui.actionView_Duplicate_list_2.setEnabled(True)
        self.ui.actionView_Duplicate_image.setEnabled(True)




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())

#QtWidgets.QListWidget.selectedItems()
QtWidgets.QLabel.childAt().pos()
QtWidgets.QMainWindow.childAt()
