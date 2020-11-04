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
from basicFunctions import convType

class MainApp(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = uic.loadUi("test.ui", self)

        #--------------중복이미지------------#
        self.ui.img_check_list.setViewMode(QtWidgets.QListView.IconMode)
        self.ui.img_check_list.setIconSize(QtCore.QSize(200,200))
        self.ui.img_check_list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ui.img_delete.clicked.connect(self.Del_img)
        self.ui.img_check_delete.clicked.connect(lambda  : self.Del_img(mode=1))


        #----------메뉴 시작할때 숨기기 --------#
        self.ui.file_aug_pop.setHidden(not self.ui.file_aug_pop.isHidden())
        self.ui.file_fix_pop.setHidden(not self.ui.file_fix_pop.isHidden())
        self.ui.file_trans_pop.setHidden(not self.ui.file_trans_pop.isHidden())
        #self.ui.img_check.setHidden(not self.ui.img_check.isHidden())

        #-----------팝업 메뉴 on_off 메뉴-------#
        self.ui.file_aug.clicked.connect(lambda : self.Popup(2))
        self.ui.file_fix.clicked.connect(lambda : self.Popup(1))
        self.ui.file_trans.clicked.connect(lambda : self.Popup(0))

        #------------버튼 연결-------------------#
        self.ui.file_open.clicked.connect(self.File_open)

        #------------view 연결 -----------------#
        self.ui.file_list.clicked.connect(self.Img_view)

        #------------fix ----------------------#
        self.ui.file_fix_type.addItems('BMP, DIB, EPS, GIF, ICNS, ICO, IM, JPEG, MSP, PCX, PNG, PPM, SGI, SPIDER, TGA, TIFF, Webp'.split(', '))
        self.ui.file_fix_button.clicked.connect(self.test_button)


    def File_open(self):
        self.ui.img_view.clear()
        self.ui.file_list.clear()
        self.file_list_ = []
        self.ui.file_type.setText("COCO")
        self.ui.file_to_type1.setText("To Voc")
        self.ui.file_to_type2.setText("To Yolo")
        self.path_directory = QtWidgets.QFileDialog.getExistingDirectory(None, 'Select Directory')
        for i in ['jpg','png','bmp','jpeg'] :
            self.file_list_ += glob.glob(self.path_directory + f'/*.{i}')
        self.ui.file_list.addItems(self.file_list_)
        self.ui.file_count.setText(f'총 {len(self.file_list_)} 개')

    def Img_view(self):
        self.ui.img_check_list.clear()
        img = QtGui.QPixmap()
        img.load(self.ui.file_list.currentItem().text())
        print(self.ui.img_view.width()/img.width())
        print(self.ui.img_view.height()/img.height())
        if self.ui.img_view.width()/img.width() > self.ui.img_view.height()/img.height() :
            x = self.ui.img_view.height()/img.height()
        else :
            x = self.ui.img_view.width()/img.width()
        img = img.scaled(int(img.width()*x), int(img.height())*x)
        self.ui.img_view.setPixmap(img)

        self.test = np.random.randint(0,len(self.file_list)-1, 5)
        for i in self.test:
            i = self.file_list_[i]
            item = QtWidgets.QListWidgetItem(QtGui.QIcon(i),i)
            self.ui.img_check_list.addItem(item)

    def Del_img(self,mode=0):
        print(1)
        if mode == 1 :
            print(2)
            print(self.ui.img_check_list.selectedItems())
        else :
            print(3)
            self.ui.img_check_list.selectAll()
            print(self.ui.img_check_list.selectedItems())

    def Popup(self, case_):
        if case_ == 0 :
            self.ui.file_trans_pop.setHidden(not self.ui.file_trans_pop.isHidden())
        elif case_ == 1 :
            self.ui.file_fix_pop.setHidden(not self.ui.file_fix_pop.isHidden())
        else :
            self.ui.file_aug_pop.setHidden(not self.ui.file_aug_pop.isHidden())

    def test_button(self):
        name = self.ui.file_fix_name.toPlainText().strip()
        x = self.ui.file_fix_x.toPlainText().strip()
        y = self.ui.file_fix_y.toPlainText().strip()
        convType.convType(self.file_list_, self.ui.file_fix_type.currentText(), name, x, y)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = MainApp()
    w.show()
    sys.exit(app.exec())


QtWidgets.QLabel.height()
QtGui.QPixmap.width()