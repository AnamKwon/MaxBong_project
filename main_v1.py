# coding: utf-8
import sys
from libs.Data_type import *
from libs.convType import *
from libs.imagecheck import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot


class test(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi("C:/Users/kan91/Desktop/QT/ui.ui", self)

    @pyqtSlot()
    def slot_1st(self):
        self.ui.label.setText("첫번째 버튼")

    @pyqtSlot()
    def slot_2nd(self):
        self.ui.label.setText("두번째 버튼")

    @pyqtSlot()
    def slot_3rd(self):
        self.ui.label.setText("세번째 버튼")

class Form(QtWidgets.QMainWindow):
    switch_window = QtCore.pyqtSignal(str)
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
        self.ui.actionImage_duplicate_check_2.triggered.connect(self.Img_check)
        self.ui.actionImage_duplicate_check_2.triggered.connect(self.Img_check2)
        self.ui.actionView_Duplicate_list_2.triggered.connect(self.View_D_Image)
        self.ui.File_L.clicked.connect(self.Img_view)
        self.startpos = None
        self.endpos = None
        self.on_similar = False

    def Img_check(self):
        self.on_similar = True
        print(self.file_path)
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())
        print(2)
        self.ui.actionView_Duplicate_list_2.setChecked(True)
        print(3)
        try :
            files = list(self.dataset.files_dict.keys())
        except :
            files = self.file_list
        print(files)
        self.similar_list = check_img(files, self.file_path)
        print(5)



    def Img_check2(self):
        pass

    def View_D_Image(self):
        if self.ui.right_layout.isHidden() :
            self.ui.actionView_Duplicate_list_2.setChecked(True)
        else :
            self.ui.actionView_Duplicate_list_2.setChecked(False)
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())

    def mouseReleaseEvent(self, e):  # e ; QMouseEvent
        print('BUTTON RELEASE')
        self.endpos = (e.x(),e.y())
        print(self.startpos)
        print(self.endpos)
        print(self.ui.Image_V.size())
        print(self.ui.scrollAreaWidgetContents_2.childAt(self.ui.Image_V.pos()).pos())
        print(self.ui.centralwidget.childAt(self.ui.widget.pos()).y())


    def resizeEvent(self, a) :
        print(a.size())
        print(self.ui.File_L.width())

    def mousePressEvent(self, e):  # e ; QMouseEvent
        print('BUTTON PRESS')
        #print(self.img_pos_view)
        self.startpos = (e.x(), e.y())

    def mouseMoveEvent(self, e):

        x = e.x() - 251
        y = e.y() - 26

        text = 'x: {0}, y: {1}'.format(x, y)
        print(self.Image_V.geometry())
        print(text)

    @pyqtSlot()
    def Img_view(self):
        self.ui.Img_D_V_L.clear()
        img = QtGui.QPixmap()
        img_file = self.ui.File_L.currentItem().text()
        img.load(img_file)
        self.ui.Image_V.setFixedSize(img.size())
        self.ui.Image_V.setPixmap(img)
        if self.on_similar :
            self.ui.Img_D_V_L.addItems(self.similar_list[img_file])


    @pyqtSlot()
    def Open(self,val):
        self.file_list = []
        if val == 1 :
            files_path= QtWidgets.QFileDialog.getOpenFileNames(filter="Image Files (*.png *.jpg *.bmp)")[0]
            self.file_list += files_path
        elif val in [2 , 'VOC' , 'YOLO']:
            options = {2:['jpg','png','bmp','jpeg'],'VOC':['xml'],'YOLO':['txt']}[val]
            files_path = QtWidgets.QFileDialog.getExistingDirectory()
            if files_path == '' :
                return
            os.chdir(files_path)
            if val == 2 :
                for i in options :
                    self.file_list += glob(f'*.{i}')
            elif val == 'VOC' :
                xml_files = glob('*.xml')
                self.dataset = VOC(xml_files)
                self.file_list = get_image(self.dataset)

            elif val == 'YOLO' :
                label_file = QtWidgets.QFileDialog.getOpenFileName(filter="Labels (*.txt)",directory=files_path)[0]
                txt_files = glob(f'*.txt')
                self.dataset = YOLO(txt_files, label_file)
                self.file_list = get_image(self.dataset)


        elif val == 'COCO' :
            files_path = QtWidgets.QFileDialog.getOpenFileName(filter="MS COCO (*.json)")[0]
            if files_path == '' :
                return
            os.chdir(files_path.rsplit('/',1)[0])
            self.dataset = COCO(files_path)
            self.file_list = get_image(self.dataset)
        self.file_path = files_path
        self.ui.File_L.addItems(self.file_list)
        self.ui.actionData_Processing.setEnabled(True)
        self.ui.actionData_Trans_Form.setEnabled(True)
        self.ui.actionImage_duplicate_check_2.setEnabled(True)
        self.ui.actionView_Duplicate_list_2.setEnabled(True)
        self.ui.actionView_Duplicate_image.setEnabled(True)
        self.on_similar = False





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())

#QtWidgets.QListWidget.selectedItems()
QtWidgets.QLabel.childAt().pos()
QtWidgets.QFileDialog.getOpenFileNames()

