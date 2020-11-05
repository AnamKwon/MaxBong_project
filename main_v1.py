# coding: utf-8
import sys
import shutil
from libs.Data_type import *
from libs.convType import *
from libs.imagecheck import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot

class Form(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = uic.loadUi("main_V2.ui", self)
        self.ui.show()
        self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())
        self.ui.actionOpen_File.triggered.connect(lambda :self.Open(1))
        self.ui.actionOpen_Directory.triggered.connect(lambda :self.Open(2))
        self.ui.actionCOCO.triggered.connect(lambda: self.Open('COCO'))
        self.ui.actionVOC.triggered.connect(lambda: self.Open('VOC'))
        self.ui.actionYOLO.triggered.connect(lambda: self.Open('YOLO'))
        self.ui.actionCOCO_2.triggered.connect(lambda: self.Save('COCO'))
        self.ui.actionVOC_2.triggered.connect(lambda: self.Save('VOC'))
        self.ui.actionYOLO_2.triggered.connect(lambda: self.Save('YOLO'))
        self.ui.actionImage_duplicate_check_2.triggered.connect(self.Img_check)
        self.ui.actionView_Duplicate_list_2.triggered.connect(self.View_D_L_Image)
        self.ui.actionView_Duplicata_image.triggered.connect(self.View_D_I_Image)
        self.ui.File_L.clicked.connect(self.Img_view)
        self.ui.Img_D_V_L.clicked.connect(self.Img_view_D)
        self.ui.label_object.clicked.connect(self.test_list)
        # self.startpos = None
        # self.endpos = None
        self.on_similar = False
        self.if_dataset = False
        a = ['1','2','3','4','5']
        self.ui.label_object.addItems(a)
        self.test_item = ''

    def Img_check(self):
        self.on_similar = True
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())
        self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
        self.ui.actionView_Duplicate_list_2.setEnabled(True)
        self.ui.actionView_Duplicata_image.setEnabled(True)
        self.ui.actionView_Duplicate_list_2.setChecked(True)
        self.ui.actionView_Duplicata_image.setChecked(True)
        try :
            files = list(self.dataset.files_dict.keys())
        except :
            files = self.file_list
        self.similar_list = check_img(files, self.file_path)

    def View_D_I_Image(self):
        if self.ui.Image_D_H.isHidden() :
            self.ui.actionView_Duplicata_image.setChecked(True)
        else :
            self.ui.actionView_Duplicata_image.setChecked(False)
        self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())

    def View_D_L_Image(self):
        if self.ui.right_layout.isHidden() :
            self.ui.actionView_Duplicate_list_2.setChecked(True)
        else :
            self.ui.actionView_Duplicate_list_2.setChecked(False)
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())

    # def mouseReleaseEvent(self, e):  # e ; QMouseEvent
    #     print('BUTTON RELEASE')
    #     self.endpos = (e.x(),e.y())
    #     print(self.startpos)
    #     print(self.endpos)
    #     print(self.ui.Image_V.size())
    #     print(self.ui.scrollAreaWidgetContents_2.childAt(self.ui.Image_V.pos()).pos())
    #     print(self.ui.centralwidget.childAt(self.ui.widget.pos()).y())
    #
    # def resizeEvent(self, a) :
    #     print(a.size())
    #     print(self.ui.File_L.width())
    #
    # def mousePressEvent(self, e):  # e ; QMouseEvent
    #     print('BUTTON PRESS')
    #     #print(self.img_pos_view)
    #     self.startpos = (e.x(), e.y())
    #     #self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
    #
    # def mouseMoveEvent(self, e):
    #     x = e.x() - 251
    #     y = e.y() - 26
    #     text = 'x: {0}, y: {1}'.format(x, y)
    #     if y > 550 :
    #         print(1)
    #     print(text)
    def test_list(self):
        self.test_item = self.ui.label_object.currentRow()

    def keyPressEvent(self, e) :
        if self.test_item != '' :
            if e.key() == QtCore.Qt.Key_Delete :
                self.ui.label_object.takeItem(self.test_item)
                print(self.test_item.text())



    @pyqtSlot()
    def Img_view(self):
        self.ui.Image_D_V.clear()
        self.ui.Img_D_V_L.clear()

        img_file = self.ui.File_L.currentItem().text()
        img = QtGui.QPixmap(img_file)
        if self.if_dataset :

            for bbox in self.dataset.files_dict[img_file]['object'] :
                painterInstance = QtGui.QPainter(img)
                label, bbox = list(bbox.items())[0]
                penRectangle = QtGui.QPen(self.category_to_color[label])
                penRectangle.setWidth(3)
                painterInstance.setPen(penRectangle)
                x,y,w,h = bbox['bbox']
                painterInstance.drawRect(x, y, w, h)
                painterInstance.setFont(QtGui.QFont('Microsoft Sans Serif',10))
                painterInstance.drawText(x+5, y+15, label)
                painterInstance.end()
                print(painterInstance)

        self.ui.Image_V.setGeometry(0,0,img.width(),img.height())
        self.ui.Image_V.setPixmap(img)
        painterInstance.end()
        if self.on_similar :
            self.similar_img = self.similar_list[img_file]
            self.ui.Img_D_V_L.addItems(self.similar_img)

    @pyqtSlot()
    def Img_view_D(self):
        img = QtGui.QPixmap()
        img_file = self.ui.Img_D_V_L.currentItem().text()
        img.load(img_file)
        self.ui.Image_D_V.setPixmap(img)

    @pyqtSlot()
    def Open(self,val):
        self.ui.File_L.clear()
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

        try :
            self.dataset
            self.if_dataset = True
        except :
            self.if_dataset = False
        self.file_path = files_path
        self.ui.File_L.addItems(self.file_list)
        self.ui.actionData_Processing.setEnabled(True)
        self.ui.actionData_Trans_Form.setEnabled(True)
        self.ui.menuSave_to.setEnabled(True)
        self.ui.actionImage_duplicate_check_2.setEnabled(True)
        self.on_similar = False
        self.category_to_color = {}
        if self.if_dataset :
            for label in self.dataset.categories_to_id.keys():
                self.category_to_color[label] = [0,0,0]
                count = 0
                for i in label :
                    self.category_to_color[label][count] = (self.category_to_color[label][count] + ord(i))%255
                    count = (count+1)%3
                self.category_to_color[label] = QtGui.QColor(self.category_to_color[label][0],self.category_to_color[label][1],self.category_to_color[label][2])

    def Save(self,val):
        files_path = QtWidgets.QFileDialog.getExistingDirectory()
        if val == 'COCO' :
            to_coco(self.dataset,files_path)
            for i in self.dataset.files_dict.keys() :
                shutil.copyfile(i,files_path)
        elif val == 'VOC' :
            to_voc(self.dataset, files_path)
            for i in self.dataset.files_dict.keys() :
                shutil.copyfile(i,files_path)
        elif val == 'YOLO' :
            to_yolo(self.dataset, files_path)
            for i in self.dataset.files_dict.keys() :
                shutil.copy(i, files_path)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())

QtWidgets.QListWidget.currentItem()
QtWidgets.QListWidget.currentRow()