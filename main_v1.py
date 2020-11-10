# coding: utf-8
import sys
import shutil
from libs.Data_type import *
from PIL import Image
from libs.imagecheck import *
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot


# 팝업윈도우
class Transform(QtWidgets.QDialog) :
    def __init__(self, parent=None):
        super().__init__()
        QtWidgets.QDialog.__init__(self, parent)
        self.ui = uic.loadUi('dialog.ui',self)
        self.ui.file_name.clicked.connect(self.Name)
        self.ui.height.clicked.connect(self.Height)
        self.ui.width.clicked.connect(self.Width)

    def Name(self):
        self.ui.file_name_input.setEnabled(not self.ui.file_name_input.isEnabled())

    def Height(self):
        self.ui.height_input.setEnabled(not self.ui.height_input.isEnabled())

    def Width(self):
        self.ui.width_input.setEnabled(not self.ui.width_input.isEnabled())

    def showModal(self):
        return super().exec_()

class Form(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.ui = uic.loadUi("main_V2.ui", self)
        self.ui2 = Transform()
        self.ui.show()
        self.ui.Img_D_V_L.setViewMode(QtWidgets.QListView.IconMode)
        self.ui.Img_D_V_L.setIconSize(QtCore.QSize(200,200))
        self.ui.Img_D_V_L.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())
        self.ui.actionOpen_File.triggered.connect(lambda :self.Open(1))
        self.ui.actionOpen_Directory.triggered.connect(lambda :self.Open(2))
        self.ui.actionOpen_File.triggered.connect(lambda :self.Open(1))
        self.ui.actionOpen_Directory.triggered.connect(lambda :self.Open(2))
        self.ui.actionOpen_File.setEnabled(False)
        self.ui.actionOpen_Directory.setEnabled(False)
        self.ui.actionCOCO.triggered.connect(lambda: self.Open('COCO'))
        self.ui.actionVOC.triggered.connect(lambda: self.Open('VOC'))
        self.ui.actionYOLO.triggered.connect(lambda: self.Open('YOLO'))
        self.ui.actionCOCO_2.triggered.connect(lambda: self.Save('COCO'))
        self.ui.actionVOC_2.triggered.connect(lambda: self.Save('VOC'))
        self.ui.actionYOLO_2.triggered.connect(lambda: self.Save('YOLO'))
        self.ui.actionCOCO_3.triggered.connect(lambda: self.Open('COCO', 'Add'))
        self.ui.actionVOC_3.triggered.connect(lambda: self.Open('VOC', 'Add'))
        self.ui.actionYOLO_3.triggered.connect(lambda: self.Open('YOLO', 'Add'))
        self.ui.actionImage_duplicate_check_2.triggered.connect(self.Img_check)
        self.ui.actionView_Duplicate_list_2.triggered.connect(self.View_D_L_Image)
        self.ui.actionView_Duplicata_image.triggered.connect(self.View_D_I_Image)
        self.ui.File_L.clicked.connect(self.Img_view)
        self.ui.Img_D_V_L.clicked.connect(self.Img_view_D)
        self.ui.select_D.clicked.connect(lambda : self.delete_file(1))
        self.ui.all_D.clicked.connect(lambda :self.delete_file(2))
        self.ui.actionData_Trans_Form.triggered.connect(self.Trans_form)
        self.ui.actionData_Trans_Form.triggered.connect(self.Trans_form_prossing)
        # self.startpos = None
        # self.endpos = None
        self.on_similar = False
        self.on_trans = False


    def delete_file(self, mode):
        if mode == 2 :
            self.ui.Img_D_V_L.selectAll()
        for i in self.ui.Img_D_V_L.selectedItems() :
            self.similar_list[self.ui.File_L.currentItem().text()].remove(i.text())
            self.similar_list[i.text()].remove(self.ui.File_L.currentItem().text())
            try :
                self.file_list.remove(i.text())
            except :
                pass
            for file in self.similar_list[i.text()] :
                self.similar_list[file].remove(i.text())
        self.ui.File_L.clear()
        self.ui.Img_D_V_L.clear()
        self.ui.File_L.addItems(self.file_list)

    def Trans_form(self):
        r = self.ui2.showModal()
        self.trans_form_bools = [False,False,False]
        if r :
            if self.ui2.ui.file_name.isChecked() :
                self.trans_form_bools[0] = self.ui2.ui.file_name_input.text()
            if self.ui2.ui.width.isChecked():
                self.trans_form_bools[1] = self.ui2.ui.width_input.text()
            if self.ui2.ui.height.isChecked():
                self.trans_form_bools[2] = self.ui2.ui.height_input.text()

#--------------------변환작업-------------------------------------------#
    def Trans_form_prossing(self):
        if [False,False,False] == self.trans_form_bools:
            return
        names,  widths, heights = self.trans_form_bools
        cnt = 0
        self.on_trans = True
        self.new_name_to_old_name = {}
        self.new_dataset= {}
        for i in self.file_list :
            if not names :
                name = i
            else :
                name = f'{names}_{cnt}.{i.rsplit(".",1)[1]}'
                self.new_name_to_old_name[i] = name
                cnt += 1
            if not widths :
                width = self.dataset.files_dict[i]['width']
                x = 1
            else :
                width = int(widths)
                x = width/self.dataset.files_dict[i]['width']
            if not heights :
                height = self.dataset.files_dict[i]['height']
                y = 1
            else :
                height = int(heights)
                y = height/self.dataset.files_dict[i]['height']
            objects = []
            for object in self.dataset.files_dict[i]['object'] :
                key, item = list(object.items())[0]
                bbox = item['bbox']
                bbox[0], bbox[1], bbox[2], bbox[3] = round(bbox[0]*x), round(bbox[1]*y), round(bbox[2] *x), round(bbox[3] *y)
                objects.append({key:{'bbox':bbox,'segmentation':[]}})

            self.new_dataset[name] = {'width':width,'height':height,'depth':3,'object':objects}

    # --------------------변환작업-------------------------------------------#
    def Img_check(self):
        self.on_similar = True
        self.ui.right_layout.setHidden(not self.ui.right_layout.isHidden())
        self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
        self.ui.actionView_Duplicate_list_2.setEnabled(True)
        self.ui.actionView_Duplicata_image.setEnabled(True)
        self.ui.actionView_Duplicate_list_2.setChecked(True)
        self.ui.actionView_Duplicata_image.setChecked(True)
        try :
            files = [f'{path}/{name}' for name, path in self.name_to_path.items()]
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
    def mousePressEvent(self, e):  # e ; QMouseEvent
        print('BUTTON PRESS')
        if e.button() == QtCore.Qt.LeftButton :
            self.ui.frame.setAutoFillBackground(True)
        if e.button() == QtCore.Qt.RightButton :
            self.ui.frame.setAutoFillBackground(False)
        #print(self.img_pos_view)


        #self.ui.Image_D_H.setHidden(not self.ui.Image_D_H.isHidden())
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

    @pyqtSlot()
    def Img_view(self):
        self.ui.Image_D_V.clear()
        self.ui.Img_D_V_L.clear()

        img_file = self.ui.File_L.currentItem().text()
        file_path = f'{self.name_to_path[img_file]}/{img_file}'
        img = QtGui.QPixmap(file_path)
        if self.if_dataset :

            for bbox in self.dataset.files_dict[img_file]['object'] :
                painterInstance = QtGui.QPainter(img)
                label, bbox = list(bbox.items())[0]
                penRectangle = QtGui.QPen(self.category_to_color[label])
                penRectangle.setWidth(3)
                painterInstance.setPen(penRectangle)
                x, y, w, h = bbox['bbox']
                painterInstance.drawRect(x, y, w, h)
                painterInstance.setFont(QtGui.QFont('Microsoft Sans Serif',10))
                painterInstance.drawText(x+5, y+15, label)
                painterInstance.end()
        self.ui.Image_V.setPixmap(img)
        painterInstance.end()
        if self.on_similar :
            self.similar_img = self.similar_list[img_file]
            self.ui.Img_D_V_L.addItems(self.similar_img)

    @pyqtSlot()
    def Img_view_D(self):
        # Qframe으로 백그라운드 방식
        # 오브젝트들을 선택할수 잇게 가능할듯?
        # img = QtGui.QPixmap(r'./dataset/coco/000002.jpg')
        # p = QtGui.QPalette()
        # p.setBrush(QtGui.QPalette.Background, QtGui.QBrush(img))
        # self.ui.frame.setMinimumSize(1000,1000)
        # self.ui.frame.setMaximumSize(1000,1000)
        # self.ui.frame.setAutoFillBackground(True)
        # self.ui.frame.setPalette(p)
        # self.ui.mdiArea.setStyleSheet('color:rgb(255,0,0);')
        img = QtGui.QPixmap()
        img_file = self.ui.Img_D_V_L.currentItem().text()
        file_path = f'{self.name_to_path[img_file]}/{img_file}'
        img.load(file_path)
        self.ui.Image_D_V.setPixmap(img)

    @pyqtSlot()
    def Open(self,val,mode=''):
        self.ui.File_L.clear()
        if mode == '':
            self.file_list = []
            self.name_to_path = {}
        if val == 1 :
            files_path= QtWidgets.QFileDialog.getOpenFileNames(filter="Image Files (*.png *.jpg *.bmp)")[0]
            file_list = files_path

        elif val in [2 , 'VOC' , 'YOLO']:
            options = {2:['jpg','png','bmp','jpeg'],'VOC':['xml'],'YOLO':['txt']}[val]
            files_path = QtWidgets.QFileDialog.getExistingDirectory()
            os.chdir(files_path)
            if files_path == '' :
                return
            if val == 2 :
                for i in options :
                    file_list = glob(f'*.{i}')
            elif val == 'VOC' :
                xml_files = glob('*.xml')
                dataset = VOC(xml_files)
                file_list = get_image(dataset)
            elif val == 'YOLO' :
                label_file = QtWidgets.QFileDialog.getOpenFileName(filter="Labels (*label*.txt)",directory=files_path)[0]
                if label_file == '':
                    return
                txt_files = glob(f'*.txt')
                dataset = YOLO(txt_files, label_file)
                file_list = get_image(dataset)

        elif val == 'COCO' :
            files_path = QtWidgets.QFileDialog.getOpenFileName(filter="MS COCO (*.json)")[0]
            if files_path == '' :
                return
            os.chdir(files_path.rsplit('/',1)[0])
            dataset = COCO(files_path)
            files_path = files_path.rsplit('/',1)[0]
            file_list = get_image(dataset)

        if mode == 'Add' and val in ['COCO','VOC','YOLO'] :
            self.dataset.files_dict.update(dataset.files_dict)
            num = max(self.dataset.id_to_categories.keys())
            values = self.dataset.id_to_categories.values()
            for category in dataset.categories_to_id.keys() :
                if category not in values :
                    self.dataset.id_to_categories[num+1] = category
                    self.dataset.categories_to_id[category] = num+1
                    num += 1
        elif val in ['COCO','VOC','YOLO'] :
            self.dataset = dataset
        for file in file_list :
            if file not in self.file_list :
                self.file_list.append(file)
        try :
            self.dataset
            self.name_to_path.update({key: files_path for key in self.dataset.files_dict.keys()})
            self.if_dataset = True
        except :
            self.if_dataset = False
        self.file_path = files_path
        self.ui.File_L.addItems(self.file_list)
        self.ui.menuadd_Data_set.setEnabled(True)
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
        if self.on_trans :
            self.dataset.files_dict = self.new_dataset

        if val == 'COCO' :
            to_coco(self.dataset, files_path)
        elif val == 'VOC' :
            to_voc(self.dataset, files_path)
        elif val == 'YOLO' :
            to_yolo(self.dataset, files_path)

        for i in self.file_list:
            if self.on_trans:
                new_name = self.new_name_to_old_name[i]
                img = Image.open(f'{self.name_to_path[i]}/{i}').resize(
                    (self.dataset.files_dict[new_name]['width'], self.dataset.files_dict[new_name]['height']))
                img.save(f'{files_path}/{new_name}')
            else:
                shutil.copy(i, files_path);
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = Form()
    sys.exit(app.exec())

QtWidgets.QFrame.setMinimumSize()
QtWidgets.QLabel.setStyle()
QtWidgets.QMdiArea