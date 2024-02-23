import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
from PIL import Image
from models import Models
from operations import Operations
from perspektif_screen import Ui_MainWindow 
from imgeArama_screen import Ui_MainWindow 
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1068, 842)
        MainWindow.setMinimumSize(QtCore.QSize(0, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(170, 150, 171, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(850, 150, 171, 16))
        self.label_2.setObjectName("label_2")
        self.input_image = QtWidgets.QLabel(self.centralwidget)
        self.input_image.setGeometry(QtCore.QRect(10, 190, 47, 13))
        self.input_image.setText("")
        self.input_image.setObjectName("input_image")
        self.output_image = QtWidgets.QLabel(self.centralwidget)
        self.output_image.setGeometry(QtCore.QRect(700, 190, 47, 13))
        self.output_image.setText("")
        self.output_image.setObjectName("output_image")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(430, 150, 181, 341))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.comboBox = QtWidgets.QComboBox(self.widget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout.addWidget(self.comboBox, 1, 0, 1, 1)
        self.textbox_varr2 = QtWidgets.QLineEdit(self.widget)
        self.textbox_varr2.setObjectName("textbox_varr2")
        self.gridLayout.addWidget(self.textbox_varr2, 3, 0, 1, 1)
        self.label_varr2 = QtWidgets.QLabel(self.widget)
        self.label_varr2.setMinimumSize(QtCore.QSize(0, 30))
        self.label_varr2.setObjectName("label_varr2")
        self.gridLayout.addWidget(self.label_varr2, 4, 0, 1, 1)
        self.textbox_varr1 = QtWidgets.QLineEdit(self.widget)
        self.textbox_varr1.setObjectName("textbox_varr1")
        self.gridLayout.addWidget(self.textbox_varr1, 5, 0, 1, 1)
        self.confirmFilter = QtWidgets.QPushButton(self.widget)
        self.confirmFilter.setObjectName("confirmFilter")
        self.gridLayout.addWidget(self.confirmFilter, 6, 0, 1, 1)
        self.uploadFile = QtWidgets.QPushButton(self.widget)
        self.uploadFile.setObjectName("uploadFile")
        self.gridLayout.addWidget(self.uploadFile, 0, 0, 1, 1)
        self.label_varr1 = QtWidgets.QLabel(self.widget)
        self.label_varr1.setMinimumSize(QtCore.QSize(0, 30))
        self.label_varr1.setObjectName("label_varr1")
        self.gridLayout.addWidget(self.label_varr1, 2, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1068, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        Operations.getComboboxItem(self)
        self.selected_item = "Siyah Beyaz\'a çevir"
        self.perspektif_button = QtWidgets.QPushButton(self.centralwidget)
        self.perspektif_button.setObjectName('perspektif_button')
        self.perspektif_button.setText('Perspektif Düzenleme Ekranı')
        self.gridLayout.addWidget(self.perspektif_button,7,0,1,1)
        self.imgeArama_button = QtWidgets.QPushButton(self.centralwidget)
        self.imgeArama_button.setObjectName('imgeArama_button')
        self.imgeArama_button.setText('İmge Arama Ekranı')
        self.gridLayout.addWidget(self.imgeArama_button,8,0,1,1)

        
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.uploadFile.clicked.connect(self.uploadFileButton)
        self.confirmFilter.clicked.connect(self.confirmFilterButton)
        self.comboBox.currentIndexChanged.connect(self.handle_combobox_selection)
        self.perspektif_button.clicked.connect(self.perspektif_button_click)
        self.imgeArama_button.clicked.connect(self.imgeArama_button_click)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Converter"))
        self.label.setText(_translate("MainWindow", "GİRİŞ GÖRÜNTÜSÜ"))
        self.label_2.setText(_translate("MainWindow", "ÇIKIŞ GÖRÜNTÜSÜ"))
        self.label_varr2.setText(_translate("MainWindow", "Girdi 2"))
        self.confirmFilter.setText(_translate("MainWindow", "Uygula"))
        self.uploadFile.setText(_translate("MainWindow", "Resim Yükle"))
        self.label_varr1.setText(_translate("MainWindow", "Girdi 1"))
        self.label_varr1.setText('Girdi Yok')
        self.label_varr2.setText('Girdi Yok')
        self.textbox_varr2.setEnabled(False)
        self.textbox_varr1.setEnabled(False)
    
    def perspektif_button_click(self):
        os.system("python perspektif_screen.py")
    
    def imgeArama_button_click(self):
        os.system("python imgeArama_screen.py")

    _fname = ""
    def handle_combobox_selection(self):
        self.selected_item = self.comboBox.currentText()
        Operations.labelNameInputs(self=self,selected_item=self.selected_item)
        

    def show_image(self,file_path,label):
        pixmap = QtGui.QPixmap(file_path)
    
        pixmap = pixmap.scaled(QtCore.QSize(400, 406),QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        label.setPixmap(pixmap)
        
        label.setScaledContents(True)
        label.setFixedSize(QtCore.QSize(400, 406))

    def uploadFileButton(self):
        try:
            fname = QFileDialog.getOpenFileName(None, 'Open file', 
            'c:\\',"Image files (*.jpg *.gif *.png)")
            global _fname
            _fname = fname
            if fname:
                _fname = fname[0]
                self.show_image(str(fname[0]),self.input_image)
        except Exception as error:
            self.output_image.setText("Hata!!! => " + str(error))
    
    def confirmFilterButton(self):
        try:
            self.output_image.setText('')
            converted_image = Operations.getSelectedItemToModel(self=self,selected_item=self.selected_item,file_name=_fname)
            converted_image_path = 'converted_image.jpg'
            converted_image.save(converted_image_path)
            self.show_image(converted_image_path,self.output_image)
        except Exception as error:
            self.output_image.setText("Hata!!! => " + str(error))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.setFixedSize(1200,700)
    MainWindow.show()
    sys.exit(app.exec_())