from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
from models import Models
from matplotlib import pyplot as plt

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(937, 704)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(350, 80, 160, 141))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.imgeBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.imgeBtn.setObjectName("imgeBtn")
        self.verticalLayout.addWidget(self.imgeBtn)
        self.imge2Btn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.imge2Btn.setObjectName("imge2Btn")
        self.verticalLayout.addWidget(self.imge2Btn)
        self.hesaplaBtn = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.hesaplaBtn.setObjectName("hesaplaBtn")
        self.verticalLayout.addWidget(self.hesaplaBtn)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 20, 121, 20))
        self.label.setObjectName("label")
        self.image1 = QtWidgets.QLabel(self.centralwidget)
        self.image1.setGeometry(QtCore.QRect(20, 50, 47, 13))
        self.image1.setText("")
        self.image1.setObjectName("image1")
        self.image2 = QtWidgets.QLabel(self.centralwidget)
        self.image2.setGeometry(QtCore.QRect(30, 330, 47, 13))
        self.image2.setText("")
        self.image2.setObjectName("image2")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(620, 20, 81, 16))
        self.label_2.setObjectName("label_2")
        self.output_image = QtWidgets.QLabel(self.centralwidget)
        self.output_image.setGeometry(QtCore.QRect(530, 50, 47, 13))
        self.output_image.setText("")
        self.output_image.setObjectName("output_image")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(360, 10, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 937, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.imgeBtn.setText(_translate("MainWindow", "Ana Görüntü"))
        self.imge2Btn.setText(_translate("MainWindow", "Aranacak Görüntü"))
        self.hesaplaBtn.setText(_translate("MainWindow", "Hesapla"))
        self.label.setText(_translate("MainWindow", "Ana Görüntü"))
        self.label_2.setText(_translate("MainWindow", "Çıktı Görüntüsü"))
        self.label_3.setText(_translate("MainWindow", "İMGE ARAMA"))

        self.imgeBtn.clicked.connect(self.imgeBtn_click)
        self.imge2Btn.clicked.connect(self.imge2Btn_click)
        self.hesaplaBtn.clicked.connect(self.hesaplaBtn_click)

    def show_image(self,file_path,label):
        pixmap = QtGui.QPixmap(file_path)
    
        pixmap = pixmap.scaled(QtCore.QSize(250, 256),QtCore.Qt.AspectRatioMode.KeepAspectRatio)

        label.setPixmap(pixmap)
        
        label.setScaledContents(True)
        label.setFixedSize(QtCore.QSize(250, 256))
    
    def show_image2(self, file_path, label):
        pixmap = QtGui.QPixmap(file_path)
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        label.setFixedSize(pixmap.size())
    
    _fname = ""
    _fname2 = ""
    def imgeBtn_click(self):
        try:
            fname = QFileDialog.getOpenFileName(None, 'Open file', 
            'c:\\',"Image files (*.jpg *.gif *.png)")
            global _fname
            _fname = fname
            if fname:
                _fname = fname[0]
                self.show_image(str(fname[0]),self.image1)
        except Exception as error:
            self.output_image.setText("Hata!!! => " + str(error))
    
    def imge2Btn_click(self):
        try:
            fname2 = QFileDialog.getOpenFileName(None, 'Open file', 
            'c:\\',"Image files (*.jpg *.gif *.png)")
            global _fname2
            _fname2 = fname2
            if fname2:
                _fname2 = fname2[0]
                self.show_image(str(fname2[0]),self.image2)
        except Exception as error:
            self.output_image.setText("Hata!!! => " + str(error))
    
    def hesaplaBtn_click(self):
        try:
            self.output_image.setText('')
            converted_image = Models.crossCorrelation(_fname, _fname2)

            
            threshold = 0.5

            # Skorları normalize etme (0-1 aralığına getirme)
            normalized_image = (converted_image - converted_image.min()) / (converted_image.max() - converted_image.min())

            # Eşik değerini aşan bölgeleri vurgulama
            highlighted_image = normalized_image.copy()
            highlighted_image[highlighted_image < threshold] = 1

            plt.imshow(highlighted_image, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig('imge_search_result.jpg')
            self.show_image2('imge_search_result.jpg', self.output_image)
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
