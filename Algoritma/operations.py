from models import Models
import numpy as np
class Operations:

    def getComboboxItem(self):
        items = ['Siyah Beyaz\'a çevir','Eşikleme','Parlaklık Ayarı','Histogram',
                 'Negatif Görüntü','Alçak Geçiren Filtre','Gauss Alçak Geçiren Filtre',
                 'Ortalama Alçak Geçiren Filtre','Medyan Alçak Geçiren Filtre',
                 'Kontrast Ayarlama','Kontrast Germe','Histogram Eşitleme',
                 'Laplas Filtresi','Sobel Filtresi','Prewitt Fitresi','Açılı Döndürme','Görüntüyü Ters Çevir',
                 'Görüntü Aynalama','Görüntü Öteleme','Yakınlaştır','Uzaklaştır',
                 'Yayma','Aşındırma','Açma','Kapama','Kenar Görüntüsü ile Netleştirme',
                 'Konvolüsyon ile Netleştirme','Korelasyon','Konvolüsyon']

        for item in items:
            self.comboBox.addItem(item)

    def getSelectedItemToModel(self,selected_item,file_name):
        if selected_item == 'Siyah Beyaz\'a çevir':
            return Models.convertBlackAndWhite(file_name)
        elif selected_item == 'Eşikleme':
            threshold = int(self.textbox_varr2.text())
            return Models.esikleme(file_name,threshold=threshold)
        elif selected_item == 'Parlaklık Ayarı':
            brightness = int(self.textbox_varr2.text())
            return Models.parlaklikAyarlama(file_name,brightness=brightness)
        elif selected_item == 'Negatif Görüntü':
            return Models.negativeImage(file_name)
        elif selected_item == 'Histogram':
            return Models.histogram(file_name)
        elif selected_item == 'Alçak Geçiren Filtre':
            kernel = int(self.textbox_varr2.text())
            return Models.alcakGecirenFiltre(file_name,kernel)
        elif selected_item == 'Gauss Alçak Geçiren Filtre':
            sigma = int(self.textbox_varr2.text())
            return Models.gaussAlcakGecirenFiltre(file_name,sigma=sigma)
        elif selected_item == 'Ortalama Alçak Geçiren Filtre':
            kernel = int(self.textbox_varr2.text())
            return Models.meanAlcakGecirenFiltre(file_name,kernel_size=kernel)
        elif selected_item == 'Medyan Alçak Geçiren Filtre':
            kernel = int(self.textbox_varr2.text())
            return Models.medianAlcakGecirenFiltre(file_name,kernel_size=kernel)
        elif selected_item == 'Kontrast Ayarlama':
            a = int(self.textbox_varr2.text())
            b = int(self.textbox_varr1.text())
            return Models.kontrastAyarlama(file_name,a=a,b=b)
        elif selected_item == 'Kontrast Germe':
            return Models.kontrastGerme(file_name)
        elif selected_item == 'Histogram Eşitleme':
            return Models.histogramEsitleme(file_name)
        elif selected_item == 'Laplas Filtresi':
            return Models.laplasFiltre(file_name)
        elif selected_item == 'Sobel Filtresi':
            return Models.sobelFiltre(file_name)
        elif selected_item == 'Prewitt Fitresi':
            return Models.prewittFiltre(file_name)
        elif selected_item == 'Açılı Döndürme':
            angle = int(self.textbox_varr2.text())
            return Models.aciliDondurme(file_name,angle)
        elif selected_item == 'Görüntüyü Ters Çevir':
            return Models.goruntuTersCevirme(file_name)
        elif selected_item == 'Görüntü Aynalama':
            return Models.goruntuAynalama(file_name)
        elif selected_item == 'Görüntü Öteleme':
            yatay = int(self.textbox_varr2.text())
            dikey = int(self.textbox_varr1.text())
            return Models.goruntuOteleme(file_name,yatay=yatay,dikey=dikey)
        elif selected_item == 'Yakınlaştır':
            zoom = int(self.textbox_varr2.text())
            return Models.yakinlastirma(file_name,zoom=zoom)
        elif selected_item == 'Uzaklaştır':
            zoom = int(self.textbox_varr2.text())
            return Models.uzaklastirma(file_name,zoom=zoom)
        elif selected_item == 'Yayma':
            return Models.yayma(file_name)
        elif selected_item == 'Aşındırma':
            return Models.asindirma(file_name)
        elif selected_item == 'Açma':
            return Models.acma(file_name)
        elif selected_item == 'Kapama':
            return Models.kapama(file_name)
        elif selected_item == 'Kenar Görüntüsü ile Netleştirme':
            k = float(self.textbox_varr2.text())
            return Models.kenarGoruntusuIleResmiNetlestirme(file_name,k)
        elif selected_item == 'Konvolüsyon ile Netleştirme':
            return Models.konvolusyonIleNetlestirme(file_name)
        elif selected_item == 'Korelasyon':
            return Models.korelasyon(file_name)
        elif selected_item == 'Konvolüsyon':
            return Models.konvolusyon(file_name)

    def labelNameInputs(self,selected_item):
        self.textbox_varr2.setEnabled(False)
        self.textbox_varr1.setEnabled(False)
        self.label_varr1.setText('Girdi Yok')
        self.label_varr2.setText('Girdi Yok')

        if selected_item == 'Eşikleme':
            self.label_varr1.setText('Eşik Değeri:')
            self.textbox_varr2.setEnabled(True)
        
        elif selected_item == 'Parlaklık Ayarı':
            self.label_varr1.setText('Parlaklık Değeri:')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Alçak Geçiren Filtre':
            self.label_varr1.setText('Çekirdek Değeri:')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Gauss Alçak Geçiren Filtre':
            self.label_varr1.setText('Sigma Değeri:')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Ortalama Alçak Geçiren Filtre':
            self.label_varr1.setText('Çekirdek Değeri:')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Medyan Alçak Geçiren Filtre':
            self.label_varr1.setText('Çekirdek Değeri:')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Kontrast Ayarlama':
            self.label_varr1.setText('Kontrast Değeri:')
            self.textbox_varr2.setEnabled(True)
            self.label_varr2.setText('Parlaklık Değeri:')
            self.textbox_varr1.setEnabled(True)
        elif selected_item == 'Görüntü Öteleme':
            self.label_varr1.setText('Yatay Öteleme Değeri:')
            self.textbox_varr2.setEnabled(True)
            self.label_varr2.setText('Dikey Öteleme Değeri:')
            self.textbox_varr1.setEnabled(True)
        elif selected_item == 'Yakınlaştır':
            self.label_varr1.setText('Yakınlaştırma Miktarı(1,2,3,...):')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Uzaklaştır':
            self.label_varr1.setText('Uzaklaştırma Miktarı(1,2,3,...):')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Kenar Görüntüsü ile Netleştirme':
            self.label_varr1.setText('Keskinleştirme Miktarı \n(k için makul değerler 0,2-0,7\n arasında değişir.) : ')
            self.textbox_varr2.setEnabled(True)
        elif selected_item == 'Açılı Döndürme':
            self.label_varr1.setText('Döndürme Açısı :')
            self.textbox_varr2.setEnabled(True)

    def perspektifDuzeltme(self,file_name):
        x1 = int(self.txt_x1.text())
        y1 = int(self.txt_y1.text())
        x2 = int(self.txt_x2.text())
        y2 = int(self.txt_y2.text())
        x3 = int(self.txt_x3.text())
        y3 = int(self.txt_y3.text())
        x4 = int(self.txt_x4.text())
        y4 = int(self.txt_y4.text())

        X1 = int(self.txt_X1.text())
        Y1 = int(self.txt_Y1.text())
        X2 = int(self.txt_X2.text())
        Y2 = int(self.txt_Y2.text())
        X3 = int(self.txt_X3.text())
        Y3 = int(self.txt_Y3.text())
        X4 = int(self.txt_X4.text())
        Y4 = int(self.txt_Y4.text())

        GirisMatrisi = np.array([
            [x1, y1, 1, 0, 0, 0, -x1 * X1, -y1 * X1],
            [0, 0, 0, x1, y1, 1, -x1 * Y1, -y1 * Y1],
            [x2, y2, 1, 0, 0, 0, -x2 * X2, -y2 * X2],
            [0, 0, 0, x2, y2, 1, -x2 * Y2, -y2 * Y2],
            [x3, y3, 1, 0, 0, 0, -x3 * X3, -y3 * X3],
            [0, 0, 0, x3, y3, 1, -x3 * Y3, -y3 * Y3],
            [x4, y4, 1, 0, 0, 0, -x4 * X4, -y4 * X4],
            [0, 0, 0, x4, y4, 1, -x4 * Y4, -y4 * Y4]
        ], dtype=float)

        MatrisinTersi = np.linalg.inv(GirisMatrisi) 
        
        #a00,a01,a02,a10,a11,a12,a20,a21,a22 = 0

        a00 = MatrisinTersi[0, 0] * X1 + MatrisinTersi[0, 1] * Y1 + MatrisinTersi[0, 2] * X2 + MatrisinTersi[0,3] * Y2 + MatrisinTersi[0,4] * X3 + MatrisinTersi[0,5] * Y3 + MatrisinTersi[0,6] * X4 + MatrisinTersi[0,7] * Y4
        a01 = MatrisinTersi[1, 0] * X1 + MatrisinTersi[1, 1] * Y1 + MatrisinTersi[1, 2] * X2 + MatrisinTersi[1,3] * Y2 + MatrisinTersi[1,4] * X3 + MatrisinTersi[1,5] * Y3 + MatrisinTersi[1,6] * X4 + MatrisinTersi[1,7] * Y4
        a02 = MatrisinTersi[2, 0] * X1 + MatrisinTersi[2, 1] * Y1 + MatrisinTersi[2, 2] * X2 + MatrisinTersi[2,3] * Y2 + MatrisinTersi[2,4] * X3 + MatrisinTersi[2,5] * Y3 + MatrisinTersi[2,6] * X4 + MatrisinTersi[2,7] * Y4
        a10 = MatrisinTersi[3, 0] * X1 + MatrisinTersi[3, 1] * Y1 + MatrisinTersi[3, 2] * X2 + MatrisinTersi[3,3] * Y2 + MatrisinTersi[3,4] * X3 + MatrisinTersi[3,5] * Y3 + MatrisinTersi[3,6] * X4 + MatrisinTersi[3,7] * Y4
        a11 = MatrisinTersi[4, 0] * X1 + MatrisinTersi[4, 1] * Y1 + MatrisinTersi[4, 2] * X2 + MatrisinTersi[4,3] * Y2 + MatrisinTersi[4,4] * X3 + MatrisinTersi[4,5] * Y3 + MatrisinTersi[4,6] * X4 + MatrisinTersi[4,7] * Y4
        a12 = MatrisinTersi[5, 0] * X1 + MatrisinTersi[5, 1] * Y1 + MatrisinTersi[5, 2] * X2 + MatrisinTersi[5,3] * Y2 + MatrisinTersi[5,4] * X3 + MatrisinTersi[5,5] * Y3 + MatrisinTersi[5,6] * X4 + MatrisinTersi[5,7] * Y4
        a20 = MatrisinTersi[6, 0] * X1 + MatrisinTersi[6, 1] * Y1 + MatrisinTersi[6, 2] * X2 + MatrisinTersi[6,3] * Y2 + MatrisinTersi[6,4] * X3 + MatrisinTersi[6,5] * Y3 + MatrisinTersi[6,6] * X4 + MatrisinTersi[6,7] * Y4
        a21 = MatrisinTersi[7, 0] * X1 + MatrisinTersi[7, 1] * Y1 + MatrisinTersi[7, 2] * X2 + MatrisinTersi[7,3] * Y2 + MatrisinTersi[7,4] * X3 + MatrisinTersi[7,5] * Y3 + MatrisinTersi[7,6] * X4 + MatrisinTersi[7,7] * Y4
        a22 = 1

        return Models.perspektifDuzeltme(file_name,a00,a01,a02,a10,a11,a12,a20,a21,a22)




            
