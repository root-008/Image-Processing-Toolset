from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Models:

    def openImageAndGetSize(img_path):
        image = Image.open(img_path).convert('RGB')
        width,height = image.size

        return image,width,height
    
    def convertBlackAndWhite(img_path):

        image,width,height = Models.openImageAndGetSize(img_path)

        image = image.convert("RGB")
        new_image = Image.new("RGB",(width,height))

        for x in range(width):
            for y in range(height):
                r,g,b = image.getpixel((x,y))
                gray = int(0.299*r + 0.587*g + 0.114*b)
                new_image.putpixel((x,y),(gray,gray,gray))

        return new_image
    
    def esikleme(img_path,threshold):
        image = Image.open(img_path)
        image = image.convert("L")
        width,height = image.size

        new_image = Image.new("L",(width,height))

        for x in range(width):
            for y in range(height):
                pixel_value = image.getpixel((x,y))
                if pixel_value < threshold:
                    new_pixel_value = 0
                else:
                    new_pixel_value = 255
                new_image.putpixel((x,y),new_pixel_value)
        
        return new_image
    
    def parlaklikAyarlama(img_path,brightness):
        image,width,height = Models.openImageAndGetSize(img_path)
        new_image = Image.new("RGB",(width,height))
        for x in range(width):
            for y in range(height):
                r,g,b = image.getpixel((x,y))

                r = int(r * brightness)
                g = int(g * brightness)
                b = int(b * brightness)

                r = min(255,max(0,r))
                g = min(255,max(0,g))
                b = min(255,max(0,b))

                new_image.putpixel((x,y),(r,g,b))
        
        return new_image

    def histogram(file_name):
        image = Image.open(file_name).convert('RGB')
        width,height = image.size

        histogram = [0] * 256

        for x in range(width):
            for y in range(height):
                r,g,b = image.getpixel((x,y))
                gray = int((r+g+b)/3)
                histogram[gray] += 1
        
        plt.bar(range(256),histogram,width=1.0,color='b')
        plt.xlabel('Piksel Değeri')
        plt.ylabel('Piksel Sayısı')
        plt.title('Görüntünün Histogramı')
        plt.savefig('converted_image2.jpg')
        image = Image.open('converted_image2.jpg')
        return image

    def negativeImage(img_path):
        image = Image.open(img_path)
        image = image.convert("L")
        width,height = image.size
        new_image = Image.new("L",(width,height))

        for x in range(width):
            for y in range(height):
                pixel_value = image.getpixel((x,y))
                negative_pixel_value = 255-pixel_value
                new_image.putpixel((x,y),negative_pixel_value)
        
        return new_image
        
    def alcakGecirenFiltre(img_path,ker):
        image = Image.open(img_path).convert('L')
        image_array = np.array(image, dtype=np.float64) / 255.0
        kernel = np.ones((ker,ker))
        result = np.zeros_like(image_array)
        height, width = image_array.shape

        k_height, k_width = kernel.shape
        k_center = (k_height // 2, k_width // 2)

        for i in range(height):
            for j in range(width):
                value = 0.0
                for m in range(k_height):
                    for n in range(k_width):
                        ii = i + m - k_center[0]
                        jj = j + n - k_center[1]
                        if ii >= 0 and ii < height and jj >= 0 and jj < width:
                            value += image_array[ii, jj] * kernel[m, n]
                result[i, j] = value
                


        return Image.fromarray((result * 255).astype(np.uint8))

    def gaussAlcakGecirenFiltre(img_path,sigma):
        image = Image.open(img_path).convert('L')
        # filtre boyutu
        size = 5  
        image = np.array(image)
        width,height = image.shape[:2]
        # Gauss filtresi kernel'ini oluştur
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        #print('Kernel : ' , kernel)
        # Normalize et
        kernel = kernel / np.sum(kernel)
        #print('Normalize Kernel : ', kernel)
        # Resmi kenarlardan genişleterek bir dizi oluştur
        I2 = np.pad(image,((2, 2), (2, 2)), mode='constant', constant_values=0)
        #print(I2.shape)
        I3 = np.zeros_like(image,dtype=float)
        #print(I3.shape)
        for x in range(width):
            for y in range(height):
                I3[x,y] = np.sum(I2[x:x+5, y:y+5]*kernel)
        
        return  Image.fromarray(I3.astype('uint8'))

    def meanAlcakGecirenFiltre(img_path,kernel_size):
        
        image = Image.open(img_path).convert('RGB')

        image_array = np.array(image)

        width, height, channels = image_array.shape
        border_size = kernel_size // 2

        I2 = np.pad(image_array, ((border_size, border_size), (border_size, border_size), (0, 0)), mode='constant', constant_values=0)

        I3 = np.zeros_like(image_array, dtype=float)

        for x in range(border_size, width + border_size):
            for y in range(border_size, height + border_size):
                for z in range(channels):
                    I3[x - border_size, y - border_size, z] = np.mean(I2[x - border_size:x + border_size + 1, y - border_size:y + border_size + 1, z])
        return Image.fromarray(I3.astype('uint8'))
        """
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        width,height = image.shape[:2]
        border_size = kernel_size // 2

        # Resmi kenarlardan genişleterek bir dizi oluştur
        I2 = np.pad(image, ((border_size, border_size), (border_size, border_size)), mode='constant', constant_values=0)
        # Filtrelenmiş resim için bir dizi oluştur
        I3 = np.zeros_like(image, dtype=float)

        for x in range(border_size, width + border_size):
            for y in range(border_size, height + border_size):
                I3[x - border_size, y - border_size] = np.mean(I2[x - border_size:x + border_size + 1, y - border_size:y + border_size + 1])
        """
        
    def medianAlcakGecirenFiltre(img_path,kernel_size):
        image = Image.open(img_path).convert('L')

        image = np.array(image)
        width,height = image.shape[:2]
        border_size = kernel_size // 2

        # Resmi kenarlardan genişleterek bir dizi oluştur
        I2 = np.pad(image, ((border_size, border_size), (border_size, border_size)), mode='constant', constant_values=0)
        
        # Filtrelenmiş resim için bir dizi oluştur
        I3 = np.zeros_like(image, dtype=float)
        for x in range(width):
            for y in range(height):
                # Çekirdek boyutu içindeki pikselleri al
                kernel_pixels = I2[x:x+kernel_size, y:y+kernel_size].flatten()
                
                # Median değeri hesapla
                median_value = np.median(kernel_pixels)
                
                I3[x, y] = median_value
    
        return Image.fromarray(I3.astype('uint8'))
    
    def kontrastAyarlama(img_path,a,b):
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        width,height = image.shape[:2]

        new_image = np.zeros_like(image,dtype=float)

        for x in range(width):
            for y in range(height):
                pixel_value = image[x,y]
                new_pixel_value = a * pixel_value + b # g(x,y) = a*f(x,y) + b
                new_pixel_value = max(0,min(255,new_pixel_value))
                new_image[x,y] = new_pixel_value
        
        return Image.fromarray(new_image.astype('uint8'))

    def kontrastGerme(img_path):
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        width,height = image.shape[:2]
        # Görüntüdeki en küçük ve en büyük değerleri bul
        min_value = np.min(image)
        max_value = np.max(image)

        new_image = np.zeros_like(image,dtype=float)

        for x in range(width):
            for y in range(height):
                pixel_value = image[x,y]
                new_pixel_value = 255 * (pixel_value - min_value) / (max_value - min_value)

                new_image[x,y] = new_pixel_value

        return Image.fromarray(new_image.astype('uint8'))
    
    def histogramEsitleme(img_path):
        # https://en.wikipedia.org/wiki/Histogram_equalization
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        width,height = image.shape[:2]

        his = [0] * 256

        new_image = np.zeros_like(image,dtype=float)

        # histogramı hesapla
        for x in range(width):
            for y in range(height):
                pixel_value = image[x,y]
                his[pixel_value] += 1
        
        cdf = np.cumsum(his) #kümülatif dağılım fonksiyonunu
        cdf_min = np.min(cdf[cdf != 0])

        for x in range(width):
            for y in range(height):
                pixel_value = image[x,y]
                new_image[x,y] = round( ((cdf[pixel_value] - cdf_min) / ((width*height) - cdf_min )) * 255 )

        return Image.fromarray(new_image.astype('uint8'))

    def laplasFiltre(img_path):
        # https://bilgisayarkavramlari.com/2008/10/29/laplas-filitresi-laplace-filter/
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        width, height = image.shape[:2]

        laplace = np.array([[1, 1, 1],
                            [1, -8, 1],
                            [1, 1, 1]])

        new_image = np.zeros_like(image, dtype=float)

        for x in range(1, width-1):
            for y in range(1, height-1):
                
                new_pixel_value = np.sum(image[x-1:x+2, y-1:y+2] * laplace)
                
                new_pixel_value = np.clip(new_pixel_value, 0, 255)

                new_image[x, y] = new_pixel_value

        return Image.fromarray(new_image.astype('uint8'))

    def sobelFiltre(img_path):
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        
        sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])
        
        # x ve y yönlü türevleri hesapla
        grad_x = np.convolve(image.flatten(), sobel_x.flatten(), 'same').reshape(image.shape)
        grad_y = np.convolve(image.flatten(), sobel_y.flatten(), 'same').reshape(image.shape)

        # Gradyanları birleştirerek genel gradyan elde et
        grad = np.sqrt(grad_x**2 + grad_y**2)

        # Gradyanları 0 ile 255 arasına kırp
        grad = np.clip(grad, 0, 255)
        
        return Image.fromarray(grad.astype('uint8'))

    def prewittFiltre(img_path):
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        
        prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

        prewitt_y = np.array([[-1, -1, -1],
                            [0, 0, 0],
                            [1, 1, 1]])
        
        # x ve y yönlü türevleri hesapla
        grad_x = np.convolve(image.flatten(), prewitt_x.flatten(), 'same').reshape(image.shape)
        grad_y = np.convolve(image.flatten(), prewitt_y.flatten(), 'same').reshape(image.shape)

        # Gradyanları birleştirerek genel gradyan elde et
        grad = np.sqrt(grad_x**2 + grad_y**2)

        # Gradyanları 0 ile 255 arasına kırp
        grad = np.clip(grad, 0, 255)
        
        return Image.fromarray(grad.astype('uint8'))
    
    def aciliDondurme(img_path,angle):
        image = Image.open(img_path)
        
        angle_rad = np.radians(angle)
        
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
            ])

        width, height = image.size

        new_width = int(np.ceil(width * np.abs(np.cos(angle_rad)) + height * np.abs(np.sin(angle_rad))))
        new_height = int(np.ceil(width * np.abs(np.sin(angle_rad)) + height * np.abs(np.cos(angle_rad))))
        output_image = Image.new("RGB", (new_width, new_height), (255, 255, 255))
        # Dönüş merkezini belirle
        center_x, center_y = new_width // 2, new_height // 2

        for x in range(new_width):
            for y in range(new_height):
                # Pikselin orijinal konumunu bul
                original_x = int((x - center_x) * np.cos(-angle_rad) - (y - center_y) * np.sin(-angle_rad) + width / 2)
                original_y = int((x - center_x) * np.sin(-angle_rad) + (y - center_y) * np.cos(-angle_rad) + height / 2)

                # Eğer orijinal konum geçerliyse, pikseli al ve yeni resme ekle
                if 0 <= original_x < width and 0 <= original_y < height:
                    pixel = image.getpixel((original_x, original_y))
                    output_image.putpixel((x, y), pixel)
        
        return output_image

    def goruntuTersCevirme(img_path):
        image = Image.open(img_path).convert('RGB')
        width,height= image.size

        new_image = Image.new("RGB",(width,height))

        for x in range(width):
            for y in range(height):
                r,g,b = image.getpixel((x, y))
            
                # Yeni görüntüdeki koordinatlar
                w1 = width - x - 1
                h1 = height - y - 1
                
                new_image.putpixel((w1, h1), (r,g,b))
            
        return new_image

    def goruntuAynalama(img_path):
        image = Image.open(img_path)
        width,height = image.size

        new_image = Image.new("RGB",(width,height))

        for x in range(width):
            for y in range(height):
                pixel_value = image.getpixel((x, y))

                w1 = width-x-1
               
                
                new_image.putpixel((w1, y), pixel_value)
            
        return new_image

    def goruntuOteleme(img_path, yatay=0, dikey=0):
        image = Image.open(img_path)
        width, height = image.size
        new_image = Image.new("RGB", (width, height))

        for x in range(width):
            i = x + yatay
            if 0 <= i < width:  # Yeni konum sınırlar içinde mi kontrol et
                for y in range(height):
                    j = y + dikey
                    if 0 <= j < height:  # Yeni konum sınırlar içinde mi kontrol et
                        pixel_value = image.getpixel((x, y))
                        new_image.putpixel((i, j), pixel_value)

        return new_image

    def yakinlastirma(img_path, zoom=1):
        image = Image.open(img_path).convert('L')
        image = np.array(image)

       
        height, width = image.shape[:2]

        # Yeni bir görüntü oluştur (zoom faktörü ile çarpılarak boyutlar arttırılır)
        new_height, new_width = int(height * zoom), int(width * zoom)
        new_image = np.zeros((new_height, new_width), dtype=float)

        # Interpolasyon işlemi ile yakınlaştırma yap
        for i in range(new_height):
            for j in range(new_width):
                # Orjinal görüntüdeki piksel konumunu belirle
                orig_i, orig_j = i / zoom, j / zoom

                # Eğer orijinal piksel konumu görüntü sınırları içerisindeyse
                if 0 <= orig_i < height - 1 and 0 <= orig_j < width - 1:
                    # Yakın pikselleri kullanarak interpolasyon yap
                    i0, i1 = int(round(orig_i)), min(int(round(orig_i)) + 1, height - 1)
                    j0, j1 = int(round(orig_j)), min(int(round(orig_j)) + 1, width - 1)

                    # Bilinear İnterpolasyon hesaplaması
                    """
                    İki eksen boyunca lineer interpolasyon yapılır.
                    İki yönü (yatay ve dikey) için ağırlıklı ortalama kullanılır.
                    Daha pürüzsüz sonuçlar elde etmek için kullanılır.
                    """
                    new_image[i, j] = (
                        (i1 - orig_i) * ((j1 - orig_j) * image[i0, j0] + (orig_j - j0) * image[i0, j1]) +
                        (orig_i - i0) * ((j1 - orig_j) * image[i1, j0] + (orig_j - j0) * image[i1, j1])
                    )

        return Image.fromarray(new_image.astype('uint8'))
        
    def uzaklastirma(img_path, zoom=1):
        image = Image.open(img_path).convert('L')
        image = np.array(image)

        height, width = image.shape[:2]

        # Yeni bir görüntü oluştur (zoom faktörü ile bölünerek boyutlar azaltılır)
        new_height, new_width = int(height / zoom), int(width / zoom)
        new_image = np.zeros((new_height, new_width), dtype=float)

        # Uzaklaştırma işlemi
        for i in range(new_height):
            for j in range(new_width):
                # Orjinal görüntüdeki piksel konumunu belirle
                orig_i, orig_j = i * zoom, j * zoom

                # Eğer orijinal piksel konumu görüntü sınırları içerisindeyse
                if 0 <= orig_i < height - 1 and 0 <= orig_j < width - 1:
                    # Uzak pikselleri kullanarak interpolasyon yap
                    i0, i1 = int(orig_i), min(int(orig_i) + 1, height - 1)
                    j0, j1 = int(orig_j), min(int(orig_j) + 1, width - 1)

                    # Bilinear İnterpolasyon hesaplaması
                    """
                    İki eksen boyunca lineer interpolasyon yapılır.
                    İki yönü (yatay ve dikey) için ağırlıklı ortalama kullanılır.
                    Daha pürüzsüz sonuçlar elde etmek için kullanılır.
                    """
                    new_image[i, j] = (
                        (i1 - orig_i) * ((j1 - orig_j) * image[i0, j0] + (orig_j - j0) * image[i0, j1]) +
                        (orig_i - i0) * ((j1 - orig_j) * image[i1, j0] + (orig_j - j0) * image[i1, j1])
                    )

        return Image.fromarray(new_image.astype('uint8'))

    def yayma(img_path):
        # https://medium.com/@hilalkrkrt360/bilgisayarl%C4%B1-g%C3%B6r%C3%BC-yap%C4%B1land%C4%B1rma-morfolojik-i%CC%87%C5%9Flemler-e0974371daee
        # https://guraysonugur.aku.edu.tr/wp-content/uploads/sites/11/2018/04/GI-Ders-9.pdf
        image = Models.esikleme(img_path, 128)
        image_array = np.array(image)

        height, width = image_array.shape[:2]

        new_image = np.zeros((height, width), dtype='uint8')

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                # Çevresindeki piksellerin değerleri
                region = image_array[y-1:y+2, x-1:x+2]

                if np.any(region == 255):
                    new_image[y, x] = 255
                else:
                    new_image[y,x] = 0

        return Image.fromarray(new_image)
        
    def asindirma(img_path):
        image = Models.esikleme(img_path, 128)
        image_array = np.array(image)

        height, width = image_array.shape[:2]

        new_image = np.zeros((height-1, width-1), dtype='uint8')
        

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                region = image_array[y-1:y+2,x-1:x+2]

                if np.any(region == 0):
                    new_image[y,x] = 0
                else:
                    new_image[y,x] = 255

        #count_zeros2 = np.count_nonzero(new_image == 0) # image7 için 168267
        #count_2552 = np.count_nonzero(new_image == 255) # image7 için 25531
        #count_zeros1 = np.count_nonzero(image_array == 0) # image7 için 168049
        #count_2551 = np.count_nonzero(image_array == 255) # image7 için 26639
        return Image.fromarray(new_image)
    
    def acma(img_path):
        #aşındırma ardından yayma
        img = Models.asindirma(img_path)

        image_array = np.array(img)

        height, width = image_array.shape[:2]

        new_image = np.zeros((height, width), dtype='uint8')

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                # Çevresindeki piksellerin değerleri
                region = image_array[y-1:y+2, x-1:x+2]

                if np.any(region == 255):
                    new_image[y, x] = 255
                else:
                    new_image[y,x] = 0

        return Image.fromarray(new_image)
        
    def kapama(img_path):
        #yayma ardından aşındırma

        img = Models.yayma(img_path)
        image_array = np.array(img)

        height, width = image_array.shape[:2]

        new_image = np.zeros((height-1, width-1), dtype='uint8')
        

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                region = image_array[y-1:y+2,x-1:x+2]

                if np.any(region == 0):
                    new_image[y,x] = 0
                else:
                    new_image[y,x] = 255
        return Image.fromarray(new_image)
    
    def kenarGoruntusuIleResmiNetlestirme(img_path,k):

        # g(x,y)=f(x,y)-fsmooth(x,y)  fsharp(x,y)=f(x,y) + k* g(x,y))
        # Burada k bir ölçekleme sabitidir. k için makul değerler 0,2-0,7 arasında değişir. k büyüdükçe keskinleştirme miktarı artar.
        # g(x,y) çıkış görüntüsü, f(x,y) giriş görüntüsü ve fsmooth(x,y) de bu görünün yumuşatılmış hali
        # Görüntünün yumuşatılmış versiyonunu oluşturmak için 3x3 lük Ortalama filtresi (mean filter) kullanılabilir.

        inputImage = Image.open(img_path).convert('RGB')
        smoothImage = Models.meanAlcakGecirenFiltre(img_path,kernel_size=9).convert('RGB')

        width,height = inputImage.size

        edgeImage = Image.new("RGB",(width,height))
        sharpImage = Image.new("RGB",(width,height))

        for x in range(width):
            for y in range(height):
                inputImage_pixel_R,inputImage_pixel_G,inputImage_pixel_B = inputImage.getpixel((x,y))
                smoothImage_pixel_R,smoothImage_pixel_G,smoothImage_pixel_B = smoothImage.getpixel((x,y))

                outputImage_pixel_R = int((inputImage_pixel_R-smoothImage_pixel_R)*k)
                outputImage_pixel_G = int((inputImage_pixel_G-smoothImage_pixel_G)*k)
                outputImage_pixel_B = int((inputImage_pixel_B-smoothImage_pixel_B)*k)

                outputImage_pixel_R = min(255,max(0,outputImage_pixel_R))
                outputImage_pixel_G = min(255,max(0,outputImage_pixel_G))
                outputImage_pixel_B = min(255,max(0,outputImage_pixel_B))

                edgeImage.putpixel((x,y),(outputImage_pixel_R,outputImage_pixel_G,outputImage_pixel_B))
        
        for i in range(width):
            for j in range(height):
                inputImage_pixel_R,inputImage_pixel_G,inputImage_pixel_B = inputImage.getpixel((i,j))
                edgeImage_pixel_R,edgeImage_pixel_G,edgeImage_pixel_B = edgeImage.getpixel((i,j))

                sharpImage_pixel_R = inputImage_pixel_R + edgeImage_pixel_R
                sharpImage_pixel_G = inputImage_pixel_G + edgeImage_pixel_G
                sharpImage_pixel_B = inputImage_pixel_B + edgeImage_pixel_B

                sharpImage_pixel_R = min(255,max(0,sharpImage_pixel_R))
                sharpImage_pixel_G = min(255,max(0,sharpImage_pixel_G))
                sharpImage_pixel_B = min(255,max(0,sharpImage_pixel_B))
                
                sharpImage.putpixel((i,j),(sharpImage_pixel_R,sharpImage_pixel_G,sharpImage_pixel_B))
        
        return sharpImage
                
    def konvolusyonIleNetlestirme(img_path):
        giris_resmi = Image.open(img_path).convert('RGB')
        cikis_resmi = giris_resmi.copy()

        sablon_boyutu = 3
        matris = [0, -2, 0, -2, 11, -2, 0, -2, 0]
        matris_toplami = sum(matris)

        for x in range((sablon_boyutu - 1) // 2, giris_resmi.width - (sablon_boyutu - 1) // 2):
            for y in range((sablon_boyutu - 1) // 2, giris_resmi.height - (sablon_boyutu - 1) // 2):
                toplam_r, toplam_g, toplam_b = 0, 0, 0
                k = 0

                for i in range(-((sablon_boyutu - 1) // 2), (sablon_boyutu - 1) // 2 + 1):
                    for j in range(-((sablon_boyutu - 1) // 2), (sablon_boyutu - 1) // 2 + 1):
                        okunan_renk = giris_resmi.getpixel((x + i, y + j))
                        toplam_r += okunan_renk[0] * matris[k]
                        toplam_g += okunan_renk[1] * matris[k]
                        toplam_b += okunan_renk[2] * matris[k]
                        k += 1

                r = toplam_r // matris_toplami
                g = toplam_g // matris_toplami
                b = toplam_b // matris_toplami

                r = min(255, max(0, r))
                g = min(255, max(0, g))
                b = min(255, max(0, b))

                cikis_resmi.putpixel((x, y), (r, g, b))

        return cikis_resmi

    def perspektifDuzeltme(img_path,a00,a01,a02,a10,a11,a12,a20,a21,a22):
        image = Image.open(img_path).convert('RGB')
        width,height = image.size
        output_image = Image.new('RGB',(width,height))

        for x in range(width):
            for y in range(height):
                pixel = image.getpixel((x,y))
                z = a20 * x + a21 * y + 1 
                X = (a00 * x + a01 *y + a02) / z
                Y = (a10 * x + a11 *y + a12) / z

                if (X > 0 and X < width and Y > 0 and Y < height):
                    output_image.putpixel((int(X),int(Y)),pixel)
        
        return output_image

    def korelasyon(img_path):
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        height, width = image.shape[:2]

        new_image = np.zeros((height, width), dtype='uint8')

        kernel = np.array([[-1,-1,-1],
                            [-2,10,-2],
                            [-1,-1,-1]])
        
        for x in range(1,width-1):
            for y in range(1,height-1):
                region = image[y-1:y+2, x-1:x+2]
                result = np.multiply(region,kernel)
                result_sum = np.sum(result)
                result_sum = min(255, max(0, result_sum))
                new_image[y,x] = result_sum

        return Image.fromarray(new_image)

    def konvolusyon(img_path):
        image = Image.open(img_path).convert('L')
        image = np.array(image)
        height, width = image.shape[:2]

        new_image = np.zeros((height, width), dtype='uint8')

        kernel = np.array([[-1,-1,-1],
                            [-2,10,-2],
                            [-1,-1,-1]])
        
        for x in range(1,width-1):
            for y in range(1,height-1):
                region = image[y-1:y+2, x-1:x+2]
                flip_region = np.flip(region)
                result = np.multiply(flip_region,kernel)
                result_sum = np.sum(result)
                result_sum = min(255, max(0, result_sum))
                new_image[y,x] = result_sum

        return Image.fromarray(new_image)

    def crossCorrelation(img_path, img_path2):
        image1 = Image.open(img_path).convert('L')
        image2 = Image.open(img_path2).convert('L')

        img1_width, img1_height = image1.size
        img2_width, img2_height = image2.size

        result_matrix = np.zeros((img1_height - img2_height + 1, img1_width - img2_width + 1))

        for y in range(img1_height - img2_height + 1):
            for x in range(img1_width - img2_width + 1):
                region = image1.crop((x, y, x + img2_width, y + img2_height))
                region_array = np.array(region)
                result_matrix[y, x] = np.sum(region_array * np.array(image2))

        return result_matrix




      