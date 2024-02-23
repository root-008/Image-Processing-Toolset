import cv2
import os
from matplotlib import pyplot as plt
import numpy as np

img_path = 'images/image2.png'


#Siyah Beyaz Goruntu
def BlackandWhite(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Siyah Beyaz Görüntü', gray_image)
    cv2.waitKey(0)
 
def esikleme(img_path):
    img = cv2.imread(img_path, 0)
    threshold = 128
    ret, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('Binary Image', binary_img)
    cv2.waitKey(0)

def parlaklikAyarlama(img_path):
    img = cv2.imread(img_path, 1)
    brightness_value = 50
    brighter_img = cv2.add(img, brightness_value)
    cv2.imshow('Brighter Image', brighter_img)
    cv2.waitKey(0)

def histogram(img_path):
    img = cv2.imread(img_path,0)
    histogram = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.figure()
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()

def negativeImage(img_path):
    img = cv2.imread(img_path, 1)
    negative_img = 255 - img
    cv2.imshow('Negative Image', negative_img)
    cv2.waitKey(0)

def gaussAlcakGecirenFiltre(img_path):
    img = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow('Gaussian Blurred', blur)
    cv2.waitKey(0)

def meanAlcakGecirenFiltre(img_path):
    img = cv2.imread(img_path, 0)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img,-1,kernel)
    cv2.imshow('Mean filter', dst)
    cv2.waitKey(0)

def medianAlcakGecirenFiltre(img_path):
    img = cv2.imread(img_path, 0)
    median = cv2.medianBlur(img,5)
    cv2.imshow('Median filter', median)
    cv2.waitKey(0)

def kontrastAyarlama(img_path):
    img = cv2.imread(img_path, 0)
    alpha = 1.5 # Kontrast
    beta = 50   # Parlaklığ
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    cv2.imshow('Adjusted Image', adjusted_img)
    cv2.waitKey(0)

def histogramEsitleme(img_path):
    img = cv2.imread(img_path, 0)
    eq_img = cv2.equalizeHist(img)
    cv2.imshow('Equalized Image', eq_img)
    cv2.waitKey(0)

def laplasFiltre(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    cv2.imshow('Laplacian Filter', laplacian)
    cv2.waitKey(0)

def sobelFiltre(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_result = cv2.magnitude(sobel_x, sobel_y)
    sobel_result = np.uint8(np.absolute(sobel_result))
    cv2.imshow('Sobel Result', sobel_result)
    cv2.waitKey(0)

def prewittFiltre(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    prewitt_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=1)
    prewitt_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=1)
    prewitt_result = cv2.magnitude(prewitt_x, prewitt_y)
    prewitt_result = np.uint8(np.absolute(prewitt_result))
    cv2.imshow('Prewitt Result', prewitt_result)
    cv2.waitKey(0)

def aciliDondurme(img_path):
    image = cv2.imread(img_path)
    angle = 45
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)

def goruntuTersCevirme(img_path):
    image = cv2.imread(img_path)
    angle = 180
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)

def goruntuAynalama(img_path):
    image = cv2.imread(img_path)
    mirrored_image = cv2.flip(image, 1)
    cv2.imshow('Mirrored Image', mirrored_image)
    cv2.waitKey(0)

def goruntuOteleme(img_path):
    image = cv2.imread(img_path)
    shift_x = 50
    shift_y = 30
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    cv2.imshow('Translated Image', translated_image)
    cv2.waitKey(0)

def yakinlastirma(img_path):
    image = cv2.imread(img_path)
    scale_factor = 2.0
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('resized_image', resized_image)
    cv2.waitKey(0)

def uzaklastirma(img_path):
    image = cv2.imread(img_path)
    scale_factor = 0.5
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('resized_image', resized_image)
    cv2.waitKey(0)

def yayma(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=1)
    cv2.imshow('Dilated Image', dilated_image)
    cv2.waitKey(0)

def asindirma(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    threshold = 128
    ret, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    eroded_image = cv2.erode(binary_img, kernel, iterations=1)
    cv2.imshow('Eroded Image', eroded_image)
    cv2.waitKey(0)

def acma(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    threshold = 128
    ret, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    opened_image = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    cv2.imshow('Opened Image', opened_image)
    cv2.waitKey(0)

def kapama(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    threshold = 128
    ret, binary_img = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('Closed Image', closed_image)
    cv2.waitKey(0)
    
def kenarGoruntusuIleResmiNetlestirme(img_path,k):
    input_image = cv2.imread(img_path)
    smooth_image = cv2.blur(input_image, (9, 9))
    diff_image = cv2.subtract(input_image, smooth_image)
    scaled_diff_image = cv2.multiply(diff_image, k)
    output_image = cv2.add(input_image, scaled_diff_image)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    cv2.imshow('output_image', output_image)
    cv2.waitKey(0)

def konvolusyonIleNetlestirme(img_path):
    input_image = cv2.imread(img_path)   
    input_image_double = input_image.astype(np.float64) / 255.0
    convolution_kernel = np.array([[0, -2, 0], [-2, 11, -2], [0, -2, 0]])
    output_image_double = cv2.filter2D(input_image_double, -1, convolution_kernel, borderType=cv2.BORDER_REPLICATE)
    output_image = np.clip(output_image_double * 255, 0, 255).astype(np.uint8)
    cv2.imshow('Netleştirilmiş Görüntü', output_image)
    cv2.waitKey(0)

def konvolusyon(img_path):
    input_image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-1, -1, -1], [-2, 10, -2], [-1, -1, -1]])
    result_image = cv2.filter2D(gray_image, -1, kernel)
    cv2.imshow('', result_image)
    cv2.waitKey(0)

def korelasyon(img_path):
    input_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.array([[-1, -1, -1], [-2, 10, -2], [-1, -1, -1]], dtype=np.float32)
    result_image = cv2.filter2D(input_image, -1, kernel)
    cv2.imshow('',result_image)
    cv2.waitKey(0)



#BlackandWhite(img_path)
#esikleme(img_path)
#parlaklikAyarlama(img_path)
#histogram(img_path)
#negativeImage(img_path)
#gaussAlcakGecirenFiltre(img_path)
#meanAlcakGecirenFiltre(img_path)
#medianAlcakGecirenFiltre(img_path)
#kontrastAyarlama(img_path)
#histogramEsitleme(img_path)
#sobelFiltre(img_path)
korelasyon(img_path)
