% Siyah Beyaza Çevir
% image = imread('images/image.png');
% grayimage = rgb2gray(image);
% imshow(grayimage);

% Eşikleme
% img = imread('images/image.png');
% gray_img = rgb2gray(img);
% threshold = 128;
% binary_img = gray_img > threshold;
% imshow(binary_img);

% Parlaklık Ayarlama
% img = imread('images/image.png');
% brightness_value = 50;
% brighter_img = img + brightness_value;
% imshow(brighter_img);

% Histogram
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% histogram = imhist(gray_img);
% bar(histogram);

% Negatif Resim
% img = imread('images/image2.png');
% negative_img = 255 - img;
% imshow(negative_img);

% Gauss Alçak Geçiren Filtre
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% h = fspecial('gaussian', [3 3], 0.5);
% filtered_img = imfilter(gray_img, h);
% imshow(filtered_img);

% Mean Alçak Geçiren Filtre
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% h = fspecial('average', [3 3]);
% filtered_img = imfilter(img, h);
% imshow(filtered_img);

% Meadian Alçak Geçiren Filtre
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% filtered_img = medfilt2(gray_img, [3 3]);
% imshow(filtered_img);

% Kontrast Ayarlama
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% low_in = 0; %düşük yoğunluklu ucu
% high_in = 1; %yüksek yoğunluklu ucu
% low_out = 0.5; %düşük yoğunluklu ucu
% high_out = 0.75; % yüksek yoğunluklu ucu
% adjusted_img = imadjust(gray_img, [low_in high_in], [low_out high_out]);
% imshow(adjusted_img);

% Kontrast Germe
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% min_val = double(min(gray_img(:)));
% max_val = double(max(gray_img(:)));
% stretched_img = (gray_img - min_val) * ((255 - 0) / (max_val - min_val)) + 0;
% imshow(stretched_img, []);

% Histogram Eşitleme
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% eq_img = histeq(gray_img);
% imshow(eq_img);

% Laplas Filtresi
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% h = fspecial('laplacian');
% filtered_img = imfilter(gray_img, h);
% imshow(filtered_img, []);

% Sobel Filtresi
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% sobel_filter = fspecial('sobel');
% filtered_img = imfilter(gray_img, sobel_filter);
% imshow(filtered_img, []);

% Prewitt Filtresi
% img = imread('images/image2.png');
% gray_img = rgb2gray(img);
% prewitt_filter = fspecial('prewitt');
% filtered_img = imfilter(gray_img, prewitt_filter);
% imshow(filtered_img, []);

% Açılı Döndürme
% img = imread('images/image2.png');
% aci = 45;
% dondurulmus_img = imrotate(img, aci);
% imshow(dondurulmus_img);

% Ters Çevir
% img = imread('images/image2.png');
% aci = 180;
% dondurulmus_img = imrotate(img, aci);
% imshow(dondurulmus_img);

% Aynala
% img = imread('images/image2.png');
% aynalanmis_img = flip(img, 2);
% imshow(aynalanmis_img);

% Öteleme
% image = imread('images/image2.png');
% shift_x = 50;
% shift_y = 30;
% translation_matrix = [1, 0, shift_x; 0, 1, shift_y];
% translated_image = imtranslate(image, [shift_x, shift_y]);
% imshow(translated_image);

% Yakınlaştırma
% image = imread('images/image2.png');
% scale_factor = 2;
% resized_image = imresize(image, scale_factor);
% imshow(resized_image);

% Uzaklaştırma
% image = imread('images/image2.png');
% scale_factor = 0.5;
% resized_image = imresize(image, scale_factor);
% imshow(resized_image);

% Yayma
% image = imread('images/image7.png');
% se = strel('square', 3);
% dilated_image = imdilate(image, se);
% imshow(dilated_image);

% Aşındırma
% image = imread('images/image9.png');
% se = strel('square', 3);
% gray_img = rgb2gray(image);
% threshold = 128;
% binary_img = gray_img > threshold;
% eroded_image = imerode(binary_img, se);
% imshow(eroded_image);

% Açma
% image = imread('images/image9.png');
% se = strel('square', 3);
% gray_img = rgb2gray(image);
% threshold = 128;
% binary_img = gray_img > threshold;
% opened_image = imopen(binary_img, se);
% imshow(opened_image);

% Kapama
% image = imread('images/image9.png');
% se = strel('square', 3);
% gray_img = rgb2gray(image);
% threshold = 128;
% binary_img = gray_img > threshold;
% closed_image = imclose(binary_img, se);
% imshow(closed_image);

% Kenar Görüntüsü ile resim netleştirme
% input_image = imread('images/image11.png');
% k = 0.2;
% smooth_image = imfilter(input_image, fspecial('average', [9, 9]));
% diff_image = imsubtract(input_image, smooth_image);
% scaled_diff_image = immultiply(diff_image, k);
% output_image = imadd(input_image, scaled_diff_image);
% output_image = max(0, min(output_image, 255));
% imshow(output_image);

% Konvolüsyon ile Resim Netleştirme
% input_image = imread('images/image11.png');
% input_image_double = im2double(input_image);
% convolution_kernel = [0, -2, 0; -2, 11, -2; 0, -2, 0];
% output_image_double = imfilter(input_image_double, convolution_kernel, 'same');
% output_image = uint8(output_image_double * 255);
% imshow(output_image);

%Konvolüsyon
% input_image = imread('images/image2.png');
% gray_image = rgb2gray(input_image);
% kernel = [-1, -1, -1; -2, 10, -2; -1, -1, -1];
% result_image = conv2(double(gray_image), kernel, 'same');
% imshow(result_image, []);

% Korelasyon
% input_image = imread('images/image2.png');
% gray_image = rgb2gray(input_image);
% kernel = [-1, -1, -1; -2, 10, -2; -1, -1, -1];
% result_image = conv2(double(gray_image), rot90(kernel,2), 'same');
% imshow(result_image, []);




