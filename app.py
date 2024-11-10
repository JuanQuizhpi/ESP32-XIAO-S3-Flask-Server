from flask import Flask, render_template, Response, stream_with_context, request, jsonify
from io import BytesIO

import cv2
import numpy as np
import requests
import time
import os

app = Flask(__name__)
# IP Address
_URL = 'http://192.168.18.53'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])

# Sustractor de fondo adaptativo
back_sub = cv2.createBackgroundSubtractorMOG2()

# Ruta donde se almacenan las imágenes locales
IMAGES_FOLDER = 'static/images/'
IMAGES = ['IM000001.jpg', 'IM000008.jpg', 'IM000013.jpg']  #

# Función para calcular FPS
def calculate_fps(prev_time):
    new_time = time.time()
    fps = 1 / (new_time - prev_time)
    return fps, new_time

def video_capture():
    prev_time = time.time()
    # Abrir la transmisión de video desde la URL
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                # Convertimos chunk a imagen
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                # Convertimos imagen a escala de grises
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                cv2.putText(gray,"Escala gris", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                #Imagen Escala de grises para detectar fondo
                fg_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                # Aplicar sustracción de fondo adaptativa
                fg_mask = back_sub.apply(fg_gray)
                cv2.putText(fg_mask,"Deteccion Movimiento", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                #Mejoramiento de luminosidad ecualizacion histograma
                ecuHisto_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                equ = cv2.equalizeHist(ecuHisto_gray)
                cv2.putText(equ,"Ecualizacion Histograma", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                #Uso de CLAHE mejoramiento luminosidad
                clahe_gray= cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(clahe_gray)
                cv2.putText(clahe_img,"CLAHE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Filtro de aumento de brillo (cambio de gama)
                gamma_gray= cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                gamma = 1.5  # Valor de gama para incrementar el brillo
                lookUpTable = np.empty((1, 256), np.uint8)
                for i in range(256):
                    lookUpTable[0][i] = np.clip((i * gamma), 0, 255)
                gamma_image = cv2.LUT(gamma_gray, lookUpTable)
                cv2.putText(gamma_image,"Filtro Gamma", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                #Filtro rudio sal y pimienta
                salpe_gray=cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                N = 537
                height, width = salpe_gray.shape
                noise = np.full((height, width), 0, dtype=np.uint8)
                random_positions = (np.random.randint(0, height, N), np.random.randint(0, width, N))
                noise[random_positions[0], random_positions[1]] = 255
                noise_image = cv2.bitwise_or(salpe_gray, noise)
                cv2.putText(noise_image,"Sal y pimienta", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                grayFiltrosRuido=cv2.bitwise_or(salpe_gray, noise)


                #Tamaño Kernel
                kernel_size = 5
                #Aplicar filtro de mediana
                median_filtered = cv2.medianBlur(grayFiltrosRuido, kernel_size)
                #median_filtered.copyTo(filtro_mediana)
                cv2.putText(median_filtered,"Filtro de mediana", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Aplicar filtro de Blur (promedio)
                average_filtered = cv2.blur(grayFiltrosRuido, (kernel_size, kernel_size))
                #average_filtered.copyTo(filtro_blur)
                cv2.putText(average_filtered ,"Filtro de Blur", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Aplicar filtro Gaussiano
                gaussian_filtered = cv2.GaussianBlur(grayFiltrosRuido, (kernel_size, kernel_size), 0)
                #gaussian_filtered.copyTo(filtro_gausiano)
                cv2.putText(gaussian_filtered,"Filtro Gaussiano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                #Filtro Gray
                kernel_size1 = 5
                grayParaFiltros= cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                median_filtered1 = cv2.medianBlur(grayParaFiltros, kernel_size1)

                ## Aplicar filtro Sobel
                sobel_x = cv2.Sobel(grayFiltrosRuido, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(grayFiltrosRuido, cv2.CV_64F, 0, 1, ksize=3)

                # Magnitud del gradiente de Sobel (combinación de ambos filtros)
                sobel_combined = cv2.magnitude(sobel_x, sobel_y)
                sobel_combined = np.uint8(np.absolute(sobel_combined))  # Convertir a valores enteros
                cv2.putText(sobel_combined,"Sobel Ruido", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                ##Filtro Sobel Reduccion Ruido
                sobel_x1 = cv2.Sobel(median_filtered1, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y1 = cv2.Sobel(median_filtered1, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined1 = cv2.magnitude(sobel_x1, sobel_y1)
                sobel_combined1 = np.uint8(np.absolute(sobel_combined1))  # Convertir a valores enteros
                cv2.putText(sobel_combined1,"Sobel Mediana", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


                # Aplicar filtro Canny
                kernel_size2 = 5
                grayParaFiltros1= cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                median_filtered2 = cv2.medianBlur(grayParaFiltros1, kernel_size2)

                filtro_canny_ruido = cv2.Canny(grayFiltrosRuido, 100, 200)
                cv2.putText(filtro_canny_ruido,"Canny Ruido", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                filtro_canny_ruido2 = cv2.Canny(median_filtered2, 70, 180)
                cv2.putText(filtro_canny_ruido2,"Canny Mediana", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                # Aplicar Laplaciano de Gaussiano (LoG)
                kernel_size3 = 5
                grayParaFiltros2= cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                median_filtered3 = cv2.medianBlur(grayParaFiltros2, kernel_size3)

                log_filtered = cv2.GaussianBlur(median_filtered3, (kernel_size, kernel_size), 0)  # Suavizar la imagen
                log_filtered = cv2.Laplacian(log_filtered, cv2.CV_64F)  # Aplicar el operador Laplaciano
                log_filtered = cv2.convertScaleAbs(log_filtered)  # Convertir a escala de 8 bits

                # Mostrar la imagen resultante del LoG
                cv2.putText(log_filtered, "Laplaciano de Gaussiano", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


                # Calcular y mostrar los FPS
                fps, prev_time = calculate_fps(prev_time)
                cv2.putText(cv_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


                # Crear total_image con 5 espacios de ancho y 3 filas
                height, width = gray.shape
                total_image = np.zeros((height * 5, width *3, 3), dtype=np.uint8)

                # Fila 1
                total_image[:height, :width] = cv_img  # Imagen original con FPS
                total_image[:height, width:width*2] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  
                total_image[:height, width*2:width*3] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)  

                # Fila 2 
                total_image[height:height*2, :width] = cv2.cvtColor(gamma_image, cv2.COLOR_GRAY2BGR)  
                total_image[height:height*2, width:width*2] = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)  
                total_image[height:height*2, width*2:width*3] = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)  

                # Fila 3 
                total_image[height*2:height*3, :width] = cv2.cvtColor(noise_image, cv2.COLOR_GRAY2BGR)  
                total_image[height*2:height*3, width:width*2] = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)  
                total_image[height*2:height*3, width*2:width*3] = cv2.cvtColor(gaussian_filtered, cv2.COLOR_GRAY2BGR)  

                # Fila 4 
                total_image[height*3:height*4, :width] = cv2.cvtColor(average_filtered, cv2.COLOR_GRAY2BGR)  
                total_image[height*3:height*4, width:width*2] = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)  
                total_image[height*3:height*4, width*2:width*3] = cv2.cvtColor(sobel_combined1, cv2.COLOR_GRAY2BGR)

                # Fila 5 
                total_image[height*4:height*5, :width] = cv2.cvtColor(filtro_canny_ruido, cv2.COLOR_GRAY2BGR)  
                total_image[height*4:height*5, width:width*2] = cv2.cvtColor(filtro_canny_ruido2, cv2.COLOR_GRAY2BGR)  
                total_image[height*4:height*5, width*2:width*3] = cv2.cvtColor(log_filtered, cv2.COLOR_GRAY2BGR)

                # Codificar la imagen para transmitirla
                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                # Enviar el frame
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue

def image_capture(image_path1, image_path2,image_path3):
    #Cargamos las images a procesar
    image_1 = cv2.imread(image_path1, cv2.COLOR_GRAY2BGR)
    image_2 = cv2.imread(image_path2, cv2.COLOR_GRAY2BGR)
    image_3 = cv2.imread(image_path3, cv2.COLOR_GRAY2BGR)

    #Convertimos a escala de grises las imagenes
    gray1 =cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    cv2.putText(gray1, "IM000001", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    gray2 =cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    cv2.putText(gray2, "IM000008", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    gray3 =cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    cv2.putText(gray3, "IM000013", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #Erosion 3 kernels
    kernel1 = np.ones((17, 17), np.uint8)
    kernel2 = np.ones((27, 27), np.uint8)
    kernel3 = np.ones((37, 37), np.uint8)

    # Erosión IM000001
    grayErosion1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    erosion1_17 = cv2.erode(grayErosion1, kernel1, iterations=1)
    cv2.putText(erosion1_17, "IM000001 Erosion 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    erosion1_27 = cv2.erode(grayErosion1, kernel2, iterations=1)
    cv2.putText(erosion1_27, "IM000001 Erosion 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    erosion1_37 = cv2.erode(grayErosion1, kernel3, iterations=1)
    cv2.putText(erosion1_37, "IM000001 Erosion 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Erosión IM000008
    grayErosion2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    erosion2_17 = cv2.erode(grayErosion2, kernel1, iterations=1)
    cv2.putText(erosion2_17, "IM000008 Erosion 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    erosion2_27 = cv2.erode(grayErosion2, kernel2, iterations=1)
    cv2.putText(erosion2_27, "IM000008 Erosion 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    erosion2_37 = cv2.erode(grayErosion2, kernel3, iterations=1)
    cv2.putText(erosion2_37, "IM000008 Erosion 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Erosión IM000013
    grayErosion3 = cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    erosion3_17 = cv2.erode(grayErosion3, kernel1, iterations=1)
    cv2.putText(erosion3_17, "IM000013 Erosion 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    erosion3_27 = cv2.erode(grayErosion3, kernel2, iterations=1)
    cv2.putText(erosion3_27, "IM000013 Erosion 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    erosion3_37 = cv2.erode(grayErosion3, kernel3, iterations=1)
    cv2.putText(erosion3_37, "IM000013 Erosion 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Dilatación IM000001
    grayDilatacion1 = cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    dilatacion1_17 = cv2.dilate(grayDilatacion1, kernel1, iterations=1)
    cv2.putText(dilatacion1_17, "IM000001 Dilatacion 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dilatacion1_27 = cv2.dilate(grayDilatacion1, kernel2, iterations=1)
    cv2.putText(dilatacion1_27, "IM000001 Dilatacion 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dilatacion1_37 = cv2.dilate(grayDilatacion1, kernel3, iterations=1)
    cv2.putText(dilatacion1_37, "IM000001 Dilatacion 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Dilatación IM000008
    grayDilatacion2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    dilatacion2_17 = cv2.dilate(grayDilatacion2, kernel1, iterations=1)
    cv2.putText(dilatacion2_17, "IM000008 Dilatacion 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dilatacion2_27 = cv2.dilate(grayDilatacion2, kernel2, iterations=1)
    cv2.putText(dilatacion2_27, "IM000008 Dilatacion 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dilatacion2_37 = cv2.dilate(grayDilatacion2, kernel3, iterations=1)
    cv2.putText(dilatacion2_37, "IM000008 Dilatacion 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Dilatación IM000013
    grayDilatacion3 = cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    dilatacion3_17 = cv2.dilate(grayDilatacion3, kernel1, iterations=1)
    cv2.putText(dilatacion3_17, "IM000013 Dilatacion 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dilatacion3_27 = cv2.dilate(grayDilatacion3, kernel2, iterations=1)
    cv2.putText(dilatacion3_27, "IM000013 Dilatacion 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dilatacion3_37 = cv2.dilate(grayDilatacion3, kernel3, iterations=1)
    cv2.putText(dilatacion3_37, "IM000013 Dilatacion 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    #Using morphological transforms to enhance the contrast of medical images

    #IM000001 kernel 17x17
    grayIMG1 =cv2.cvtColor(image_1,cv2.COLOR_BGR2GRAY)
    tophat1_17 = cv2.morphologyEx(grayIMG1, cv2.MORPH_TOPHAT, kernel1)
    blackhat1_17 = cv2.morphologyEx(grayIMG1, cv2.MORPH_BLACKHAT, kernel1)
    result1_17= cv2.add(grayIMG1, cv2.subtract(tophat1_17, blackhat1_17 ))
    cv2.putText(result1_17, "IM000001 Imagen Original + (Top Hat - Black Hat) 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat1_17, "IM000001 Top Hat 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat1_17, "IM000001 Black Hat 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000001 kernel 27x27
    tophat1_27 = cv2.morphologyEx(grayIMG1, cv2.MORPH_TOPHAT, kernel2)
    blackhat1_27 = cv2.morphologyEx(grayIMG1, cv2.MORPH_BLACKHAT, kernel2)
    result1_27= cv2.add(grayIMG1, cv2.subtract(tophat1_27, blackhat1_27 ))
    cv2.putText(result1_27, "IM000001 Imagen Original + (Top Hat - Black Hat) 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat1_27, "IM000001 Top Hat 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat1_27, "IM000001 Black Hat 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000001 kernel 37x37
    tophat1_37 = cv2.morphologyEx(grayIMG1, cv2.MORPH_TOPHAT, kernel3)
    blackhat1_37 = cv2.morphologyEx(grayIMG1, cv2.MORPH_BLACKHAT, kernel3)
    result1_37= cv2.add(grayIMG1, cv2.subtract(tophat1_37, blackhat1_37 ))
    cv2.putText(result1_37, "IM000001 Imagen Original + (Top Hat - Black Hat) 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat1_37, "IM000001 Top Hat 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat1_37, "IM000001 Black Hat 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000008 kernel 17x17
    grayIMG2 =cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    tophat2_17 = cv2.morphologyEx(grayIMG2, cv2.MORPH_TOPHAT, kernel1)
    blackhat2_17 = cv2.morphologyEx(grayIMG2, cv2.MORPH_BLACKHAT, kernel1)
    result2_17= cv2.add(grayIMG2, cv2.subtract(tophat2_17, blackhat2_17 ))
    cv2.putText(result2_17, "IM000008 Imagen Original + (Top Hat - Black Hat) 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat2_17, "IM000008 Top Hat 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat2_17, "IM000008 Black Hat 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000008 kernel 27x27
    grayIMG2 =cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    tophat2_27 = cv2.morphologyEx(grayIMG2, cv2.MORPH_TOPHAT, kernel2)
    blackhat2_27 = cv2.morphologyEx(grayIMG2, cv2.MORPH_BLACKHAT, kernel2)
    result2_27= cv2.add(grayIMG2, cv2.subtract(tophat2_27, blackhat2_27 ))
    cv2.putText(result2_27, "IM000008 Imagen Original + (Top Hat - Black Hat) 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat2_27, "IM000008 Top Hat 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat2_27, "IM000008 Black Hat 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000008 kernel 37x37
    grayIMG2 =cv2.cvtColor(image_2,cv2.COLOR_BGR2GRAY)
    tophat2_37 = cv2.morphologyEx(grayIMG2, cv2.MORPH_TOPHAT, kernel3)
    blackhat2_37 = cv2.morphologyEx(grayIMG2, cv2.MORPH_BLACKHAT, kernel3)
    result2_37= cv2.add(grayIMG2, cv2.subtract(tophat2_37, blackhat2_37 ))
    cv2.putText(result2_37, "IM000008 Imagen Original + (Top Hat - Black Hat) 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat2_37, "IM000008 Top Hat 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat2_37, "IM000008 Black Hat 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    #IM000013 kernel 17x17
    grayIMG3 =cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    tophat3_17 = cv2.morphologyEx(grayIMG3, cv2.MORPH_TOPHAT, kernel1)
    blackhat3_17 = cv2.morphologyEx(grayIMG3, cv2.MORPH_BLACKHAT, kernel1)
    result3_17= cv2.add(grayIMG3, cv2.subtract(tophat3_17, blackhat3_17 ))
    cv2.putText(result3_17, "IM000013 Imagen Original + (Top Hat - Black Hat) 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat3_17, "IM000013 Top Hat 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat3_17, "IM000013 Black Hat 17x17", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000013 kernel 27x27
    grayIMG3 =cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    tophat3_27 = cv2.morphologyEx(grayIMG3, cv2.MORPH_TOPHAT, kernel2)
    blackhat3_27 = cv2.morphologyEx(grayIMG3, cv2.MORPH_BLACKHAT, kernel2)
    result3_27= cv2.add(grayIMG3, cv2.subtract(tophat3_27, blackhat3_27 ))
    cv2.putText(result3_27, "IM000013 Imagen Original + (Top Hat - Black Hat) 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat3_27, "IM000013 Top Hat 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat3_27, "IM000013 Black Hat 27x27", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    #IM000013 kernel 37x37
    grayIMG3 =cv2.cvtColor(image_3,cv2.COLOR_BGR2GRAY)
    tophat3_37 = cv2.morphologyEx(grayIMG3, cv2.MORPH_TOPHAT, kernel3)
    blackhat3_37 = cv2.morphologyEx(grayIMG3, cv2.MORPH_BLACKHAT, kernel3)
    result3_37= cv2.add(grayIMG3, cv2.subtract(tophat3_37, blackhat3_37 ))
    cv2.putText(result3_37, "IM000013 Imagen Original + (Top Hat - Black Hat) 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.putText(tophat3_37, "IM000013 Top Hat 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(blackhat3_37, "IM000013 Black Hat 37x37", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # Crear un Imagem de 3 columnas por 16 filas
    height, width = gray1.shape
    total_image = np.zeros((height*16, width * 3, 3), dtype=np.uint8)

    #Fila 1
    total_image[:height, :width] = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)  
    total_image[:height, width:width*2] = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)  
    total_image[:height, width*2:width*3] = cv2.cvtColor(gray3, cv2.COLOR_GRAY2BGR)
    
    #Fila 2
    total_image[height:height*2, :width] = cv2.cvtColor(erosion1_17, cv2.COLOR_GRAY2BGR)  
    total_image[height:height*2, width:width*2] = cv2.cvtColor(erosion1_27, cv2.COLOR_GRAY2BGR)  
    total_image[height:height*2, width*2:width*3] = cv2.cvtColor(erosion1_37, cv2.COLOR_GRAY2BGR)

    # Fila 3 
    total_image[height*2:height*3, :width] = cv2.cvtColor(erosion2_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*2:height*3, width:width*2] = cv2.cvtColor(erosion2_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*2:height*3, width*2:width*3] = cv2.cvtColor(erosion2_37, cv2.COLOR_GRAY2BGR) 

    # Fila 4 
    total_image[height*3:height*4, :width] = cv2.cvtColor(erosion3_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*3:height*4, width:width*2] = cv2.cvtColor(erosion3_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*3:height*4, width*2:width*3] = cv2.cvtColor(erosion3_37, cv2.COLOR_GRAY2BGR)

    # Fila 5 
    total_image[height*4:height*5, :width] = cv2.cvtColor(dilatacion1_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*4:height*5, width:width*2] = cv2.cvtColor(dilatacion1_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*4:height*5, width*2:width*3] = cv2.cvtColor(dilatacion1_37, cv2.COLOR_GRAY2BGR)

    # Fila 6 
    total_image[height*5:height*6, :width] = cv2.cvtColor(dilatacion2_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*5:height*6, width:width*2] = cv2.cvtColor(dilatacion2_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*5:height*6, width*2:width*3] = cv2.cvtColor(dilatacion2_37, cv2.COLOR_GRAY2BGR)

    # Fila 7 
    total_image[height*6:height*7, :width] = cv2.cvtColor(dilatacion3_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*6:height*7, width:width*2] = cv2.cvtColor(dilatacion3_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*6:height*7, width*2:width*3] = cv2.cvtColor(dilatacion3_37, cv2.COLOR_GRAY2BGR)

    # Fila 8 
    total_image[height*7:height*8, :width] = cv2.cvtColor(tophat1_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*7:height*8, width:width*2] = cv2.cvtColor(blackhat1_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*7:height*8, width*2:width*3] = cv2.cvtColor(result1_17, cv2.COLOR_GRAY2BGR)

    # Fila 9 
    total_image[height*8:height*9, :width] = cv2.cvtColor(tophat1_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*8:height*9, width:width*2] = cv2.cvtColor(blackhat1_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*8:height*9, width*2:width*3] = cv2.cvtColor(result1_27, cv2.COLOR_GRAY2BGR)

    # Fila 10 
    total_image[height*9:height*10, :width] = cv2.cvtColor(tophat1_37, cv2.COLOR_GRAY2BGR)  
    total_image[height*9:height*10, width:width*2] = cv2.cvtColor(blackhat1_37, cv2.COLOR_GRAY2BGR)  
    total_image[height*9:height*10, width*2:width*3] = cv2.cvtColor(result1_37, cv2.COLOR_GRAY2BGR)
    
    # Fila 11 
    total_image[height*10:height*11, :width] = cv2.cvtColor(tophat2_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*10:height*11, width:width*2] = cv2.cvtColor(blackhat2_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*10:height*11, width*2:width*3] = cv2.cvtColor(result2_17, cv2.COLOR_GRAY2BGR)

    # Fila 12 
    total_image[height*11:height*12, :width] = cv2.cvtColor(tophat2_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*11:height*12, width:width*2] = cv2.cvtColor(blackhat2_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*11:height*12, width*2:width*3] = cv2.cvtColor(result2_27, cv2.COLOR_GRAY2BGR)

    # Fila 13 
    total_image[height*12:height*13, :width] = cv2.cvtColor(tophat2_37, cv2.COLOR_GRAY2BGR)  
    total_image[height*12:height*13, width:width*2] = cv2.cvtColor(blackhat2_37, cv2.COLOR_GRAY2BGR)  
    total_image[height*12:height*13, width*2:width*3] = cv2.cvtColor(result2_37, cv2.COLOR_GRAY2BGR)

    # Fila 14 
    total_image[height*13:height*14, :width] = cv2.cvtColor(tophat3_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*13:height*14, width:width*2] = cv2.cvtColor(blackhat3_17, cv2.COLOR_GRAY2BGR)  
    total_image[height*13:height*14, width*2:width*3] = cv2.cvtColor(result3_17, cv2.COLOR_GRAY2BGR)

    # Fila 15 
    total_image[height*14:height*15, :width] = cv2.cvtColor(tophat3_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*14:height*15, width:width*2] = cv2.cvtColor(blackhat3_27, cv2.COLOR_GRAY2BGR)  
    total_image[height*14:height*15, width*2:width*3] = cv2.cvtColor(result3_27, cv2.COLOR_GRAY2BGR)

    # Fila 16 
    total_image[height*15:height*16, :width] = cv2.cvtColor(tophat3_37, cv2.COLOR_GRAY2BGR)  
    total_image[height*15:height*16, width:width*2] = cv2.cvtColor(blackhat3_37, cv2.COLOR_GRAY2BGR)  
    total_image[height*15:height*16, width*2:width*3] = cv2.cvtColor(result3_37, cv2.COLOR_GRAY2BGR)


    # Codificar la imagen para transmitirla
    (flag, encodedImage) = cv2.imencode(".jpg", total_image)
    if not flag:
        return None

    # Devolver la imagen codificada para la transmisión en vivo
    return encodedImage

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/video_feed')
def video_feed():
    # Suponiendo que tenemos imágenes locales, procesarlas y mostrar el mosaico
    img_path1 = os.path.join(IMAGES_FOLDER, 'IM000001.jpg')
    img_path2 = os.path.join(IMAGES_FOLDER, 'IM000008.jpg')
    img_path3 = os.path.join(IMAGES_FOLDER, 'IM000013.jpg')
    encodedImage = image_capture(img_path1,img_path2,img_path3)
    if encodedImage is not None:
        return Response(encodedImage.tobytes(), mimetype='image/jpeg')
    else:
        return "Error al procesar la imagen"

if __name__ == "__main__":
    app.run(debug=False)
