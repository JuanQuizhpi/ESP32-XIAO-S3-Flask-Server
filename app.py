from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO
import cv2
import numpy as np
import requests
import time

app = Flask(__name__)
# IP Address
_URL = 'http://192.168.18.53'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])

# Crear el sustractor de fondo adaptativo
back_sub = cv2.createBackgroundSubtractorMOG2()

# Función para calcular FPS
def calculate_fps(prev_time):
    new_time = time.time()
    fps = 1 / (new_time - prev_time)
    return fps, new_time

# 1. Método para capturar el video
def video_capture():
    res = requests.get(stream_url, stream=True)
    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                # Convertimos chunk a imagen
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                
                # Retornamos el frame capturado
                yield cv_img

            except Exception as e:
                print(f"Error capturando el frame: {e}")
                continue

# 2. Método para procesar el frame
def process_frame(cv_img, prev_time):
    # Convertimos la imagen a escala de grises
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar sustracción de fondo adaptativa
    fg_mask = back_sub.apply(gray)

    # Calcular y mostrar los FPS
    fps, prev_time = calculate_fps(prev_time)
    cv2.putText(cv_img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Crear total_image con 3 espacios de ancho
    height, width = gray.shape
    total_image = np.zeros((height, width * 3, 3), dtype=np.uint8)
    
    # Colocar las diferentes vistas en `total_image`
    total_image[:, :width] = cv_img  # Imagen original con FPS
    total_image[:, width:width*2] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Escala de grises
    total_image[:, width*2:] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)  # Sustracción de fondo

    # Retornar la imagen procesada
    return total_image, prev_time

# 3. Método para el flujo de video completo (captura + procesamiento)
def video_stream():
    prev_time = time.time()
    for frame in video_capture():
        # Procesamos cada frame capturado
        processed_frame, prev_time = process_frame(frame, prev_time)

        # Codificar la imagen procesada para transmitirla
        (flag, encodedImage) = cv2.imencode(".jpg", processed_frame)
        if not flag:
            continue

        # Enviar el frame procesado
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_stream")
def stream_video():
    return Response(video_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)
