from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model/best_model.h5')
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Webcam capture
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Preprocessing for prediction
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            img_array = img_to_array(resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Predict
            prediction = model.predict(img_array)
            label = class_names[np.argmax(prediction)].upper()  # Teks jadi huruf besar

            # Styling teks
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            position = (10, 50)

            # Shadow hitam
            cv2.putText(frame, label, (position[0] + 2, position[1] + 2),
                        font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)

            # Teks utama putih
            cv2.putText(frame, label, position,
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Encode ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
