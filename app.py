from flask import Flask, render_template, request
from load_model import *
from flask_socketio import SocketIO
import cv2

app = Flask(__name__)
socketio = SocketIO(app)
is_running = False


@app.route('/', methods=['GET'])
def Hello_World():
    return render_template('./index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (227, 227))
    img_flatten = img.flatten().reshape(1, -1)
    predict_text = model_predict(img_flatten)
    return render_template('index.html', prediction=predict_text)

if __name__ == '__main__':
    socketio.run(app, port=3000, debug=True)
