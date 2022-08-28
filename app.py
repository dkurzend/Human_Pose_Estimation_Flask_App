from flask import Flask, render_template, Response, request,redirect, url_for, jsonify
from flask import jsonify
import time
import cv2
import torch
from PIL import Image
import sys
import matplotlib.pyplot as plt
import utils.model
from models.stacked_hourglass import PoseNet, SGSC_PoseNet
from preprocessing.img import prepare_image, opencv_to_PIL, PIL_to_opencv
from preprocessing.keypoints import get_keyppoints, post_process_keypoints, draw_cv2_keypoints, draw_keypoints, upScaleKeypoints


app = Flask(__name__)

net_SGSC = utils.model.load_model(arch='SGSC')
net_SHG = utils.model.load_model(arch='SHG')
net = net_SGSC

has_gpu = torch.cuda.is_available()
device = torch.device('cpu')

cap = cv2.VideoCapture(0)
mode = 0
threshold = 0.2
use_gpu = False
current_model = 'SGSC'
seconds = None


def update_model():
    if current_model == 'SHG':
        return net_SHG
    else:
        return net_SGSC

def update_device(net):
    if use_gpu:
        if not next(net.parameters()).is_cuda:
            net = net.cuda()
    if not use_gpu:
        if next(net.parameters()).is_cuda:
            net = net.cpu()
    return net


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html', has_gpu=int(has_gpu))


def gen():
    """Video streaming generator function."""
    global net, seconds

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:

            input, c, s = prepare_image(img) # input is tensor of shape [1, C x H x W]

            if next(net.parameters()).is_cuda:
                input = input.cuda()

            with torch.no_grad():
                start_prediction = time.time()
                heatmaps = net(input) # returns shape  (1, 4, 16, 64, 64) = (bs, hg_modules, 16 kp, height, width)
                end_prediction = time.time()
            seconds = end_prediction - start_prediction
            if heatmaps.is_cuda:
                heatmaps = heatmaps.cpu()
            # get keypoints from predicted heatmaps as (x, y) = (width, height)
            pred_keypoints = get_keyppoints(heatmaps[:, -1], threshold) # returns (batch_size, 16, 2)
            input_res = 256
            keypoints = post_process_keypoints(pred_keypoints, input, c, s, input_res)
            img_with_keypoints = draw_cv2_keypoints(img, keypoints[0], radius=6, mode=mode)

            net = update_model()
            net = update_device(net)

            frame = cv2.imencode('.jpg', img_with_keypoints)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else:
            break


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/mode", methods=['GET', 'POST'])
def set_mode():
    global mode
    if request.method == "POST":
        mode = request.json['data']
    return render_template('index.html')


@app.route("/threshold", methods=['GET', 'POST'])
def set_threshold():
    global threshold
    if request.method == "POST":
        threshold = request.json['data']
    return render_template('index.html')



@app.route("/setCuda", methods=['GET', 'POST'])
def set_cuda():
    global use_gpu
    if request.method == "POST":
        use_gpu = request.json['data']
    return render_template('index.html')


@app.route("/model", methods=['GET', 'POST'])
def set_model():
    global current_model
    if request.method == "POST":
        current_model = request.json['data']
    return render_template('index.html')


@app.route("/performance", methods=['GET'])
def get_performance():
    global seconds
    return jsonify({'speed': seconds})





if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
