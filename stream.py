from flask import Flask, render_template, Response
import cv2
import numpy as np
import time

app = Flask(__name__)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Set width
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height

motion_detected = False
motion_threshold = 10000  # Adjust this value based on your environment and sensitivity
last_motion_time = time.time()
motion_notification_duration = 10  # Seconds

background_subtractor = cv2.createBackgroundSubtractorMOG2()

def detect_motion(frame):
    global motion_detected, last_motion_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fg_mask = background_subtractor.apply(gray)

    if motion_detected:
        if time.time() - last_motion_time > motion_notification_duration:
            print()  # Print a new line for new motion detection notification
            last_motion_time = time.time()

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > motion_threshold:
            motion_detected = True
            last_motion_time = time.time()
            print("Motion detected!")
            break

    return frame

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to read frame")
            break
        else:
            frame = detect_motion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
