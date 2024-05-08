from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()  # Read the frame from the webcam
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
            frame = buffer.tobytes()  # Convert the frame to bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield the frame in the correct format for streaming

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML template

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Return the streaming response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

