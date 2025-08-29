# run_script.py (Backend)
import numpy as np
import cv2
import os
from flask import Flask, render_template, Response, request

app = Flask(__name__)
camera = None


# List of glasses images for the backend
BACKEND_GLASSES_FOLDER = 'glasses'
BACKEND_GLASSES_FILES = [os.path.join(BACKEND_GLASSES_FOLDER, f) for f in os.listdir(BACKEND_GLASSES_FOLDER) if f.endswith('.png')]

# List of paths for the frontend
FRONTEND_GLASSES_PATHS = [f'glasses/{f}' for f in os.listdir(BACKEND_GLASSES_FOLDER) if f.endswith('.png')]

current_glasses_index = 0
current_glasses_img = None

# Pre-trained face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Helper functions 
def grayscale_from_rgb(rgb_image):
    return np.dot(rgb_image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)


def resize_bilinear(src_image, new_width, new_height):
    h_src, w_src, c_src = src_image.shape
    new_image = np.zeros((new_height, new_width, c_src), dtype=np.uint8)
    w_ratio, h_ratio = (w_src - 1) / (new_width - 1), (h_src - 1) / (new_height - 1)
    for i in range(new_height):
        for j in range(new_width):
            x, y = j * w_ratio, i * h_ratio
            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, w_src - 1), min(y1 + 1, h_src - 1)
            dx, dy = x - x1, y - y1
            q11, q12 = src_image[y1, x1], src_image[y2, x1]
            q21, q22 = src_image[y1, x2], src_image[y2, x2]
            interpolated_pixel = q11 * (1 - dx) * (1 - dy) + q21 * dx * (1 - dy) + q12 * (1 - dx) * dy + q22 * dx * dy
            new_image[i, j] = interpolated_pixel.astype(np.uint8)
    return new_image


def overlay_with_alpha(background, foreground, x_offset, y_offset):
    h_fg, w_fg, c_fg = foreground.shape
    h_bg, w_bg, c_bg = background.shape
    x1, y1 = max(x_offset, 0), max(y_offset, 0)
    x2, y2 = min(x_offset + w_fg, w_bg), min(y_offset + h_fg, h_bg)
    bg_roi = background[y1:y2, x1:x2]
    fg_roi = foreground[y1 - y_offset: y2 - y_offset, x1 - x_offset: x2 - x_offset]
    if fg_roi.size > 0 and fg_roi.shape[2] == 4:
        alpha_channel = fg_roi[:, :, 3] / 255.0
        alpha_channel_3d = np.stack([alpha_channel] * 3, axis=-1)
        bg_roi[...] = bg_roi * (1 - alpha_channel_3d) + fg_roi[:, :, :3] * alpha_channel_3d
    return background


def generate_frames():
    global current_glasses_img, camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open webcam.")
            return

    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        processed_frame = frame.copy()

        # Face detection and overlay logic
        gray_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            if current_glasses_img is not None:
                glasses_width = int(w * 0.8)
                resized_glasses = cv2.resize(
                    current_glasses_img,
                    (glasses_width, int(current_glasses_img.shape[0] * (glasses_width / current_glasses_img.shape[1])))
                )

                left_eye = (x + int(w * 0.3), y + int(h * 0.4))
                right_eye = (x + int(w * 0.7), y + int(h * 0.4))

                glasses_center_x = (left_eye[0] + right_eye[0]) // 2
                glasses_center_y = (left_eye[1] + right_eye[1]) // 2

                pos_x = glasses_center_x - glasses_width // 2
                pos_y = glasses_center_y - int(resized_glasses.shape[0] * 0.4)

                processed_frame = overlay_with_alpha(processed_frame, resized_glasses, pos_x, pos_y)

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', glasses_files=FRONTEND_GLASSES_PATHS)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/select_glasses', methods=['POST'])
def select_glasses():
    global current_glasses_img
    index = int(request.form['index'])

    if index == -1:
        # No glasses → clear selection
        current_glasses_img = None
        return "No glasses selected", 200

    if 0 <= index < len(FRONTEND_GLASSES_PATHS):
        current_glasses_img = cv2.imread(FRONTEND_GLASSES_PATHS[index], cv2.IMREAD_UNCHANGED)
        if current_glasses_img is None:
            return "Error: Could not load glasses image.", 400
        return "Success", 200
    return "Invalid index", 400
@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("Camera released.")
    return "Camera stopped", 200


if __name__ == '__main__':
    if FRONTEND_GLASSES_PATHS:
        current_glasses_img = cv2.imread(FRONTEND_GLASSES_PATHS[0], cv2.IMREAD_UNCHANGED)
    app.run(debug=True, port=8000)