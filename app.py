from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import base64
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Flask_app/static/uploads/'
app.config['PROCESSED_FOLDER'] = 'Flask_app/static/processed/'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[0, 0, 255], thickness=15):
    if lines is not None:
        img_shape = img.shape
        right_lines = []
        left_lines = []
        
        left_lane_longest = {'length': 0, 'm': None, 'b': None}
        right_lane_longest = {'length': 0, 'm': None, 'b': None}
    
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope > 0.4:  # Right lane
                    right_lines.append([x1, y1])
                    right_lines.append([x2, y2])
                elif slope < -0.4:  # Left lane
                    left_lines.append([x1, y1])
                    left_lines.append([x2, y2])
           
        if len(right_lines) >= 2:
            for x1, y1 in right_lines[:-1]:
                for x2, y2 in right_lines[1:]:
                    distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                    if distance > left_lane_longest['length']:
                        left_lane_longest['length'] = distance
                        left_lane_longest['m'] = abs(float(y1 - y2) / (x1 - x2))
                        left_lane_longest['b'] = y1 - left_lane_longest['m'] * x1

            y1 = img_shape[0]
            x1 = int((y1 - left_lane_longest['b']) / left_lane_longest['m'])
            y2 = int(y1 / 1.6)
            x2 = int((y2 - left_lane_longest['b']) / left_lane_longest['m'])
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        if len(left_lines) >= 2:
            for x1, y1 in left_lines[:-1]:
                for x2, y2 in left_lines[1:]:
                    distance = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                    if distance > right_lane_longest['length']:
                        right_lane_longest['length'] = distance
                        right_lane_longest['m'] = -abs(float(y1 - y2) / (x1 - x2))
                        right_lane_longest['b'] = y1 - right_lane_longest['m'] * x1
                        
            y1 = img_shape[0]
            x1 = int((y1 - right_lane_longest['b']) / right_lane_longest['m'])
            y2 = int(y1 / 1.6)
            x2 = int((y2 - right_lane_longest['b']) / right_lane_longest['m'])
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    return cv2.addWeighted(initial_img, α, img, β, γ)

def process_image(image):
    initial_image = np.copy(image)
    gray = grayscale(initial_image)
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)
    low_threshold = 50
    high_threshold = 150
    edge_image = canny(blur_gray, low_threshold, high_threshold)
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (480, 320), (510, 320), (imshape[1], imshape[0])]], dtype=np.int32)
    image_after_masking = region_of_interest(edge_image, vertices)
    rho = 1
    theta = np.pi/180 
    threshold = 20    
    min_line_len = 20
    max_line_gap = 300 
    line_image = hough_lines(image_after_masking, rho, theta, threshold, min_line_len, max_line_gap)
    result_image = weighted_img(line_image, image, α=1.0, β=0.95, γ=0.)
    return result_image

def process_frame(frame):
    return process_image(frame)

def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(input_path))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print("Error: Could not open input video file.")
        return None

    if not out.isOpened():
        print("Error: Could not open output video file.")
        return None

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    cap.release()

    with ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(process_frame, frames))

    for frame in processed_frames:
        out.write(frame)

    out.release()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Processed video saved at: {output_path}")
    else:
        print("Error: Processed video file was not created correctly.")

    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            output_filepath = process_video(filepath)
            return redirect(url_for('download_file', filename=os.path.basename(output_filepath)))
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = cv2.imread(filepath)
            processed_image = process_image(image)
            processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
            cv2.imwrite(processed_image_path, processed_image)
            return redirect(url_for('download_file', filename=filename))
        else:
            return "Unsupported file format"

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)