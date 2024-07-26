# Lane-Detection
This repository contains the implementation of a Lane Detection system designed to enhance vehicle safety and support autonomous driving capabilities.
The Lane Detection project aims to enhance vehicle safety and autonomous driving capabilities by accurately identifying and tracking lane markers on the road. This system utilizes advanced computer vision techniques to process real-time video feeds from vehicle-mounted cameras, detecting lane boundaries and providing critical information for safe navigation.

# demo link
https://youtu.be/Um0bbGlsMpE?si=AeEz5vOjP18f9d9p

# Features
Real-Time Video Processing: Efficient handling of video frames using concurrent processing.
Lane Detection: Accurate identification of lane markers using OpenCV's image processing tools.
Web Interface: A simple Flask-based web application for uploading video files and viewing lane detection results.

# Technologies Used
1.Flask
2.OpenCV
3.Concurrent Processing

# Getting Started
1. Clone the repository: git clone https://github.com/username/lane-detection-project.git
2. Navigate to the project directory: cd lane-detection-project
3. Install the required dependencies: pip install -r requirements.txt
4. Run the Flask application: python app.py

NOTE: Create a flask_app folder in your project directory, then add a static folder inside it containing processed and upload folders for images and videos, and also add templates folder in your falsk_app folder that containing index.html.
Ensure you place your app.py file in the flask_app directory and check the paths correctly.

# Usage
Upload a video file through the web interface to see the lane detection in action. The processed video with highlighted lane boundaries will be displayed on the interface.

# Future Enhancements
1. Advanced lane detection algorithms
2. Database integration
3. Mobile application support

# License
This project is licensed under the MIT License.
