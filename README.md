# Face-Recognition-Model-Using-Machine-Learning.

This project implements a face detection and recognition system using OpenCV and the face_recognition library. It can detect faces in images, extract their features, and match them to known faces.

Note - Make sure you have Python 3.x installed on your system. You can download Python from python.org.

Clone the Repository - 
git clone https://github.com/your-username/face-detection-recognition.git
cd face-detection-recognition

Create a virtual environment (optional but recommended):

Install the required packages:
pip install -r requirements.txt
Requirements File

Here’s the requirements.txt content:
opencv-python
face_recognition
numpy
Usage
Face Detection

Function: detect_faces(image_path)
Description: Detects faces in an image.
image_path: Path to the input image file.
Returns: List of bounding boxes for detected faces.

Example:

def detect_faces(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

faces = detect_faces('image.jpg')
print(faces)

Face Recognition
Function: recognize_face(image_path, known_faces_encodings, known_face_names)
Description: Recognizes faces in an image using known face encodings.
image_path: Path to the input image file.
known_faces_encodings: List of known face encodings.
known_face_names: List of names corresponding to known face encodings.
Returns: List of tuples containing face locations and recognized names.

Example:
import face_recognition
import numpy as np

def recognize_face(image_path, known_faces_encodings, known_face_names):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_faces_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_faces_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    
    return list(zip(face_locations, face_names))

known_faces_encodings = [/* list of known face encodings */]
known_face_names = ['John Doe', 'Jane Smith']
results = recognize_face('test_image.jpg', known_faces_encodings, known_face_names)
print(results)

Performance - 

Accuracy: The accuracy of face recognition is evaluated based on how correctly the model identifies known faces.
Speed: The system's performance is measured by the time it takes to process images and detect/recognize faces.
Optimization
Inference Time: Techniques such as model pruning or quantization may be applied to speed up face recognition.
Scalability: Efficient data handling and batch processing methods are used to handle large datasets.

