import base64
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, Response
import os
import face_recognition
import pickle
import cv2
from train.knn_train import train_knn_model
from deepface import DeepFace
import unicodedata
import re
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def normalize_name(name):
    name = unicodedata.normalize('NFD', name)
    name = name.encode('ascii', 'ignore').decode('utf-8')
    name = re.sub(r'\s+', '_', name)
    return name.lower()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    name_raw = data.get('name')
    image_data = data.get('image')

    if not name_raw or not image_data:
        return {"success": False, "message": "Thiếu dữ liệu"}, 400

    name = normalize_name(name_raw)

    person_folder = os.path.join('dataset', name)
    os.makedirs(person_folder, exist_ok=True)

    header, encoded = image_data.split(",", 1)
    img_bytes = base64.b64decode(encoded)

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"success": False, "message": "Ảnh rỗng"}, 400

    img_count = len(os.listdir(person_folder))
    img_path = os.path.join(person_folder, f"{name}_{img_count + 1}.jpg")

    saved = cv2.imwrite(img_path, frame)

    if not saved:
        return {"success": False, "message": "Không lưu được ảnh"}, 500

    return {"success": True}

@app.route('/train', methods=['POST'])
def train():
    train_knn_model()
    return redirect(url_for('index'))

@app.route('/camera')
def camera():
    return render_template('camera.html')  # cần tạo camera.html

def generate_video():
    with open('knn_model.pkl', 'rb') as f:
        knn_clf = pickle.load(f)

    cap = cv2.VideoCapture(0)
    process_every_n_frame = 3
    frame_count = 0
    cached_faces = []

    attendance_cache = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if frame_count % process_every_n_frame == 0:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            cached_faces = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=1)
                distance = closest_distances[0][0][0]

                if distance < 0.5:
                    name = knn_clf.predict([face_encoding])[0]
                else:
                    name = "Unknown"

                # Phân tích cảm xúc
                face_img = rgb_small_frame[top:bottom, left:right]
                try:
                    analysis = DeepFace.analyze(
                        face_img,
                        actions=['emotion'],
                        enforce_detection=False
                    )
                    emotion = analysis[0]['dominant_emotion']
                except:
                    emotion = "N/A"

                # Nếu nhận diện được và chưa chấm công → ghi thời gian
                checkin_time = None
                if name != "Unknown":
                    if name not in attendance_cache:
                        attendance_cache[name] = datetime.now().strftime("%H:%M:%S")
                    checkin_time = attendance_cache[name]

                cached_faces.append({
                    "top": top * 4,
                    "right": right * 4,
                    "bottom": bottom * 4,
                    "left": left * 4,
                    "name": name,
                    "distance": distance,
                    "emotion": emotion,
                    "checkin_time": checkin_time
                })

        for face in cached_faces:
            top = face["top"]
            right = face["right"]
            bottom = face["bottom"]
            left = face["left"]
            name = face["name"]
            best_distance = face["distance"]
            emotion = face["emotion"]
            checkin_time = face["checkin_time"]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"{name} ({best_distance:.2f})",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"Emotion: {emotion}",
                (left, bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

            if checkin_time:
                cv2.putText(
                    frame,
                    f"✔ Check-in: {checkin_time}",
                    (left, bottom + 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 0),
                    2
                )

        frame_count += 1
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    
@app.route('/video_feed')
def video_feed():
    return Response(generate_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

from datetime import datetime

@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    if request.method == 'GET':
        return render_template('recognize.html')

    file = request.files['image']
    if not file:
        return "No file uploaded.", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    with open('knn_model.pkl', 'rb') as f:
        knn_clf = pickle.load(f)

    image = face_recognition.load_image_file(filepath)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    distance_results = []

    if len(face_locations) == 0:
        distance_results.append("Không phát hiện khuôn mặt.")
    else:
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            name = "Unknown"
            closest_distances = knn_clf.kneighbors([face_encoding], n_neighbors=1)
            distance = closest_distances[0][0][0]

            # Nhận diện thành công
            if distance < 0.5:
                name = knn_clf.predict([face_encoding])[0]
                checkin_time = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
            else:
                checkin_time = None

            # Phân tích cảm xúc
            face_img = image[top:bottom, left:right]
            try:
                analysis = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False
                )
                emotion = analysis[0]['dominant_emotion']
            except:
                emotion = "Không xác định"

            # Text hiển thị
            if checkin_time:
                result_text = (
                    f"{name} | {emotion} | "
                    f"Chấm công lúc: {checkin_time}"
                )
            else:
                result_text = f"Unknown | {emotion}"

            distance_results.append(result_text)

            # Vẽ khung + tên
            cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                image_bgr,
                name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            # Vẽ cảm xúc
            cv2.putText(
                image_bgr,
                f"Emotion: {emotion}",
                (left, bottom + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            # Vẽ thời gian chấm công (nếu có)
            if checkin_time:
                cv2.putText(
                    image_bgr,
                    f"Check-in: {checkin_time}",
                    (left, bottom + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 128, 0),
                    2
                )

    result_path = os.path.join(UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(result_path, image_bgr)

    return render_template(
        'recognize.html',
        result_image=result_path,
        distances=distance_results
    )

if __name__ == '__main__':
    app.run(debug=True)
