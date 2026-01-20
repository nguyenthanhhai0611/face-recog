from sklearn.neighbors import KNeighborsClassifier
import os
import face_recognition
import pickle

def train_knn_model(dataset_path='dataset', model_path='knn_model.pkl', n_neighbors=3):
    X = []
    y = []

    if not os.path.exists(dataset_path):
        print(f"[!] Không tìm thấy thư mục dataset: {dataset_path}")
        return

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            try:
                image = face_recognition.load_image_file(image_path)
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) != 1:
                    print(f"[!] Ảnh bỏ qua (số khuôn mặt != 1): {image_path}")
                    continue

                face_encoding = face_recognition.face_encodings(image, face_locations)[0]
                X.append(face_encoding)
                y.append(person_name)

                print(f"[✓] Đã xử lý: {image_path} -> {person_name}")

            except Exception as e:
                print(f"[X] Lỗi khi xử lý {image_path}: {e}")

    if not X:
        print("[!] Không có dữ liệu huấn luyện.")
        return

    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X, y)

    with open(model_path, 'wb') as f:
        pickle.dump(knn_clf, f)

    print(f"\n✅ Huấn luyện xong. Mô hình lưu tại: {model_path}")

if __name__ == '__main__':
    train_knn_model()
