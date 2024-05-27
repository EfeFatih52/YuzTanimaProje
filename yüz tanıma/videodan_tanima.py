import cv2
import numpy as np
import os

# Yüz tanıma için kullanılan model dosyasının yolu
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# LBPH yüz tanıma modelini oluşturma
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Eğitim verilerini yükleme
training_data_path = 'face_db'

# Kişi etiketleri
labels = []
faces = []
label_dict = {}

# Eğitim verilerini yükleme
label_id = 0
for person_name in os.listdir(training_data_path):
    person_path = os.path.join(training_data_path, person_name)
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(gray_image)
        labels.append(label_id)
    label_dict[label_id] = person_name
    label_id += 1

# Modeli eğitme
recognizer.train(faces, np.array(labels))

# Videodan yüz tanıma
video_path = 'efe.mp4'  # Tanıma yapılacak video dosyasının yolu, eğer kameradan alınacaksa 0 yazılmalı
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face)

        if confidence < 100:  # Güven eşiği
            name = label_dict[label]
            confidence_text = f"{round(100 - confidence)}%"
        else:
            name = "Unknown"
            confidence_text = ""

        # Yüzün etrafına dikdörtgen çizme ve ismi yazdırma
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} {confidence_text}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    # 'q' tuşuna basıldığında döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
