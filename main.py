import cv2
import mediapipe as mp

# โหลดโมเดล Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe สำหรับตรวจจับมือ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # กลับด้านภาพ
    frame = cv2.flip(frame, 1)  # 1 คือกลับด้านในแนวนอน

    # แปลงภาพจาก BGR เป็น RGB (สำหรับ MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = hands.process(rgb_frame)

    # แปลงภาพเป็นขาวดำ (Grayscale) สำหรับ Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # วาดจุดเชื่อมต่อบนมือที่ตรวจจับได้
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

   
    cv2.imshow('Face and Hand Detection', frame)

    # oooooooooooo
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
