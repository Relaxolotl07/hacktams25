import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import os

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

def detect_fingers():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(1)  # Try changing to cv2.VideoCapture(1) if needed

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    labels_dict = {0: "Click", 1: "No Click"}
    last_click_time = time.time()

    while True:
        data_aux = []
        x_ = []
        y_ = []
        
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Check if camera is connected.")
            continue

        H, W, _ = frame.shape
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing_styles.get_default_hand_landmarks_style(),
                                       mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                if(predicted_character == "Click" and  time.time() - last_click_time > 3):
                    last_click_time = time.time()
                    os.system("python Voice.py")
                    print("Click")

            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_character = "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow("Finger Detection", frame)

        # Check if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)  # Add a small delay to avoid high CPU usage

    cap.release()
    cv2.destroyAllWindows()

detect_fingers()
