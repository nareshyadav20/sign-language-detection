import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("sign_language_model.h5")
letters = [chr(i) for i in range(65, 91)]  # A-Z

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]  # Region of Interest
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 64, 64, 1)

    result = model.predict(reshaped)
    predicted_class = np.argmax(result)
    predicted_letter = letters[predicted_class]

    cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
    cv2.putText(frame, f"Letter: {predicted_letter}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
