
import cv2
import csv
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "test_gesture_data.csv"

# 0: Fist, 1: Thumbs Up, 2: Peace
current_label = 2

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

if not cap.isOpened():
    raise RuntimeError("無法開啟 Webcam")

try:
    while True:
        success, img = cap.read()

        if not success:
            print("無法讀取 Webcam 畫面")
            break

        hands, img = detector.findHands(img)

        cv2.imshow("Data Collection (Press S to Save)", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s") and hands:
            lmList = hands[0]["lmList"]

            data = []
            for lm in lmList:
                data.extend([lm[0] / 640, lm[1] / 480])

            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

            with open(DATA_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([current_label] + data)

            print(f"已儲存標籤 {current_label} 的一組數據")

finally:
    cap.release()
    cv2.destroyAllWindows()