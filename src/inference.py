
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from cvzone.HandTrackingModule import HandDetector


# 定義神經網絡
class GestureNet(nn.Module):
    def __init__(self):
        super(GestureNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(42, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.model(x)


# 載入模型
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "gesture_model.pth"

model = GestureNet()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()

labels = ["Fist", "Thumbs Up", "Peace"]


# 初始化 Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("無法開啟 Webcam")

detector = HandDetector(maxHands=1)


# 即時手勢辨識
try:
    while True:
        success, img = cap.read()

        if not success:
            print("無法讀取 Webcam 畫面")
            break

        hands, img = detector.findHands(img)

        if hands:
            lmList = hands[0]["lmList"]

            feat = []
            for lm in lmList:
                feat.extend([lm[0] / 640, lm[1] / 480])

            input_tensor = torch.tensor(
                feat, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                prediction = model(input_tensor)
                class_id = torch.argmax(prediction).item()
                conf = torch.softmax(
                    prediction, dim=1
                )[0][class_id].item()
            
            # 檢查手指狀態
            fingers = detector.fingersUp(hands[0])
            #print("Hand Type:", hands[0]["type"])
            #print("Fingers:", fingers)
            #print("AI Prediction:", labels[class_id], conf)
    
            # 只檢查食指、中指、無名指、小指
            is_open_palm = fingers[1:] == [1, 1, 1, 1]

            if not is_open_palm and conf > 0.8:
                cv2.putText(
                    img,
                    labels[class_id],
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3
                )

        cv2.imshow("AI Gesture Recognition", img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()