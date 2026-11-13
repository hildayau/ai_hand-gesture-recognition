import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.7, maxHands=1)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    gesture = ""

    if hands:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        print(fingers)

        # ✊ 拳頭
        if fingers == [0, 0, 0, 0, 0]:
            gesture = "Fist"

        # 👍（只開拇指）
        elif fingers == [1, 0, 0, 0, 0]:
            gesture = "Thumbs Up"

        # ✌️（index + middle）
        elif fingers == [0, 1, 1, 0, 0]:
            gesture = "Peace"

        else:
            gesture = ""

        # 只顯示有定義手勢
        if gesture:
            cv2.putText(img, gesture, (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0,255,0), 3)

    cv2.imshow("Hand Tracking", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()