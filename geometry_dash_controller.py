import cv2
from cvzone.HandTrackingModule import HandDetector
import concurrent.futures
import pyautogui

camera = cv2.VideoCapture(0)

detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

jumping = False

while True:
    success, img = camera.read()
    x1, y1, x2, y2 = 0, 0, img.shape[1], int(img.shape[0] / 2)

    img = img[y1:y2, x1:x2]

    hands, img = detector.findHands(img, draw=True, flipType=True)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1['lmList']
        bbox1 = hand1['bbox']
        center1 = hand1['center']
        handType1 = hand1['type']

        fingers1 = detector.fingersUp(hand1)

        length, info, img = detector.findDistance(lmList1[8][0:2], lmList1[4][0:2], img, color=(255, 0, 255),
                                                  scale=10)

        length = int(length) // 10 * 10
        if length > 40:
            if not jumping:
                jumping = True
                executor.submit(pyautogui.keyDown, 'space')
            cv2.putText(img, 'jump  ', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            if jumping:
                jumping = False
                executor.submit(pyautogui.keyUp, 'space')

    cv2.imshow('test', img)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
