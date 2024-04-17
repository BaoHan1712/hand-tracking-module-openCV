import cv2 
import time
import mediapipe as mp

cap =cv2.VideoCapture(0)
mpHands = mp.solutions.hands
# phát hiện tay
hands = mpHands.Hands()
# vẽ lại các khớp tay
mpDraw = mp.solutions.drawing_utils
# tính FPS
pTime =0
cTime =0

while True:
    success,img=cap.read()
# chuyển màu từ BGR sang RGB
    imRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# xử lý ảnh để phát hiện bàn tay
    results = hands.process(imRGB)
    if results.multi_hand_landmarks:
# duyệt qua từng ngón tay
        for handLms in results.multi_hand_landmarks:
            lmList = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in handLms.landmark]
# Vẽ ra các khớp nối
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
     
#thời gian thực và ghi ra
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv2.imshow("Runtime", img)
# Thoát khỏi vòng lặp 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
