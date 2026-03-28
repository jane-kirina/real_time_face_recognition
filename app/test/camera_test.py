import cv2
# NOTE: Small script for manual test of webcam

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame")
        break

    cv2.imshow("Camera test", frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == ord('й'):
        break

cap.release()
cv2.destroyAllWindows()