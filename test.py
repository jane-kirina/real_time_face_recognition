import cv2
import numpy as np
import random
import math

# canvas
size = 800
img = np.ones((size, size, 3), dtype=np.uint8) * 255

center = (size // 2, size // 2)

for angle in range(360):
    length = random.randint(100, 350)

    rad = math.radians(angle)

    x_end = int(center[0] + length * math.cos(rad))
    y_end = int(center[1] + length * math.sin(rad))

    cv2.line(img, center, (x_end, y_end), (0, 0, 0), 2)

cv2.imshow("Radial Art", img)
cv2.waitKey(0)
cv2.destroyAllWindows()