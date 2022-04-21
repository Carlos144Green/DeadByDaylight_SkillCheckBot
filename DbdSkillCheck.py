import cv2
import numpy as np
import pyautogui
import math
import winsound

def empty(a):
    pass


breakFlag = False
frequency1 = 1000
duration = 50
threshold = .70
kernel = np.ones((3, 3), np.uint8)
methods = 'cv2.TM_CCOEFF_NORMED'
i = None


SCREEN_SIZE = (2560, 1440)                                                              ############ EDIT ME ###########
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, 20.0, (SCREEN_SIZE))

template = cv2.imread('space.JPG', 0)                                                   ############ EDIT ME ###########
h, w = template.shape
clipRegion = 300, 300
clipXY = (SCREEN_SIZE[0] - clipRegion[0])/2, (SCREEN_SIZE[1] - clipRegion[1])/2

while True:
    rawImg = pyautogui.screenshot(region=(clipXY[0], clipXY[1], clipRegion[0], clipRegion[1]))      # Screenshot
    frame = np.array(rawImg)                                    # convert into numpy array
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)                # change to RGB for visual
    out.write(img)                                              # write the frame
    # cv2.imshow("screenshot", img)                               # Show color result
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)             # RGB to Gray for template matching
    anchor = img.copy()                                         # Clone result for anchor

    img = anchor.copy()                                         # Clone result for anchor
    res = cv2.matchTemplate(grayImg, template, eval(methods))   # Apply template Matching
    matches = np.where(res >= threshold)                        # Compares confidence levels to only use guaranteed matches
    matches = list(zip(*matches[::-1]))

    if matches:
        winsound.Beep(frequency1, duration)                     # Beep if match found
        breakFlag = True
        bottom_right = (matches[0][0] + w, matches[0][1] + h)       # Add w and h to biggest hotspot to get the bottom right corner of template

    if breakFlag is True:
        breakFlag = False
        break
    if cv2.waitKey(1) == ord("q"):
        break

circles = cv2.HoughCircles(grayImg, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=50,minRadius=50, maxRadius=95)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])                               # circle center
        radius = i[2]
        outerRadius = radius*1.05
        print("Center: ", center)

        cv2.circle(img, center, 1, (255, 0, 255), 3)
        cv2.circle(img, center, math.ceil(outerRadius), (255, 0, 255), 1)
        cv2.circle(img, center, math.ceil(outerRadius), (255, 0, 255), 1)

        #all this is to black out behind the skill check
        mask = np.zeros_like(anchor)
        cv2.circle(mask, center, math.ceil(outerRadius), (255,255,255), -1)
        only_skillcheck = cv2.bitwise_and(anchor, mask)

        gray_sk = cv2.cvtColor(only_skillcheck, cv2.COLOR_BGR2GRAY)             # RGB to Gray for template matching
        blur = cv2.GaussianBlur(gray_sk, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 242, 255, 0)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(opening, kernel, iterations=1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 45:
                # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
                M = cv2.moments(contour)
                XY = (int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"]))
                print("Great XY: ", XY)

                cv2.circle(img, XY, 2, (0, 0, 0), 3)
                cv2.line(img, XY, center, (255, 255, 255), 1)
else:
    print("skill check circle not found, edit settings")

hue_M = 7
hue_m = 0
sat_M = 255
sat_m = 174
val_M = 255
val_m = 114

hsv = cv2.cvtColor(only_skillcheck, cv2.COLOR_BGR2HSV)  # Convert to HSV
ranges = cv2.inRange(hsv, np.array([hue_m, sat_m, val_m]), np.array([hue_M, sat_M, val_M]))
dilation = cv2.dilate(ranges, kernel, iterations=2)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
erosion = cv2.erode(closing, kernel, iterations=1)
cnts, _ = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area > 45:
        # cv2.drawContours(img, cnt, -1, (0, 255, 0), 3)
        M = cv2.moments(cnt)
        XY2 = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        print("Hand XY: ", XY2)

        cv2.circle(img, XY2, 2, (255, 255, 255), 3)
        cv2.line(img, XY2, center, (0,0,255), 1)

center2red = math.sqrt((center[0] - XY2[0])**2 + (center[1] - XY2[1])**2)
center2great = math.sqrt((center[0] - XY[0])**2 + (center[1] - XY[1])**2)
red2great = math.sqrt((XY2[0] - XY[0])**2 + (XY2[1] - XY[1])**2)

radian = math.acos((center2red**2+center2great**2-red2great**2)/(2*center2red*center2great))
angle = radian*180/math.pi
print("Angle: ", angle)

if angle <= 9:
    print("CLICK")

cv2.imshow("Closing & Erosion", erosion)
cv2.imshow("gray_sk", opening)
cv2.imshow("img", img)
# cv2.imshow("only skillcheck", only_skillcheck)
# cv2.imshow("raw hand", ranges)

cv2.waitKey(0)
cv2.destroyAllWindows()
out.release()
