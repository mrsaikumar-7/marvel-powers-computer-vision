import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import mediapipe as mp

cap=cv2.VideoCapture(0q)
love=cv2.imread('strange.jpg')
love=cv2.cvtColor(love,cv2.COLOR_BGR2BGRA)
love=cv2.resize(love,(0,0),fx=0.4,fy=0.4)
h1,w1,c=love.shape

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))
    print(overlay_color.shape)

    # Optional, apply some simple filtering to the mask to remove edge noise
    mask = cv2.medianBlur(a,5)

    h, w, _ = overlay_color.shape

    x, y = int(x - (float(w) / 2.0)), int((y - float(h) / 2.0))
    roi = bg_img[y:y + h, x:x + w]

    # print(h, w)
    # print('x y: ', x, y)
    # print(roi.shape, mask.shape)

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))


    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)
    return bg_img


detector=HandDetector()

while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)

    if hands:
        hand=hands[0]
        h,k=hand['center']
        bbox=hand['bbox']

        imgres = overlay_transparent(img,love,h-h1//2,k-w1//2)
    #print(love.shape,img.shape)

        cv2.imshow('image',imgres)

    key = cv2.waitKey(1)

    if key==ord('q'):
        break

