import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np

cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
strange=cv2.imread('strange2.jpg')
love=cv2.imread('love.png')
detector=HandDetector()
angle=1
speed = 5
def mapFromTo(x,a,b,c,d):
    return (x-a)/(b-a)*(d-c)+c

def overLay (bgImg,overlay,x,y,size):
    bgh,bgw,c=bgImg.shape
    imgScale=mapFromTo(size,200,20,1.5,0.2)
    overlay=cv2.resize(overlay,(0,0),fx=imgScale,fy=imgScale)
    h,w,c=overlay.shape
    try:
        if x+w/2 >=bgw or y+h/2 >=bgh:
            return bgImg
        else:
            overlayImg=overlay[..., :3]
            mask=overlay/127.5
            bgImg[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]= (1-mask)*bgImg[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]
            return bgImg
    except:
        return bgImg

while True:
    success,img = cap.read()
    img=cv2.flip(img,flipCode=1)
    hands,img=detector.findHands(img)
    if len(hands)==2:
        hand1=hands[0]
        hand2=hands[1]
        lmList1=hand1['lmList']
        lmList2 = hand2['lmList']

        fingers1=detector.fingersUp(hand1)
        fingers2=detector.fingersUp(hand2)
        #print (fingers1,end='')
        #if fingers2:
         #   print(fingers2)

        if fingers1==[1,1,0,0,1]:
            h,k=hand1['center']
            rad = abs(hand1['center'][0] - lmList2[8][0])
            a=2*math.pi/6
            pts1=np.array([[h+int(rad*math.cos(a)),k+int(rad*math.sin(a))],[h+int(rad*math.cos(3*a)),k+int(rad*math.sin(3*a))],[h+int(rad*math.cos(5*a)),k+int(rad*math.sin(5*a))],[h+int(rad*math.cos(a)),k+int(rad*math.sin(a))]])
            pts1=np.reshape(pts1,(-1,1,2))
            pts2 = np.array([[h + int(rad * math.cos(0)), k + int(rad * math.sin(0))],
                            [h + int(rad * math.cos(2*a)), k + int(rad * math.sin(2*a))],
                            [h + int(rad * math.cos(4*a)), k + int(rad * math.sin(4*a))],
                            [h + int(rad * math.cos(0)), k + int(rad * math.sin(0))]])
            pts2=np.reshape(pts2,(-1,1,2))
            cv2.polylines(img,[pts1],isClosed=False,color=(66,132,245),thickness=9,lineType=cv2.LINE_4)
            cv2.polylines(img, [pts2], isClosed=False, color=(66,132,245), thickness=9, lineType=cv2.LINE_4)
            cv2.circle(img,hand1['center'],rad,(0,255,0),thickness=9,lineType=cv2.LINE_4)


        if fingers2==[1,0,0,0,0]:
            h, k = hand2['center']
            rad = abs(hand2['center'][0] - lmList1[8][0])


            cv2.circle(img, (h, k), rad, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (h, k), rad-rad//5, (255, 245, 245), cv2.FILLED)
            cv2.circle(img, (h, k), rad-2*rad//5, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (h, k), rad - 3 * rad // 5, (255,0,0), cv2.FILLED)

            rad=rad-3*rad//5

            a = 2 * math.pi / 6
            pts1 = np.array([[h + int(rad * math.cos(a)), k + int(rad * math.sin(a))],
                             [h + int(rad * math.cos(3 * a)), k + int(rad * math.sin(3 * a))],
                             [h + int(rad * math.cos(5 * a)), k + int(rad * math.sin(5 * a))],
                             [h + int(rad * math.cos(a)), k + int(rad * math.sin(a))]])
            pts1 = np.reshape(pts1, (-1, 1, 2))
            pts2 = np.array([[h + int(rad * math.cos(0)), k + int(rad * math.sin(0))],
                             [h + int(rad * math.cos(2 * a)), k + int(rad * math.sin(2 * a))],
                             [h + int(rad * math.cos(4 * a)), k + int(rad * math.sin(4 * a))],
                             [h + int(rad * math.cos(0)), k + int(rad * math.sin(0))]])
            pts2 = np.reshape(pts2, (-1, 1, 2))

            cv2.fillPoly(img, [pts1], color=(245, 220, 230), lineType=cv2.LINE_4)
            cv2.fillPoly(img, [pts2], color=(245, 250, 220), lineType=cv2.LINE_4)


        if fingers1==[0,0,0,0,0]:
            h1,k1=hand1['center']
            strange1=cv2.resize(strange,(0,0),None,0.4,0.4)
            bbox=hand1['bbox']
            handsize=bbox[2]
            strange1 = cvzone.rotateImage(strange1, angle)
            speed = (lmList1[5][0] - lmList1[17][0])
            angle += 10
            img=overLay(img,strange1,h1,k1,handsize)

        if fingers2==[1,1,1,1,1]:
            h2, k2 = hand2['center']
            strange2 = cv2.resize(strange, (0, 0), None, 0.4, 0.4)

            strange2=cvzone.rotateImage(strange2,angle)
            speed = (lmList2[5][0]-lmList2[17][0])
            angle+=speed


            bbox = hand2['bbox']
            handsize = bbox[2]
            img = overLay(img, strange2, h2, k2, handsize)
        if fingers1 == [1,1,0,0,0]:
            h3,k3,z3=lmList1[4]

            #love=cv2.resize(love,(0,0),fx=0.4,fy=0.4)

            bbox = hand2['bbox']
            handsize = bbox[2]
            img=overLay(img,love,h3,k3,handsize)

        # if fingers1==[0,1,0,0,0]:
        #     cv2.rectangle(img,(lmList1[8][0],lmList1[8][1]),(lmList2[8][0],lmList2[8][1]),(0,0,255),10,cv2.FILLED)



    cv2.imshow('image',img)
    key=cv2.waitKey(1)
    if key==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()