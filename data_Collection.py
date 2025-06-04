from time import time
import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

####################################
classID = 1  # 0 is fake, 1 is real
outputFolderPath = 'Dataset/Data Collect'
confidence = 0.8
save = True
blurThreshold = 35  # Larger means more focus

debug = False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint = 6
####################################

cap = cv2.VideoCapture(0)
cap.set(3, camWidth)
cap.set(4, camHeight)

detector = FaceDetector()

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue  # Skip to next iteration if frame capture fails

    imgOut = img.copy()  # Fixed incorrect function call
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True/False values indicating if the faces are blurry
    listInfo = []  # Normalized values and class name for label text file

    if bboxs:
        for bbox in bboxs:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            # Check confidence score
            if score > confidence:
                # Adding an offset to the face detected
                offsetW = (offsetPercentageW / 100) * w
                offsetH = (offsetPercentageH / 100) * h
                x = max(0, int(x - offsetW))
                y = max(0, int(y - offsetH * 3))
                w = int(w + offsetW * 2)
                h = int(h + offsetH * 3.5)

                # Ensure valid face cropping
                ih, iw, _ = img.shape
                x2, y2 = min(iw, x + w), min(ih, y + h)

                imgFace = img[y:y2, x:x2].copy()  # Extract face region safely
                if imgFace.size == 0:
                    continue  # Skip empty face regions

                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                listBlur.append(blurValue > blurThreshold)

                # Normalize values
                xc, yc = x + w / 2, y + h / 2
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                # Ensure values are within bounds
                xcn, ycn, wn, hn = min(1, xcn), min(1, ycn), min(1, wn), min(1, hn)
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # Draw bounding box and text
                cv2.rectangle(imgOut, (x, y), (x2, y2), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                   scale=2, thickness=3)

                if debug:
                    cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 10),
                                       scale=2, thickness=3)

        # Save images and labels if all detected faces are not blurry
        if save and all(listBlur) and listBlur:
            timeNow = str(int(time() * 1000000))  # Unique timestamp
            cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

            with open(f"{outputFolderPath}/{timeNow}.txt", 'a') as f:
                f.writelines(listInfo)

    cv2.imshow("Image", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
