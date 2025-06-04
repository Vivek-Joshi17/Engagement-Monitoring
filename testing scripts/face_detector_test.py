import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

# Initialize the webcam (Change '2' to '0' or '1' if necessary)
cap = cv2.VideoCapture(0)

# Initialize the FaceDetector object
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        continue  # Skip this frame if reading fails

    # Detect faces in the image
    img, bboxs = detector.findFaces(img, draw=False)

    # Check if any face is detected
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = int(bbox['score'][0] * 100)

            # Draw Data
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
            cvzone.putTextRect(img, f'{score}%', (x, y - 10), scale=1, thickness=1, colorR=(255, 0, 255))
            cvzone.cornerRect(img, (x, y, w, h), l=10, rt=2)

    # Display the image
    cv2.imshow("Image", img)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
