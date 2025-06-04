
import cv2
import cvzone
import math
import time
import dlib
import numpy as np
import csv
from imutils import face_utils
from scipy.spatial import distance
from ultralytics import YOLO
from pdf_report import csv_to_pdf

# Load YOLO model for spoof detection
spoof_model = YOLO("D:\\spoof detection\\version_2.pt")
classNames = ["fake", "real"]

# Load dlib model for drowsiness detection
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Parameters
thresh = 0.25  # Adjusted threshold for better drowsiness detection
frame_check = 12
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Video paths
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4", "video4.mp4"]
caps = [cv2.VideoCapture(video) for video in video_paths]
summary_data = []

# CSV file setup
csv_filename = "participant_analysis.csv"
csv_header = ["Participant", "Total Frames", "Spoof Frames", "Drowsy Frames", "Attentiveness (%)"]
with open(csv_filename, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_header)

# Check if videos opened successfully
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"⚠️ Error: Could not open {video_paths[i]}")
        caps[i] = None

flags = [0] * len(caps)  # Separate flags for each participant
total_frames = [0] * len(caps)
spoof_frames = [0] * len(caps)
drowsy_frames = [0] * len(caps)

while True:
    frames = []
    face_reals = []
    drowsy_counts = [0] * len(caps)
    not_distracted_counts = [0] * len(caps)
    saved_photos = [None] * len(caps)

    for i, cap in enumerate(caps):
        if cap is None:
            frames.append(None)
            face_reals.append(False)
            continue

        success, frame = cap.read()
        if not success:
            caps[i].release()
            caps[i] = None
            frames.append(None)
            face_reals.append(False)
            continue
        
        total_frames[i] += 1
        frame = cv2.resize(frame, (300, 200))  # Reduce output window size further
        frames.append(frame)
        
        # Step 1: Spoof Detection
        spoof_results = spoof_model(frame, stream=True)
        face_real = False

        for r in spoof_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                label = f'{classNames[cls]} {conf}'
                cvzone.putTextRect(frame, label, (max(0, x1), max(35, y1)), scale=1, thickness=1)
                if classNames[cls] == "real":
                    face_real = True
                else:
                    spoof_frames[i] += 1

        face_reals.append(face_real)

    # Step 2: Drowsiness Detection (only if real face detected)
    for i, frame in enumerate(frames):
        if frame is None or not face_reals[i]:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray)

        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
            leftEAR, rightEAR = eye_aspect_ratio(leftEye), eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

            not_distracted_counts[i] += 1
            if ear < thresh:
                flags[i] += 1
                if flags[i] >= frame_check:
                    drowsy_frames[i] += 1
                    cv2.putText(frame, "****************ALERT!****************", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                flags[i] = 0  # Reset only for this participant
                cv2.putText(frame, "**********NOT DISTRACTED************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 125, 0), 2)

            if saved_photos[i] is None:
                photo_path = f"participant_{i+1}.png"
                cv2.imwrite(photo_path, frame)
                saved_photos[i] = photo_path

    # Display videos in a 2x2 grid
    frame1 = frames[0] if len(frames) > 0 and frames[0] is not None else np.zeros((200, 300, 3), np.uint8)
    frame2 = frames[1] if len(frames) > 1 and frames[1] is not None else np.zeros((200, 300, 3), np.uint8)
    frame3 = frames[2] if len(frames) > 2 and frames[2] is not None else np.zeros((200, 300, 3), np.uint8)
    frame4 = frames[3] if len(frames) > 3 and frames[3] is not None else np.zeros((200, 300, 3), np.uint8)

    top_row = np.hstack((frame1, frame2))
    bottom_row = np.hstack((frame3, frame4))
    grid = np.vstack((top_row, bottom_row))

    cv2.imshow("Output window", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    if cap is not None:
        cap.release()
cv2.destroyAllWindows()

# Calculate and write attentiveness data to CSV
with open(csv_filename, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    for i in range(len(video_paths)):
        if total_frames[i] > 0:
            attentiveness = max(0, (1 - (spoof_frames[i] + drowsy_frames[i]) / total_frames[i])) * 100
        else:
            attentiveness = 0
        
        csv_writer.writerow([f"Participant {i+1}", total_frames[i], spoof_frames[i], drowsy_frames[i], f"{attentiveness:.2f}%"])

print(f"Analysis complete. Data saved to {csv_filename}")


#converting csv file to pdf"
csv_to_pdf("participant_analysis.csv")
