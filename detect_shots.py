#!/usr/bin/env /full/path/to/your/venv/bin/python

import cv2
import numpy as np
import csv
from ultralytics import YOLO
from collections import deque
import easyocr

# Load YOLOv8 (replace with custom modesl if available)
model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(["en"])

# Video input
cap = cv2.VideoCapture("match.mp4")

# Tracking
ball_positions = deque(maxlen=10)
player_positions = {}
player_teams = {}
player_numbers = {}

# Output CSV
output_csv = open("shots_output.csv", "w", newline="")
csv_writer = csv.writer(output_csv)
csv_writer.writerow(["frame", "ball_x", "ball_y", "jersey_number", "team"])

# Config
SHOT_SPEED_THRESHOLD = 25
frame_id = 0
team_color_refs = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    current_players = {}
    ball_pos = None

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box.astype(int)
        label = names[cls]

        if label == "person":
            pid = (x1, y1, x2, y2)
            current_players[pid] = (x1, y1, x2, y2)
        elif label == "sports ball":
            ball_pos = ((x1 + x2) // 2, (y1 + y2) // 2)

    if ball_pos:
        ball_positions.append(ball_pos)

        # Compute velocity
        if len(ball_positions) >= 2:
            dx = ball_positions[-1][0] - ball_positions[-2][0]
            dy = ball_positions[-1][1] - ball_positions[-2][1]
            speed = np.sqrt(dx ** 2 + dy ** 2)

            if speed > SHOT_SPEED_THRESHOLD:
                # Detect shooter
                min_dist = float('inf')
                shooter_bbox = None
                for bbox in current_players.values():
                    px = (bbox[0] + bbox[2]) / 2
                    py = (bbox[1] + bbox[3]) / 2
                    dist = np.linalg.norm(np.array([px, py]) - np.array(ball_pos))
                    if dist < min_dist:
                        min_dist = dist
                        shooter_bbox = bbox

                if shooter_bbox:
                    x1, y1, x2, y2 = shooter_bbox
                    crop = frame[y1:y2, x1:x2]
                    text = ocr_reader.readtext(crop)

                    jersey_number = "?"
                    for (_, txt, _) in text:
                        if txt.isdigit():
                            jersey_number = txt
                            break

                    # Team detection (naive color-based)
                    jersey = crop
                    avg_color = jersey.mean(axis=(0, 1))
                    team = "Team A" if avg_color[2] > avg_color[0] else "Team B"

                    print(f"[SHOT] Frame {frame_id} | Ball: {ball_pos} | Jersey: {jersey_number} | Team: {team}")
                    csv_writer.writerow([frame_id, ball_pos[0], ball_pos[1], jersey_number, team])

    # Visualization (optional)
    if ball_pos:
        cv2.circle(frame, ball_pos, 5, (0, 255, 255), -1)
    for bbox in current_players:
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    cv2.imshow("Shot Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_csv.close()
cv2.destroyAllWindows()
