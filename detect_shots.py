import cv2
import numpy as np
import csv
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import easyocr
from sort import Sort  # Ensure SORT is available

model = YOLO("yolov8n.pt")
ocr_reader = easyocr.Reader(["en"])

cap = cv2.VideoCapture("match.mp4")
cv2.namedWindow("Robust Shot Detector", cv2.WINDOW_NORMAL)


tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

kf = KalmanFilter(dim_x=4, dim_z=2)
dt = 1
kf.F = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])
kf.P *= 1000.
kf.R = np.array([[10, 0],
                 [0, 10]])
kf.Q = np.eye(4) * 0.01

ball_tracked = False

output_csv = open("shots_output.csv", "w", newline="")
csv_writer = csv.writer(output_csv)
csv_writer.writerow(["frame", "ball_x", "ball_y", "jersey_number", "team"])

SHOT_SPEED_THRESHOLD = 30
SHOT_COOLDOWN_FRAMES = 15
last_shot_frame = -SHOT_COOLDOWN_FRAMES

# === Team Colors ===
# Blue (Team A)
team_a_lower = np.array([100, 80, 50])
team_a_upper = np.array([130, 255, 255])

# White (Team B): Low saturation, high value
team_b_lower = np.array([0, 0, 200])
team_b_upper = np.array([180, 60, 255])

frame_id = 0

def get_team_from_crop(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, team_a_lower, team_a_upper)
    mask_b = cv2.inRange(hsv, team_b_lower, team_b_upper)
    count_a = cv2.countNonZero(mask_a)
    count_b = cv2.countNonZero(mask_b)
    if count_a > count_b:
        return "Team A (Blue)"
    elif count_b > count_a:
        return "Team B (White)"
    else:
        return "Unknown"

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    player_boxes = []
    ball_box = None

    for box, cls in zip(boxes, classes):
        if names[cls] == "person":
            player_boxes.append(box)
        elif names[cls] == "sports ball":
            ball_box = box

    if len(player_boxes) > 0:
        player_boxes_np = np.array(player_boxes)
        dets = np.hstack((player_boxes_np, np.full((len(player_boxes_np),1), 0.99)))
        tracked_players = tracker.update(dets)
    else:
        tracked_players = np.empty((0,7))

    if ball_box is not None:
        x1, y1, x2, y2 = ball_box.astype(int)
        ball_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if not ball_tracked:
            kf.x[:2] = ball_center.reshape((2,1))
            kf.x[2:] = 0
            ball_tracked = True
        else:
            kf.predict()
            kf.update(ball_center)
    else:
        if ball_tracked:
            kf.predict()

    if ball_tracked:
        ball_pos = kf.x[:2].reshape(-1)
        ball_vel = kf.x[2:].reshape(-1)
        ball_speed = np.linalg.norm(ball_vel)
    else:
        ball_pos = None
        ball_speed = 0

    if ball_pos is not None and ball_speed > SHOT_SPEED_THRESHOLD and (frame_id - last_shot_frame) > SHOT_COOLDOWN_FRAMES:
        min_dist = float('inf')
        shooter = None
        shooter_id = None
        shooter_box = None

        for trk in tracked_players:
            x1, y1, x2, y2, pid = trk
            px = (x1 + x2) / 2
            py = (y1 + y2) / 2
            dist = np.linalg.norm(ball_pos - np.array([px, py]))
            if dist < min_dist:
                min_dist = dist
                shooter = pid
                shooter_box = (int(x1), int(y1), int(x2), int(y2))

        if min_dist < 100 and ball_pos[1] > frame.shape[0] * 2 / 3:
            x1, y1, x2, y2 = shooter_box
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            processed_crop = preprocess_for_ocr(crop)
            ocr_results = ocr_reader.readtext(processed_crop)

            jersey_number = "?"
            for (_, text, prob) in ocr_results:
                if text.isdigit():
                    jersey_number = text
                    break

            team = get_team_from_crop(crop)

            print(f"[SHOT] Frame {frame_id} | Ball: {ball_pos} | Speed: {ball_speed:.2f} | Jersey: {jersey_number} | Team: {team}")
            csv_writer.writerow([frame_id, int(ball_pos[0]), int(ball_pos[1]), jersey_number, team])
            last_shot_frame = frame_id

    vis_frame = frame.copy()
    if ball_pos is not None:
        cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 7, (0, 255, 255), -1)
    for trk in tracked_players:
        x1, y1, x2, y2, pid = trk
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_frame, f"ID:{int(pid)}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Robust Shot Detector", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output_csv.close()
cv2.destroyAllWindows()
