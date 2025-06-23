import cv2
import numpy as np
import csv
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import easyocr
from iou_tracker import IouTracker  # <-- Using the new, more robust tracker

# --- Configuration ---
VIDEO_PATH = "match.mp4"
OUTPUT_CSV_PATH = "shots_output_v2.csv"
MODEL_PATH = "yolov8n.pt"

# --- Goalpost and Field Line Definitions (MUST BE ADJUSTED FOR YOUR VIDEO) ---
# Define the coordinates of the goal. You may need to adjust these for different camera angles.
# Format: [x_left, y_top, x_right, y_bottom]
GOAL_AREA = [25, 220, 1260, 600]

# --- Shot Detection Parameters ---
SHOT_SPEED_THRESHOLD = 25  # Pixels per frame
SHOT_COOLDOWN_FRAMES = 30  # Frames to wait before detecting another shot
PLAYER_PROXIMITY_THRESHOLD = 150  # Max distance in pixels for a player to be considered the shooter

# --- Team Color Definitions ---
# These are HSV color ranges. You can fine-tune them if needed.
TEAM_A_LOWER = np.array([100, 80, 50])   # Blue
TEAM_A_UPPER = np.array([130, 255, 255])
TEAM_B_LOWER = np.array([0, 0, 200])     # White
TEAM_B_UPPER = np.array([180, 60, 255])

# --- Initialization ---
model = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(["en"])
tracker = IouTracker(max_lost=15, iou_threshold=0.5) # Using the new tracker
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Shot Detector", cv2.WINDOW_NORMAL)

# Kalman Filter for Ball Tracking
kf = KalmanFilter(dim_x=4, dim_z=2)
dt = 1.0
kf.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.P *= 1000.0
kf.R = np.array([[10, 0], [0, 10]])
kf.Q = np.eye(4) * 0.01

ball_tracked = False
last_shot_frame = -SHOT_COOLDOWN_FRAMES
shot_candidate = None
team_identities = {} # Dictionary to store team affiliation for each track ID

# --- CSV Output Setup ---
output_csv = open(OUTPUT_CSV_PATH, "w", newline="")
csv_writer = csv.writer(output_csv)
csv_writer.writerow(["frame", "ball_x", "ball_y", "jersey_number", "team", "is_goal"])

def get_team_from_crop(crop):
    """Identifies the team based on the dominant jersey color in a cropped image."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, TEAM_A_LOWER, TEAM_A_UPPER)
    mask_b = cv2.inRange(hsv, TEAM_B_LOWER, TEAM_B_UPPER)
    count_a = cv2.countNonZero(mask_a)
    count_b = cv2.countNonZero(mask_b)
    if count_a > count_b:
        return "Team A (Blue)"
    elif count_b > count_a:
        return "Team B (White)"
    return "Unknown"

def preprocess_for_ocr(img):
    """Prepares a cropped image for OCR by converting to grayscale and thresholding."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def check_if_goal(ball_position, goal_area):
    """Checks if the ball's coordinates are inside the defined goal area."""
    x, y = ball_position
    return goal_area[0] < x < goal_area[2] and goal_area[1] < y < goal_area[3]

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    
    vis_frame = frame.copy()

    # --- Object Detection ---
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

    # --- Player Tracking ---
    tracked_players = tracker.update(np.array(player_boxes))

    # --- Ball Tracking & Speed Calculation ---
    prev_ball_pos = kf.x[:2].reshape(-1) if ball_tracked else None

    if ball_box is not None:
        x1, y1, x2, y2 = ball_box.astype(int)
        ball_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        if not ball_tracked:
            kf.x[:2] = ball_center.reshape((2, 1))
            ball_tracked = True
        else:
            kf.predict()
            kf.update(ball_center)
    elif ball_tracked:
        kf.predict()

    ball_pos = kf.x[:2].reshape(-1)
    ball_vel = kf.x[2:].reshape(-1)
    ball_speed = np.linalg.norm(ball_vel)

    # --- Shot Detection Logic ---
    if shot_candidate and (frame_id - shot_candidate['frame_id'] > 90): # If goal not confirmed in ~3 secs, cancel shot
        print(f"[SHOT MISSED] Shot by {shot_candidate['shooter_id']} at frame {shot_candidate['frame_id']} was not a goal.")
        shot_candidate = None

    if not shot_candidate and ball_speed > SHOT_SPEED_THRESHOLD and (frame_id - last_shot_frame) > SHOT_COOLDOWN_FRAMES:
        min_dist = float('inf')
        potential_shooter = None

        # Identify the shooter based on proximity right before the kick
        for trk in tracked_players:
            x1, y1, x2, y2, pid = trk
            player_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            dist = np.linalg.norm(prev_ball_pos - player_center)
            
            if dist < min_dist and dist < PLAYER_PROXIMITY_THRESHOLD:
                min_dist = dist
                potential_shooter = {
                    "id": int(pid),
                    "box": (int(x1), int(y1), int(x2), int(y2))
                }

        if potential_shooter:
            # --- Goal Direction Filter ---
            # Predict ball's position a few frames into the future
            predicted_pos = ball_pos + ball_vel * 15 # Predict ~0.5 seconds ahead
            if check_if_goal(predicted_pos, GOAL_AREA):
                shooter_id = potential_shooter["id"]
                shooter_box = potential_shooter["box"]
                
                # Identify team if not already known for this track ID
                if shooter_id not in team_identities:
                    x1, y1, x2, y2 = shooter_box
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        team_identities[shooter_id] = get_team_from_crop(crop)

                team = team_identities.get(shooter_id, "Unknown")

                # Get Jersey Number
                x1, y1, x2, y2 = shooter_box
                crop = frame[y1:y2, x1:x2]
                jersey_number = "?"
                if crop.size > 0:
                    processed_crop = preprocess_for_ocr(crop)
                    ocr_results = ocr_reader.readtext(processed_crop)
                    for (_, text, prob) in ocr_results:
                        if text.isdigit():
                            jersey_number = text
                            break
                
                shot_candidate = {
                    "frame_id": frame_id,
                    "shooter_id": shooter_id,
                    "ball_pos": ball_pos,
                    "jersey": jersey_number,
                    "team": team,
                }
                print(f"[SHOT DETECTED] Frame: {frame_id}, Shooter: {shooter_id}, Team: {team}, Jersey: {jersey_number}")
                last_shot_frame = frame_id


    # --- Goal Confirmation Logic ---
    if shot_candidate and check_if_goal(ball_pos, GOAL_AREA):
        print(f"[GOAL!] Frame: {frame_id}. Scored by Player {shot_candidate['shooter_id']}")
        csv_writer.writerow([
            shot_candidate['frame_id'], 
            int(shot_candidate['ball_pos'][0]), 
            int(shot_candidate['ball_pos'][1]),
            shot_candidate['jersey'],
            shot_candidate['team'],
            True # is_goal
        ])
        shot_candidate = None # Reset after confirming goal


    # --- Visualization ---
    cv2.rectangle(vis_frame, (GOAL_AREA[0], GOAL_AREA[1]), (GOAL_AREA[2], GOAL_AREA[3]), (0, 255, 255), 2)
    cv2.putText(vis_frame, "GOAL AREA", (GOAL_AREA[0], GOAL_AREA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if ball_tracked:
        cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 7, (0, 255, 255), -1)

    for trk in tracked_players:
        x1, y1, x2, y2, pid = trk
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        team = team_identities.get(int(pid), "Unknown")
        color = (0, 255, 0) if "Team A" in team else (255, 0, 0) if "Team B" in team else (255, 255, 255)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame, f"ID:{int(pid)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Shot Detector", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Handle any remaining shot candidates that didn't result in a goal
if shot_candidate:
    print(f"[SHOT MISSED] Shot by {shot_candidate['shooter_id']} at frame {shot_candidate['frame_id']} was not a goal.")
    csv_writer.writerow([
        shot_candidate['frame_id'], 
        int(shot_candidate['ball_pos'][0]), 
        int(shot_candidate['ball_pos'][1]),
        shot_candidate['jersey'],
        shot_candidate['team'],
        False # is_goal
    ])

cap.release()
output_csv.close()
cv2.destroyAllWindows()
