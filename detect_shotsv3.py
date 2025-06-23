import cv2
import numpy as np
import csv
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
import easyocr
from appearance_tracker import AppearanceTracker # <-- Using the NEW appearance-based tracker

# --- Configuration ---
VIDEO_PATH = "match.mp4"
OUTPUT_CSV_PATH = "shots_output_v3.csv"
MODEL_PATH = "yolov8n.pt"

# --- Scene Detection Parameters ---
MIN_PLAYERS_FOR_GAMEPLAY = 4  # Min players on screen to be considered live gameplay
PITCH_PERCENT_THRESHOLD = 30  # Min percentage of the screen that must be green (the pitch)

# --- Shot Detection Parameters ---
SHOT_SPEED_THRESHOLD = 25
SHOT_COOLDOWN_FRAMES = 30
PLAYER_PROXIMITY_THRESHOLD = 150

# --- Team Color Definitions (HSV) ---
TEAM_A_LOWER = np.array([100, 80, 50])
TEAM_A_UPPER = np.array([130, 255, 255])
TEAM_B_LOWER = np.array([0, 0, 200])
TEAM_B_UPPER = np.array([180, 60, 255])
PITCH_GREEN_LOWER = np.array([35, 40, 40])
PITCH_GREEN_UPPER = np.array([85, 255, 255])

# --- Initialization ---
model = YOLO(MODEL_PATH)
ocr_reader = easyocr.Reader(["en"])
tracker = AppearanceTracker(iou_threshold=0.4, max_lost=20, appearance_weight=0.6)
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
team_identities = {}
goal_area = None

# --- CSV Output Setup ---
output_csv = open(OUTPUT_CSV_PATH, "w", newline="")
csv_writer = csv.writer(output_csv)
csv_writer.writerow(["frame", "ball_x", "ball_y", "jersey_number", "team", "is_goal"])


def get_team_from_crop(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_a = cv2.inRange(hsv, TEAM_A_LOWER, TEAM_A_UPPER)
    mask_b = cv2.inRange(hsv, TEAM_B_LOWER, TEAM_B_UPPER)
    count_a = cv2.countNonZero(mask_a)
    count_b = cv2.countNonZero(mask_b)
    return "Team A (Blue)" if count_a > count_b else "Team B (White)" if count_b > count_a else "Unknown"

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def check_if_goal(ball_position, goal_rect):
    if goal_rect is None:
        return False
    x, y = ball_position
    return goal_rect[0] < x < goal_rect[2] and goal_rect[1] < y < goal_rect[3]

def is_gameplay_scene(frame, player_boxes):
    """Determines if the current frame is active gameplay."""
    if len(player_boxes) < MIN_PLAYERS_FOR_GAMEPLAY:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pitch_mask = cv2.inRange(hsv, PITCH_GREEN_LOWER, PITCH_GREEN_UPPER)
    pitch_pixels = cv2.countNonZero(pitch_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    pitch_percentage = (pitch_pixels / total_pixels) * 100

    return pitch_percentage > PITCH_PERCENT_THRESHOLD

def detect_goalposts(frame):
    """Detects goalposts using line detection and returns a bounding box."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Focus on the upper half of the screen where goals usually are
    edges[:frame.shape[0]//2, :] = 0

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20)
    
    if lines is None:
        return None

    vertical_posts = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if 85 < angle < 95:  # Filter for near-vertical lines
            # Check if the line is predominantly white
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.line(mask, (x1,y1), (x2,y2), 255, 3)
            mean_val = cv2.mean(gray, mask=mask)[0]
            if mean_val > 180: # Check for brightness (white posts)
                 vertical_posts.append(line[0])

    if len(vertical_posts) < 2:
        return None

    # Find two posts that are reasonably far apart
    vertical_posts.sort(key=lambda p: p[0])
    best_pair = None
    max_dist = 0

    for i in range(len(vertical_posts)):
        for j in range(i + 1, len(vertical_posts)):
            p1 = vertical_posts[i]
            p2 = vertical_posts[j]
            dist = abs(p1[0] - p2[0])
            # Heuristic: distance between posts should be significant
            if 200 < dist < frame.shape[1] * 0.8:
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (p1, p2)
    
    if not best_pair:
        return None

    post1, post2 = best_pair
    x1 = min(post1[0], post2[0])
    x2 = max(post1[0], post2[0])
    y1 = min(post1[1], post1[3], post2[1], post2[3])
    y2 = max(post1[1], post1[3], post2[1], post2[3])

    return [x1, y1, x2, y2]


# --- Main Loop ---
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
    player_boxes = [box for box, cls in zip(boxes, results.boxes.cls.cpu().numpy().astype(int)) if results.names[cls] == "person"]
    ball_box = next((box for box, cls in zip(boxes, classes) if results.names[cls] == "sports ball"), None)

    # --- Scene Analysis ---
    if not is_gameplay_scene(vis_frame, player_boxes):
        tracker.reset() # Reset tracker during non-gameplay to avoid stale tracks
        goal_area = None # Goal is not visible
        cv2.putText(vis_frame, "SCENE: NON-GAMEPLAY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Shot Detector", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue
    
    # --- Dynamic Goal Detection ---
    detected_goal = detect_goalposts(vis_frame)
    if detected_goal is not None:
        goal_area = detected_goal # Update goal area if detected

    # --- Player and Ball Tracking ---
    tracked_players = tracker.update(np.array(player_boxes), vis_frame)
    prev_ball_pos = kf.x[:2].reshape(-1) if ball_tracked else None

    if ball_box is not None:
        ball_center = np.array([(ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2])
        if not ball_tracked:
            kf.x[:2] = ball_center.reshape((2, 1))
            ball_tracked = True
        else:
            kf.predict(); kf.update(ball_center)
    elif ball_tracked:
        kf.predict()

    ball_pos = kf.x[:2].reshape(-1)
    ball_vel = kf.x[2:].reshape(-1)
    ball_speed = np.linalg.norm(ball_vel)
    
    # --- Shot and Goal Logic (only runs if a goal is visible) ---
    if goal_area:
        # --- Shot Detection Logic ---
        if not shot_candidate and ball_speed > SHOT_SPEED_THRESHOLD and (frame_id - last_shot_frame) > SHOT_COOLDOWN_FRAMES:
            # ... (rest of shot detection logic)
            min_dist = float('inf')
            potential_shooter = None
            if prev_ball_pos is not None:
                 for trk in tracked_players:
                    x1, y1, x2, y2, pid = trk
                    player_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    dist = np.linalg.norm(prev_ball_pos - player_center)
                    if dist < min_dist and dist < PLAYER_PROXIMITY_THRESHOLD:
                        min_dist = dist; potential_shooter = {"id": int(pid), "box": (int(x1), int(y1), int(x2), int(y2))}
            
            if potential_shooter:
                predicted_pos = ball_pos + ball_vel * 15 # Check trajectory towards goal
                if check_if_goal(predicted_pos, goal_area):
                    shooter_id = potential_shooter["id"]; shooter_box = potential_shooter["box"]
                    if shooter_id not in team_identities: team_identities[shooter_id] = get_team_from_crop(frame[shooter_box[1]:shooter_box[3], shooter_box[0]:shooter_box[2]])
                    team = team_identities.get(shooter_id, "Unknown")
                    
                    # ... (OCR logic) ...
                    crop = frame[shooter_box[1]:shooter_box[3], shooter_box[0]:shooter_box[2]]
                    jersey_number = "?"
                    if crop.size > 0:
                        ocr_results = ocr_reader.readtext(preprocess_for_ocr(crop))
                        for (_, text, prob) in ocr_results:
                            if text.isdigit(): jersey_number = text; break
                    
                    shot_candidate = {"frame_id": frame_id, "shooter_id": shooter_id, "ball_pos": ball_pos, "jersey": jersey_number, "team": team}
                    print(f"[SHOT DETECTED] Frame: {frame_id}, Shooter: {shooter_id}, Team: {team}, Jersey: {jersey_number}")
                    last_shot_frame = frame_id

        # --- Goal Confirmation Logic ---
        if shot_candidate and check_if_goal(ball_pos, goal_area):
            print(f"[GOAL!] Frame: {frame_id}. Scored by Player {shot_candidate['shooter_id']}")
            csv_writer.writerow([shot_candidate['frame_id'], int(shot_candidate['ball_pos'][0]), int(shot_candidate['ball_pos'][1]), shot_candidate['jersey'], shot_candidate['team'], True])
            shot_candidate = None
        
        # --- Handle missed shots
        if shot_candidate and (frame_id - shot_candidate['frame_id'] > 90):
            print(f"[SHOT MISSED] Shot by {shot_candidate['shooter_id']} at frame {shot_candidate['frame_id']} was not a goal.")
            csv_writer.writerow([shot_candidate['frame_id'], int(shot_candidate['ball_pos'][0]), int(shot_candidate['ball_pos'][1]), shot_candidate['jersey'], shot_candidate['team'], False])
            shot_candidate = None

    # --- Visualization ---
    if goal_area:
        cv2.rectangle(vis_frame, (goal_area[0], goal_area[1]), (goal_area[2], goal_area[3]), (0, 255, 255), 2)
        cv2.putText(vis_frame, "GOAL DETECTED", (goal_area[0], goal_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if ball_tracked: cv2.circle(vis_frame, (int(ball_pos[0]), int(ball_pos[1])), 7, (0, 255, 255), -1)

    for trk in tracked_players:
        x1, y1, x2, y2, pid = trk.astype(int)
        team = team_identities.get(pid, "Unknown")
        color = (0, 255, 0) if "Team A" in team else (255, 0, 0) if "Team B" in team else (255, 255, 255)
        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame, f"ID:{pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Shot Detector", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
output_csv.close()
cv2.destroyAllWindows()
