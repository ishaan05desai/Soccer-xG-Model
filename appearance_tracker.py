import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """Computes IoU between two bounding boxes."""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh + 1e-6)
    return o

class AppearanceTracker:
    def __init__(self, iou_threshold=0.5, max_lost=5, appearance_weight=0.5):
        """
        Initializes the tracker.
        iou_threshold: IoU threshold for matching.
        max_lost: Max frames a track can be lost.
        appearance_weight: How much to weigh appearance vs. IoU (0 to 1).
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.appearance_weight = appearance_weight
        self.tracks = []
        self.next_track_id = 0

    def _get_color_histogram(self, image_crop):
        """Calculates a color histogram for a given image crop (player)."""
        if image_crop.size == 0:
            return np.zeros((16,)).astype(np.float32) # Return empty for invalid crops
        
        hsv_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)
        # Use only Hue and Saturation for robustness to lighting changes
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [18, 25], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist.flatten()

    def reset(self):
        """Resets the tracker to its initial state."""
        self.tracks = []
        self.next_track_id = 0

    def update(self, detections, frame):
        """
        Updates the tracker with new detections.
        detections: A numpy array of detections [[x1, y1, x2, y2], ...]
        frame: The full current video frame for extracting appearance.
        Returns a list of tracked bounding boxes [[x1, y1, x2, y2, track_id], ...]
        """
        if not self.tracks:
            for det in detections:
                self.tracks.append(self._create_track(det, frame))
            return self.get_active_tracks()

        num_detections = len(detections)
        num_tracks = len(self.tracks)
        if num_detections == 0 or num_tracks == 0:
            # Handle cases with no tracks or no detections
            if num_tracks > 0:
                for track in self.tracks:
                     track['lost'] += 1
                     if track['lost'] > self.max_lost:
                        track['active'] = False
            self.tracks = [t for t in self.tracks if t['active']]
            return self.get_active_tracks()

        # Create cost matrices for IoU and appearance
        iou_cost_matrix = np.zeros((num_detections, num_tracks))
        appearance_cost_matrix = np.zeros((num_detections, num_tracks))

        detection_features = [self._get_color_histogram(frame[int(d[1]):int(d[3]), int(d[0]):int(d[2])]) for d in detections]

        for i in range(num_detections):
            for j in range(num_tracks):
                track_feature = self.tracks[j]['appearance']
                # IoU cost: 1 - iou (distance)
                iou_cost_matrix[i, j] = 1 - iou(detections[i], self.tracks[j]['bbox'])
                # Appearance cost: Lower is more similar
                appearance_cost_matrix[i, j] = cv2.compareHist(detection_features[i], track_feature, cv2.HISTCMP_CORREL)

        # Combine costs
        combined_cost_matrix = (self.appearance_weight * appearance_cost_matrix) + \
                               ((1 - self.appearance_weight) * iou_cost_matrix)

        det_indices, track_indices = linear_sum_assignment(combined_cost_matrix)

        matched_detections = []
        matched_tracks = []

        # Process matches
        for det_idx, track_idx in zip(det_indices, track_indices):
            if iou_cost_matrix[det_idx, track_idx] < (1 - self.iou_threshold):
                self.tracks[track_idx]['bbox'] = detections[det_idx]
                self.tracks[track_idx]['lost'] = 0
                self.tracks[track_idx]['active'] = True # Ensure it's active
                # Update appearance feature with a moving average for gradual changes
                new_feature = detection_features[det_idx]
                self.tracks[track_idx]['appearance'] = (self.tracks[track_idx]['appearance'] * 0.7) + (new_feature * 0.3)
                matched_detections.append(det_idx)
                matched_tracks.append(track_idx)

        # Handle unmatched tracks and detections
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['lost'] += 1
                if track['lost'] > self.max_lost:
                    track['active'] = False

        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks.append(self._create_track(det, frame))

        self.tracks = [t for t in self.tracks if t['active']]
        return self.get_active_tracks()

    def get_active_tracks(self):
        """Returns the current active tracks."""
        active_tracks_data = []
        for track in self.tracks:
            if track['active']:
                bbox = track['bbox']
                track_id = track['id']
                active_tracks_data.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        return np.array(active_tracks_data) if active_tracks_data else np.empty((0, 5))

    def _create_track(self, detection, frame):
        """Creates a new track from a detection."""
        x1, y1, x2, y2 = map(int, detection)
        appearance_feature = self._get_color_histogram(frame[y1:y2, x1:x2])
        
        track = {
            'id': self.next_track_id,
            'bbox': detection,
            'lost': 0,
            'active': True,
            'appearance': appearance_feature
        }
        self.next_track_id += 1
        return track
