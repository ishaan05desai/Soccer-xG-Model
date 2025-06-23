import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(bb_test, bb_gt):
    """
    Computes IoU between two bboxes in the form [x1,y1,x2,y2]
    """
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

class IouTracker:
    def __init__(self, iou_threshold=0.5, max_lost=5):
        """
        Initializes the tracker.
        iou_threshold: IoU threshold for matching detections to trackers.
        max_lost: Maximum number of consecutive frames a track can be lost.
        """
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.tracks = []
        self.next_track_id = 0

    def update(self, detections):
        """
        Updates the tracker with new detections.
        detections: A numpy array of detections in the format [[x1, y1, x2, y2], ...]
        Returns a list of tracked bounding boxes in the format [[x1, y1, x2, y2, track_id], ...]
        """
        if not self.tracks:
            # Initialize tracks with the first set of detections
            for det in detections:
                self.tracks.append(self._create_track(det))
            return self.get_active_tracks()

        # Create cost matrix
        num_detections = len(detections)
        num_tracks = len(self.tracks)
        cost_matrix = np.zeros((num_detections, num_tracks))

        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                cost_matrix[i, j] = 1 - iou(det, track['bbox'])

        # Use Hungarian algorithm for assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)

        matched_detections = []
        matched_tracks = []

        # Process matches
        for det_idx, track_idx in zip(det_indices, track_indices):
            if cost_matrix[det_idx, track_idx] < (1 - self.iou_threshold):
                self.tracks[track_idx]['bbox'] = detections[det_idx]
                self.tracks[track_idx]['lost'] = 0
                matched_detections.append(det_idx)
                matched_tracks.append(track_idx)

        # Handle unmatched tracks
        for i, track in enumerate(self.tracks):
            if i not in matched_tracks:
                track['lost'] += 1
                if track['lost'] > self.max_lost:
                    self.tracks[i]['active'] = False # Mark as inactive instead of deleting immediately

        # Handle unmatched detections (new tracks)
        for i, det in enumerate(detections):
            if i not in matched_detections:
                self.tracks.append(self._create_track(det))

        # Filter out inactive tracks
        self.tracks = [t for t in self.tracks if t['active']]

        return self.get_active_tracks()
    
    def get_active_tracks(self):
        """Returns the current active tracks."""
        active_tracks = []
        for track in self.tracks:
            if track['active']:
                bbox = track['bbox']
                track_id = track['id']
                active_tracks.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id])
        return np.array(active_tracks)


    def _create_track(self, detection):
        """Creates a new track from a detection."""
        track = {
            'id': self.next_track_id,
            'bbox': detection,
            'lost': 0,
            'active': True
        }
        self.next_track_id += 1
        return track

