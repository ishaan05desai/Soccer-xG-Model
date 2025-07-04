from utils import read_video, save_video
from trackers import Tracker

def main():
    #Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')

    #Initialize Tracker
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    
    #Save Video
    save_video(video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()