import cv2

def capture_frames_at_intervals(video_path):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    imgs=[]
    num_frames = int(total_frames // fps)
    interval = total_frames // (num_frames - 1) if num_frames > 1 else 0
    frame_count = 0
    for _ in range(num_frames):
        time_position = int(frame_count * interval)
        if time_position==total_frames:
            time_position-=1
        cap.set(cv2.CAP_PROP_POS_FRAMES, time_position)

        ret, frame = cap.read()

        frame_count += 1
        imgs.append(frame)

    cap.release()
    return imgs
