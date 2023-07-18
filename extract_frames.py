import os
import cv2
import argparse

EXTRA_TIME = 0.5

def get_basename(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def save_frame(frame, output_dir, frame_index):
    frame_name = f"frame_{frame_index:04d}.jpg"
    output_path = os.path.join(output_dir, frame_name)
    cv2.imwrite(output_path, frame)

def extract_frames(input_path, output_dir, start_time=0, end_time=None, rotate_direction="none"):
    video_name = get_basename(input_path)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    dt = 1 / fps
    now_time = 0
    frame_index = 0

    if end_time is None:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_time = total_frames / fps + EXTRA_TIME
    
    output_dir = os.path.join(output_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if ret:
            if start_time <= now_time and now_time <= end_time:
                if rotate_direction == "right":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotate_direction == "left":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                save_frame(frame, output_dir, frame_index)
                frame_index += 1
            elif end_time < now_time:
                break

            now_time += dt
        else:
            break

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video frame extraction script")
    parser.add_argument("-i", "--input", type=str, help="Input video path", required=True)
    parser.add_argument("-o", "--output", type=str, help="Output directory", required=True)
    parser.add_argument("-s", "--start", type=float, default=0, help="Start time (in seconds)")
    parser.add_argument("-e", "--end", type=float, default=None, help="End time (in seconds)")
    parser.add_argument(
        "-r",
        "--rotate",
        type=str,
        choices=["right", "left", "none"],
        default="none",
        help="Rotation direction: 'right' for clockwise, 'left' for counterclockwise, 'none' for no rotation (default)",
    )

    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    start_time = args.start
    end_time = args.end
    rotate_direction = args.rotate

    extract_frames(input_path, output_dir, start_time, end_time, rotate_direction)
