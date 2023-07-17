import argparse
import logging
import os

import cv2
import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
model_path = os.path.join(script_dir, "models", "keypoint.pt")
TARGET_CLASS = 0
THR_PIXEL = 30

def get_video_paths(path):
    video_extensions = [".MP4", ".mp4", ".avi", ".mkv", ".MOV"]
    video_paths = []

    if os.path.isfile(path):
        if any(path.endswith(ext) for ext in video_extensions):
            video_paths.append(path)
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                if any(file.endswith(ext) for ext in video_extensions):
                    video_paths.append(os.path.join(root, file))
    return video_paths


def extract_frames_from_video(input, output) -> None:
    video_paths = get_video_paths(input)

    model = YOLO(model_path)
    print(model.info(verbose=True))

    for video_path in video_paths:
        logging.info(f"Processing {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output, video_name)
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        count = 0
        frame_idx = 0
        pos_curr = np.array([0, 0])
        before_not_detected_flag = False
        while True:
            ret, frame = cap.read()
            if ret:
                count += 1
                print(f"\r frame: {count}", end="")
                results = model(frame, verbose=False)

                boxes = results[0].cpu().numpy().boxes
                keypoints = results[0].cpu().numpy().keypoints
                target_idxs = [
                    i for i, c in enumerate(boxes.cls) if c == TARGET_CLASS
                ]

                if len(target_idxs) > 0:
                    before_not_detected_flag = False
                    idx = np.argmax(boxes.conf[target_idxs])

                    box_x, box_y, box_width, box_height = boxes.xywhn[idx]
                    toe, hosel, grip = keypoints.xy[idx]

                    if np.linalg.norm(toe - pos_curr) > THR_PIXEL:
                        pos_curr = toe
                        frame_idx += 1
                        frame_path = os.path.join(
                            output_dir, f"{video_name}_{frame_idx}.jpg"
                        )
                        cv2.imwrite(frame_path, frame)
                else:
                    if not before_not_detected_flag:
                        frame_idx += 1
                        frame_path = os.path.join(
                            output_dir, f"{video_name}_{frame_idx}.jpg"
                        )
                        cv2.imwrite(frame_path, frame)
                        before_not_detected_flag = True
                    else:
                        pass
            else:
                print("")
                logging.info(f"Finished processing {video_path}: ")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, help="Input video path or directory", required=True
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output directory", required=True
    )
    args = parser.parse_args()
    extract_frames_from_video(args.input, args.output)