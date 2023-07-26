import argparse
import logging
import os

import cv2

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


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


def resize_video(
    input_path, output_dir, resize_rate, rotate_direction=None
) -> None:
    video_paths = get_video_paths(input_path)
    for video_path in video_paths:
        logging.info(f"Processing {video_path}")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        new_width = int(width * resize_rate)
        new_height = int(height * resize_rate)

        output_path = os.path.join(output_dir, f"{video_name}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        if rotate_direction == "right":
            video = cv2.VideoWriter(output_path, fourcc, fps, (height, width))
        elif rotate_direction == "left":
            video = cv2.VideoWriter(output_path, fourcc, fps, (height, width))
        else:
            video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (new_width, new_height))
                if rotate_direction == "right":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotate_direction == "left":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                video.write(frame)
            else:
                print("")
                logging.info(f"Finished processing {video_path}")
                break
        video.release()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input video path or directory",
        required=True,
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Output directory", required=True
    )
    parser.add_argument(
        "--resize",
        type=float,
        help="Resize rate (default is 1.0)",
        default=1.0,
    )
    parser.add_argument(
        "--rotate",
        type=str,
        choices=["right", "left", "none"],
        default="none",
        help="Rotation direction: 'right' for clockwise, 'left' for "
        + "counterclockwise, 'none' for no rotation (default)",
    )
    args = parser.parse_args()
    resize_video(args.input, args.output, args.resize, args.rotate)
