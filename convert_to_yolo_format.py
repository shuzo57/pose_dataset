import argparse
import json
import os


def convert_to_yolo_format(json_path, output_dir) -> None:
    with open(json_path) as f:
        coco_data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_data in coco_data["images"]:
        image_id = image_data["id"]
        image_name = image_data["file_name"]
        image_width = image_data["width"]
        image_height = image_data["height"]

        keypoitns_list = []
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_id:
                keypoints = annotation["keypoints"]
                keypoitns_list.append(keypoints)

        if not keypoitns_list:
            continue

        annotation_file_name = os.path.splitext(image_name)[0] + ".txt"
        annotation_file_path = os.path.join(output_dir, annotation_file_name)
        with open(annotation_file_path, "w") as f:
            for keypoints in keypoitns_list:
                x_min = min(keypoints[0::3])
                x_max = max(keypoints[0::3])
                y_min = min(keypoints[1::3])
                y_max = max(keypoints[1::3])

                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height

                f.write(
                    f"{0} {round(x_center, 6)} {round(y_center, 6)} \
                        {round(width, 6)} {round(height, 6)} "
                )

                for i in range(0, len(keypoints), 3):
                    x = round(keypoints[i] / image_width, 6)
                    y = round(keypoints[i + 1] / image_height, 6)
                    v = round(keypoints[i + 2], 6)
                    f.write(f"{x} {y} {v} ")
                f.write("\n")

    print("Convert to YOLO format is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert COCO format JSON to YOLO format"
    )
    parser.add_argument(
        "-j",
        "--json_path",
        required=True,
        help="Path to COCO format JSON file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Output directory for YOLO format files",
    )
    args = parser.parse_args()

    convert_to_yolo_format(args.json_path, args.output_dir)
