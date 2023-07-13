import os
import argparse
import shutil
import random


random.seed(0)


def create_dataset(images_dir, labels_dir, output_dir, train_ratio) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images_train_dir = os.path.join(output_dir, "images", "train")
    images_val_dir = os.path.join(output_dir, "images", "val")
    labels_train_dir = os.path.join(output_dir, "labels", "train")
    labels_val_dir = os.path.join(output_dir, "labels", "val")

    if not os.path.exists(images_train_dir):
        os.makedirs(images_train_dir)
    if not os.path.exists(images_val_dir):
        os.makedirs(images_val_dir)
    if not os.path.exists(labels_train_dir):
        os.makedirs(labels_train_dir)
    if not os.path.exists(labels_val_dir):
        os.makedirs(labels_val_dir)

    labels = os.listdir(labels_dir)
    random.shuffle(labels)

    train_size = int(len(labels) * train_ratio)
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]

    for label in train_labels:
        label_path = os.path.join(labels_dir, label)
        image_path = os.path.join(
            images_dir, os.path.splitext(label)[0] + ".jpg"
        )
        shutil.copy(image_path, images_train_dir)
        shutil.copy(label_path, labels_train_dir)

    for label in val_labels:
        label_path = os.path.join(labels_dir, label)
        image_path = os.path.join(
            images_dir, os.path.splitext(label)[0] + ".jpg"
        )
        shutil.copy(image_path, images_val_dir)
        shutil.copy(label_path, labels_val_dir)

    print("Create dataset is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset for training")
    parser.add_argument(
        "-i",
        "--images_dir",
        type=str,
        default="data/images",
        help="Path to images directory",
    )
    parser.add_argument(
        "-l",
        "--labels_dir",
        type=str,
        default="data/labels",
        help="Path to labels directory",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data/dataset",
        help="Path to output directory",
    )
    parser.add_argument(
        "-t",
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train ratio",
    )
    args = parser.parse_args()

    create_dataset(
        args.images_dir,
        args.labels_dir,
        args.output_dir,
        args.train_ratio,
    )