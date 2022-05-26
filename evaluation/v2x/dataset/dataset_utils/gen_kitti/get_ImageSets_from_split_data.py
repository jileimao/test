#!/usr/bin/env python
# coding: utf-8
import os
import json
import errno
import argparse


def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
    return my_json


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def write_txt(path, file):
    with open(path, "w") as f:
        f.write(file)


def get_ImageSet_from_split_data(ImageSets_path, split_data_path, sensor_view="vehicle"):
    split_data = read_json(split_data_path)
    test_file = ""
    train_file = ""
    val_file = ""

    sensor_view = sensor_view + "_split"
    for i in range(len(split_data[sensor_view]["train"])):
        name = split_data[sensor_view]["train"][i]
        train_file = train_file + name + "\n"

    for i in range(len(split_data[sensor_view]["val"])):
        name = split_data[sensor_view]["val"][i]
        val_file = val_file + name + "\n"

    for i in range(len(split_data[sensor_view]["test"])):
        name = split_data[sensor_view]["test"][i]
        test_file = test_file + name + "\n"

    trainval_file = train_file + val_file

    mkdir_p(ImageSets_path)
    write_txt(os.path.join(ImageSets_path, "test.txt"), test_file)
    write_txt(os.path.join(ImageSets_path, "trainval.txt"), trainval_file)
    write_txt(os.path.join(ImageSets_path, "train.txt"), train_file)
    write_txt(os.path.join(ImageSets_path, "val.txt"), val_file)


parser = argparse.ArgumentParser("v2xdataset")
parser.add_argument(
    "--write_root",
    type=str,
    default="./single-vehicle-side-release-for-training",
    help="location of read training label dir",
)
parser.add_argument(
    "--split_json_path",
    type=str,
    default="./single-vehicle-side-release",
    help="location of the saved training label dir",
)
parser.add_argument("--sensor_view", type=str, default="vehicle", help="senser views: vehicle, infrastructure")
args = parser.parse_args()

if __name__ == "__main__":
    write_root = args.write_root
    split_json_path = args.split_json_path
    sensor_view = args.sensor_view
    ImageSets_path = os.path.join(write_root, "ImageSets")
    get_ImageSet_from_split_data(ImageSets_path, split_json_path, sensor_view)
