#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import errno
import numpy as np
import pandas as pd
import argparse
import errno
import math

parser = argparse.ArgumentParser("v2xdataset")
parser.add_argument(
    "--temp_write_path",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-temp/single-vehicle-side",
    help="location of temperate write path",
)
parser.add_argument(
    "--dir_dataset_for_training",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-Examples-for-training/single-vehicle-side-point-cloud",
    help="location of the saved training dir",
)
parser.add_argument("--my_label_type", type=str, default="lidar", help="lidar for pointpillar，camera for mono3d")
args = parser.parse_args()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_json(path):
    with open(path, "r") as f:
        my_json = json.load(f)
        return my_json


def get_files_path(path_my_dir, extention):
    """
    function description:
    根据路径找寻匹配的文件路径
    input:
    + path_my_dir: 原文件的文件路径
    + extention: 寻找文件的扩展名, 例如`.json`
    output：
    匹配的文件路径组成的list
    """
    path_list = []
    for (dirpath, dirnames, filenames) in os.walk(path_my_dir):
        # dirpath:当前目录路径，dirnames当前路径下所有子目录，filenames当前路径下所有非目录子文件
        for filename in filenames:
            if os.path.splitext(filename)[1] == ".json":
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def write_kitti_in_txt(my_json, path_txt):
    wf = open(path_txt, "w")
    for item in my_json:
        i1 = str(item["type"]).title()
        i2 = str(item["truncated_state"])
        i3 = str(item["occluded_state"])
        i4 = str(item["alpha"])
        i5, i6, i7, i8 = (
            str(item["2d_box"]["xmin"]),
            str(item["2d_box"]["ymin"]),
            str(item["2d_box"]["xmax"]),
            str(item["2d_box"]["ymax"]),
        )
        i9, i10, i11 = str(item["3d_dimensions"]["h"]), str(item["3d_dimensions"]["w"]), str(item["3d_dimensions"]["l"])
        i12, i13, i14 = str(item["3d_location"]["x"]), str(item["3d_location"]["y"]), str(item["3d_location"]["z"])
        i15 = str(item["rotation"])
        item_list = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15]
        item_string = " ".join(item_list) + "\n"
        # print(item_string)
        wf.write(item_string)
    wf.close()


def json2kitti(json_file, kitti_file):
    """
    function description:
    将label从json格式转换为kitti格式
    input:
    + json_file: json文件的文件路径
    + kitti_file: kitti文件的文件路径
    output：
    在kitti_file中生成同样名字的kitti格式文件
    """
    mkdir_p(kitti_file)
    jsons_path = get_files_path(json_file, ".json")
    for json_path in jsons_path:
        my_json = read_json(json_path)
        name = json_path.split("/")[-1][:-5] + ".txt"
        path_txt = os.path.join(kitti_file, name)
        write_kitti_in_txt(my_json, path_txt)


if __name__ == "__main__":
    json_file = os.path.join(args.temp_write_path, "label", args.my_label_type)
    kitti_file = os.path.join(args.dir_dataset_for_training, "training/label_2")
    json2kitti(json_file, kitti_file)
