#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import errno
import numpy as np
import argparse
import errno
import math

parser = argparse.ArgumentParser("v2xdataset")
parser.add_argument(
    "--dir_dataset_for_training",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-Examples-for-training/single-vehicle-side-point-cloud",
    help="location of the saved training dir",
)
args = parser.parse_args()
path_file = os.path.join(args.dir_dataset_for_training, "/training/label_2")


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
            if os.path.splitext(filename)[1] == extention:
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def rewrite_txt(path):
    with open(path, "r+") as f:
        data = f.readlines()
        find_str1 = "Truck"
        find_str2 = "Van"
        find_str3 = "Bus"
        replace_str = "Car"
        new_data = ""
        for line in data:
            if find_str1 in line:
                line = line.replace(find_str1, replace_str)
            if find_str2 in line:
                line = line.replace(find_str2, replace_str)
            if find_str3 in line:
                line = line.replace(find_str3, replace_str)
            new_data = new_data + line
    os.remove(path)
    f_new = open(path, "w")
    f_new.write(new_data)
    f_new.close()


# In[2]:


def rewrite_label(path_file):
    path_list = get_files_path(path_file, ".txt")
    for path in path_list:
        rewrite_txt(path)
        print(path)


rewrite_label(path_file)
