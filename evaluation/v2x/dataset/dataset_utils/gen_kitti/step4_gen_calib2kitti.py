#!/usr/bin/env python
# coding: utf-8
import os
import json
import errno
import numpy as np
import argparse

parser = argparse.ArgumentParser("v2xdataset")
parser.add_argument(
    "--dir_dataset",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-Examples/single-vehicle-side",
    help="location of dataset path",
)
parser.add_argument(
    "--dir_dataset_for_training",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-Examples-for-training/single-vehicle-side-point-cloud",
    help="location of the saved training dir",
)
parser.add_argument("--sensor_view", type=str, default="vehicle", help="senser views: vehicle, infrastructure")
args = parser.parse_args()


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
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
        for filename in filenames:
            if os.path.splitext(filename)[1] == extention:
                path_list.append(os.path.join(dirpath, filename))
    return path_list


def convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam):
    P2 = np.zeros([3, 4])
    P2[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")
    P2 = P2.reshape(12, order="C")

    Tr_velo_to_cam = np.concatenate((r_velo2cam, t_velo2cam), axis=1)
    Tr_velo_to_cam = Tr_velo_to_cam.reshape(12, order="C")

    return P2, Tr_velo_to_cam


def get_cam_D_and_cam_K(path):
    my_json = read_json(path)
    cam_D = my_json["cam_D"]
    cam_K = my_json["cam_K"]
    return cam_D, cam_K


def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = my_json["translation"]
    r_velo2cam = my_json["rotation"]
    return t_velo2cam, r_velo2cam


def gen_calib2kitti(path_camera_intrisinc, path_lidar_to_camera, path_calib):
    """
    input:
    + path_camera_intrisinc: camera_intrisinc的文件夹路径
    + path_lidar_to_camera: lidar_to_camera的文件夹路径
    + path_calib: 最终生成的calib的文件夹路径
    output：
    + 符合kitti格式的calib在path_calib文件夹中生成
    """

    path_list_camera_intrisinc = get_files_path(path_camera_intrisinc, ".json")
    path_list_lidar_to_camera = get_files_path(path_lidar_to_camera, ".json")
    path_list_camera_intrisinc.sort()
    path_list_lidar_to_camera.sort()
    print(len(path_list_camera_intrisinc), len(path_list_lidar_to_camera))
    mkdir_p(path_calib)

    for i in range(len(path_list_camera_intrisinc)):
        cam_D, cam_K = get_cam_D_and_cam_K(path_list_camera_intrisinc[i])
        t_velo2cam, r_velo2cam = get_velo2cam(path_list_lidar_to_camera[i])
        json_name = os.path.split(path_list_camera_intrisinc[i])[-1][:-5] + ".txt"
        json_path = os.path.join(path_calib, json_name)

        t_velo2cam = np.array(t_velo2cam).reshape(3, 1)
        r_velo2cam = np.array(r_velo2cam).reshape(3, 3)
        P2, Tr_velo_to_cam = convert_calib_v2x_to_kitti(cam_D, cam_K, t_velo2cam, r_velo2cam)

        str_P2 = "P2: "
        str_Tr_velo_to_cam = "Tr_velo_to_cam: "
        for ii in range(11):
            str_P2 = str_P2 + str(P2[ii]) + " "
            str_Tr_velo_to_cam = str_Tr_velo_to_cam + str(Tr_velo_to_cam[ii]) + " "
        str_P2 = str_P2 + str(P2[11])
        str_Tr_velo_to_cam = str_Tr_velo_to_cam + str(Tr_velo_to_cam[11])

        str_P0 = str_P2
        str_P1 = str_P2
        str_P3 = str_P2
        str_R0_rect = "R0_rect: 1 0 0 0 1 0 0 0 1"
        str_Tr_imu_to_velo = str_Tr_velo_to_cam

        with open(json_path, "w") as fp:
            gt_line = (
                str_P0
                + "\n"
                + str_P1
                + "\n"
                + str_P2
                + "\n"
                + str_P3
                + "\n"
                + str_R0_rect
                + "\n"
                + str_Tr_velo_to_cam
                + "\n"
                + str_Tr_imu_to_velo
            )
            fp.write(gt_line)


if __name__ == "__main__":
    path_camera_intrinsic = os.path.join(args.dir_dataset, "calib/camera_intrinsic")
    if args.sensor_view == "vehicle":
        path_lidar_to_camera = os.path.join(args.dir_dataset, "calib/lidar_to_camera")
    else:
        path_lidar_to_camera = os.path.join(args.dir_dataset, "calib/virtuallidar_to_camera")
    path_calib = os.path.join(args.dir_dataset_for_training, "training/calib")
    gen_calib2kitti(path_camera_intrinsic, path_lidar_to_camera, path_calib)
