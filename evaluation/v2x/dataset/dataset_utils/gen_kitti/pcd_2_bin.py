from shutil import copyfile
import numpy as np
from pypcd.pypcd import pypcd
import os
import argparse
import math

parser = argparse.ArgumentParser("v2xdataset")
parser.add_argument(
    "--dir_dataset_for_training",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-Examples-for-training/single-vehicle-side-point-cloud",
    help="location of the saved training dir",
)
args = parser.parse_args()

pcd_dir = os.path.join(args.dir_dataset_for_training, "training/velodyne")


def copy_txt_num(source_file, write_path, num):
    for ii in range(num):
        print("copy txt num: ", ii)
        destination_file = write_path + "/" + str(ii).zfill(6) + ".txt"
        copyfile(source_file, destination_file)


def renameFile(filepath):
    fileList = os.listdir(filepath)

    for fileName in fileList:
        os.rename(filepath + "/" + fileName, filepath + "/" + fileName.replace(".jpg", ".png"))


def modify_label_hight_add_half_h(label_dir, write_label_dir):
    fileList = os.listdir(label_dir)
    for fileName in fileList:
        label_path = label_dir + "/" + fileName
        label_txt = open(label_path, "r")
        labels = label_txt.readlines()

        write_label_path = write_label_dir + "/" + fileName
        write_label_txt = open(write_label_path, "w")

        for label in labels:
            if len(label.split(" ")) > 1:
                [
                    obj_class,
                    istruncated,
                    isoccluded,
                    alpha,
                    obj_xmin,
                    obj_ymin,
                    obj_xmax,
                    obj_ymax,
                    dimention_h,
                    dimention_w,
                    dimention_l,
                    location_x,
                    location_y,
                    location_z,
                    rotation_y,
                ] = label.replace("\n", "").split(" ")

                gt_line = (
                    str(obj_class)
                    + " "
                    + str(int(istruncated))
                    + " "
                    + str(int(isoccluded))
                    + " "
                    + str(float(alpha))
                    + " "
                    + str(float(obj_xmin))
                    + " "
                    + str(float(obj_ymin))
                    + " "
                    + str(float(obj_xmax))
                    + " "
                    + str(float(obj_ymax))
                    + " "
                    + str(float(dimention_h))
                    + " "
                    + str(float(dimention_w))
                    + " "
                    + str(float(dimention_l))
                    + " "
                    + str(float(location_x))
                    + " "
                    + str(float(location_y) + float(dimention_h) / 2)
                    + " "
                    + str(float(location_z))
                    + " "
                    + str(float(rotation_y))
                )
                write_label_txt.write("{}\n".format(gt_line))
        label_txt.close()
        write_label_txt.close()


def pcd2bin(pcd_file_path, bin_file_path):
    pc = pypcd.PointCloud.from_path(pcd_file_path)

    np_x = (np.array(pc.pc_data["x"], dtype=np.float32)).astype(np.float32)
    np_y = (np.array(pc.pc_data["y"], dtype=np.float32)).astype(np.float32)
    np_z = (np.array(pc.pc_data["z"], dtype=np.float32)).astype(np.float32)
    np_i = (np.array(pc.pc_data["intensity"], dtype=np.float32)).astype(np.float32) / 255

    points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))
    points_32.tofile(bin_file_path)


if __name__ == "__main__":
    # print('====================== Start Your Show ==============================')
    # source_file = 'tools/000000-yiyi.txt'
    # write_path = '../road-side/road-side-small-kitti-point-cloud/training/calib'
    # copy_txt_num(source_file, write_path, 100)

    # ##########################################################################################################
    # filepath = '../road-side-small-0901/road-side-small-new/training/image_2'
    # renameFile(filepath)

    # # #########################################################################################################
    # label_dir = '../road-side-small-0901/road-side-small-kitti-new-mono3d/training/label_2_lidar'
    # write_label_dir = '../road-side-small-0901/road-side-small-kitti-new-mono3d/training/label_2'
    # modify_label_hight_add_half_h(label_dir, write_label_dir)

    ##########################################################################################################
    pcd_dir = pcd_dir
    fileList = os.listdir(pcd_dir)
    for fileName in fileList:
        if ".pcd" in fileName:
            pcd_file_path = pcd_dir + "/" + fileName
            bin_file_path = pcd_dir + "/" + fileName.replace(".pcd", ".bin")
            pcd2bin(pcd_file_path, bin_file_path)
