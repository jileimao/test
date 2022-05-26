import os
import argparse
import math

parser = argparse.ArgumentParser("v2xdataset")
parser.add_argument(
    "--dir_dataset_for_training",
    type=str,
    default="/home/DAIR/huoyy/dair-v2x/DAIR-V2X-Examples-for-training/mvxnet/single-vehicle-side-point-cloud",
    help="location of the saved training dir",
)
args = parser.parse_args()

label_dir = os.path.join(args.dir_dataset_for_training, "training/label_2")

if __name__ == "__main__":
    label_dir = label_dir
    files = os.listdir(label_dir)

    for file in files:
        path = os.path.join(label_dir, file)

        lines_write = []
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                wlh = float(line.split(" ")[9])
                if wlh > 0:
                    lines_write.append(line)

        with open(path, "w") as f:
            f.writelines(lines_write)
