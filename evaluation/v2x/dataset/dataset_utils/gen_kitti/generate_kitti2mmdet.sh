set -ex
export PYTHONPATH=`pwd`:$PYTHONPATH

dir_dataset='/home/yuhaibao/Desktop/Baseline-code/Feature-Fusion/dataset/cooperative-vehicle-infrastructure/infrastructure-side'
dir_dataset_for_training='/home/yuhaibao/Desktop/Baseline-code/Feature-Fusion/mmdetection3d/datasets/cooperative-infrastructure-side-label-infrastructure'
split_json_path='/home/yuhaibao/Desktop/Baseline-code/Feature-Fusion/dataset/cooperative-vehicle-infrastructure/split_data.json'
my_label_type='lidar' # 'camera' only for mono3d, 'lidar' for vehicle lidar
sensor_view='infrastructure' # 'vehicle', 'infrastructure'
temp_write_path='./tmp_file'

mkdir -p $dir_dataset_for_training/training/calib
mkdir -p $dir_dataset_for_training/training/label_2
mkdir -p $dir_dataset_for_training/testing
mkdir -p $dir_dataset_for_training/ImageSets
cp -r $dir_dataset/image $dir_dataset_for_training/training/image_2
cp -r $dir_dataset/velodyne $dir_dataset_for_training/training

python pcd_2_bin.py \
  --dir_dataset_for_training $dir_dataset_for_training

mkdir -p $temp_write_path
rm -rf $temp_write_path/*
python convert_point_from_lidar_to_cam.py \
  --read_path $dir_dataset \
  --write_path $temp_write_path \
  --my_label_type ${my_label_type}
python label_json2kitti.py \
  --temp_write_path $temp_write_path \
  --dir_dataset_for_training $dir_dataset_for_training \
  --my_label_type $my_label_type
python rewrite_label.py \
  --dir_dataset_for_training $dir_dataset_for_training
python label_filter.py \
  --dir_dataset_for_training $dir_dataset_for_training
rm -r $temp_write_path

python step4_gen_calib2kitti.py \
  --dir_dataset ${dir_dataset} \
  --dir_dataset_for_training $dir_dataset_for_training \
  --sensor_view=${sensor_view}

python get_ImageSets_from_split_data.py \
    --write_root $dir_dataset_for_training \
    --split_json_path ${split_json_path} \
    --sensor_view ${sensor_view}

python -u /home/yuhaibao/Desktop/Baseline-code/mm-openlab/mmdetection3d/tools/create_data.py kitti \
        --root-path ${dir_dataset_for_training} \
        --out-dir ${dir_dataset_for_training}