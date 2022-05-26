# DATA="/home/DAIR/maojl/data/DAIR-V2X-Examples/cooperative-vehicle-infrastructure"
# DATA="/home/DAIR/maojl/data/cooperative-vehicle-infrastructure"
DATA='/data/hdd01/aidata2/users/maojl/data/cooperative-vehicle-infrastructure'
OUTPUT="../cache/vic-early-lidar"
INFRA_MODEL_PATH="/home/DAIR/yuhb/cvpr2022/late-fusion/0326/pointpillars/i"
INFRA_CONFIG_NAME="config.py"
INFRA_MODEL_NAME="work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth"
# VEHICLE_MODEL_PATH="/home/DAIR/luoyz/dair-v2x/checkpoints/models-from-yiyi-1220/pointpillar/v/epoch_80.pth"
# VEHICLE_CONFIG_NAME="/home/DAIR/luoyz/dair-v2x/checkpoints/models-from-yiyi-1220/pointpillar/v/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"
# VEHICLE_MODEL_PATH='/home/DAIR/tangyj/work_file/0327/dair-v2x-40-0-100/checkpoints/yingjuan-40/v/epoch_40.pth'
# VEHICLE_MODEL_PATH="/home/DAIR/tangyj/work_file/0326/train_c_v-new-40/work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/epoch_40.pth"
# VEHICLE_CONFIG_NAME="/home/DAIR/tangyj/work_file/0326/train_c_v-new-40/work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py"
# VEHICLE_CONFIG_NAME='/home/DAIR/tangyj/work_file/0327/dair-v2x-40-0-100/checkpoints/yingjuan-40/v/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py'

VEHICLE_MODEL_PATH=/'data/hdd01/aidata2/users/tangyingjuan/0518/92range/epoch_40.pth'
VEHICLE_CONFIG_NAME='/data/hdd01/aidata2/users/tangyingjuan/0518/92range/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car.py'
CUDA_VISIBLE_DEVICES=$1 \
# train  val  test valtest
 python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model early_fusion \
  --dataset vic-sync \
  --split val \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device 0 \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1 