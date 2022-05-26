DATA="../data/dair-v2x-c/"
OUTPUT="../cache/vic-late-lidar"
INFRA_MODEL_PATH="/home/DAIR/yuhb/cvpr2022/late-fusion/0326/pointpillars/i"
INFRA_CONFIG_NAME="config.py"
INFRA_MODEL_NAME="work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/latest.pth"
VEHICLE_MODEL_PATH="../checkpoints/models-from-yiyi-1220/pointpillar/v"
VEHICLE_CONFIG_NAME="hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"

CUDA_VISIBLE_DEVICES=$1 \
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model veh_only \
  --dataset vic-sync \
  --split valtest \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/epoch_80.pth \
  --device 0 \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1 \
  --overwrite-cache \