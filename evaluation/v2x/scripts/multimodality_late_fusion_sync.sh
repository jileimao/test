DATA="/data/hdd01/aidata2/users/ad_sharing/datasets/v2x/release_data/DAIR-V2X-Train-Val/DAIR-V2X-Examples/cooperative-vehicle-infrastructure/"
OUTPUT="../cache/coop-multimodality"
INFRA_MODEL_PATH="../checkpoints/joint-model-from-yiyi-0124/mvxnet/i"
INFRA_CONFIG_NAME="dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py"
VEHICLE_MODEL_PATH="../checkpoints/joint-model-from-yiyi-0124/mvxnet/v"
VEHICLE_CONFIG_NAME="dv_mvx-fpn_second_secfpn_adamw_2x8_80e_kitti-3d-3class.py"

CUDA_VISIBLE_DEVICES=$1 \
python eval_vic.py \
  --input $DATA \
  --output $OUTPUT \
  --model late_fusion \
  --dataset vic-sync \
  --split valtest \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/latest.pth \
  --veh-config-path $VEHICLE_MODEL_PATH/$VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH/latest.pth \
  --device 0 \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 69.12 39.68 1 \
  --overwrite-cache
