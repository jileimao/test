DATA='../../../data'
# DATA='/data/hdd01/aidata2/users/maojl/data/DAIR-V2X-Examples/cooperative-vehicle-infrastructure'
OUTPUT="../cache/vic-early-lidar"
INFRA_MODEL_PATH=""
INFRA_CONFIG_NAME="config.py"
INFRA_MODEL_NAME=""

VEHICLE_MODEL_PATH='/data/hdd01/aidata2/users/maojl/MMDetection3D/v2x_work_dir/work_dirs_all_fusion_encoder_40000_20epoch_4gpu_new_filter/latest.pth'
VEHICLE_CONFIG_NAME='/data/hdd01/aidata2/users/maojl/MMDetection3D/v2x_work_dir/work_dirs_all_fusion_encoder_40000_20epoch_4gpu_new_filter/config_pointpillar.py'
# VEHICLE_MODEL_PATH='/data/hdd01/aidata2/users/maojl/MMDetection3D/v2x_work_dir/work_dirs_all_fusion_encoder_40000_20epoch_2gpu_new/latest.pth'
# VEHICLE_CONFIG_NAME='/data/hdd01/aidata2/users/maojl/MMDetection3D/v2x_work_dir/work_dirs_all_fusion_encoder_40000_20epoch_2gpu_new/config_pointpillar.py'
VEHICLE_MODEL_PATH='/data/hdd01/aidata2/users/maojl/MMDetection3D/v2x_work_dir/work_dirs_all_fusion_20epoch_5_18/latest.pth'
VEHICLE_CONFIG_NAME='/data/hdd01/aidata2/users/maojl/MMDetection3D/v2x_work_dir/work_dirs_all_fusion_20epoch_5_18/config_pointpillar.py'
CUDA_VISIBLE_DEVICES=$1 \
# train  val  test valtest
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model fature_fusion \
  --dataset vic-sync \
  --split val \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device $1 \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1 