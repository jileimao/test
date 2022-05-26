DATA='../../data/'
OUTPUT="../cache/vic-early-lidar"
INFRA_MODEL_PATH=""
INFRA_CONFIG_NAME="config.py"
INFRA_MODEL_NAME=""
# VEHICLE_MODEL_PATH='/MMDetection3D/feature_flow_work_dir/checkpoints/train_all_3epochs_with_encoder.pth'
VEHICLE_MODEL_PATH='/MMDetection3D/feature_flow_work_dir/checkpoints/train_all_2.pth'
VEHICLE_CONFIG_NAME='/MMDetection3D/feature_flow_work_dir/config_pointpillar.py'
CUDA_VISIBLE_DEVICES=$1 \
# train  val  test valtest
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model feature_flow \
  --dataset vic-sync \
  --split valtest \
  --inf-config-path $INFRA_MODEL_PATH/$INFRA_CONFIG_NAME \
  --inf-model-path $INFRA_MODEL_PATH/$INFRA_MODEL_NAME \
  --veh-config-path $VEHICLE_CONFIG_NAME \
  --veh-model-path $VEHICLE_MODEL_PATH \
  --device $1 \
  --pred-class car \
  --sensortype lidar \
  --extended-range 0 -39.68 -3 100 39.68 1 