# DATA="/data/hdd01/aidata2/users/ad_sharing/datasets/v2x/release_data/DAIR-V2X/cooperative-vehicle-infrastructure/vehicle-side/"
#DATA="/data/hdd01/aidata2/users/ad_sharing/datasets/v2x/release_data/DAIR-V2X-Train-Val/DAIR-V2X-Examples/cooperative-vehicle-infrastructure/vehicle-side/"
DATA="/home/DAIR/maoz/Datasets/dair-v2x/cooperative-vehicle-infrastructure-example/"
OUTPUT="../cache/veh_single"

MODEL_PATH="/home/DAIR/maoz/dev/dair-v2x/checkpoints/joint-model-from-taohua-mono-car-1225"
MODEL_NAME="v"
CONFIG_NAME="vehicle_raw.py"
CKPT_NAME="latest.pth"
sensortype="camera"

#MODEL_PATH="../checkpoints/models-from-yiyi-1220/pointpillar"
#MODEL_NAME="v"
#CONFIG_NAME="hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py"
#CKPT_NAME="epoch_80.pth"
#sensortype="lidar"

rm -r ${OUTPUT}
mkdir -p ${OUTPUT}
mkdir ${OUTPUT}/label
mkdir ${OUTPUT}/result

CUDA_VISIBLE_DEVICES=$1 \
python eval.py \
  --input $DATA \
  --output $OUTPUT \
  --model single_side \
  --dataset dair-v2x-v \
  --split val \
  --config-path $MODEL_PATH/$MODEL_NAME/$CONFIG_NAME \
  --model-path $MODEL_PATH/$MODEL_NAME/$CKPT_NAME \
  --device 0 \
  --pred-class car \
  --extended-range 0 -39.68 -3 69.12 39.68 1 \
  --sensortype ${sensortype} \
  --eval-single