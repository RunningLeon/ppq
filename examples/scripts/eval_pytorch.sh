mmdeploy_dir=../mmdeploy
deploy_cfg=examples/configs/deploy/trt-fp32.py

model_cfg=examples/configs/model/fcn_r50-d8_512x1024_80k_cityscapes.py
checkpoint=examples/checkpoints/fcn_r50-d8_512x1024_80k_cityscapes_20200606_113019-03aa804d.pth
json_file=../workdir/fcn/pt.json

python $mmdeploy_dir/tools/test.py \
  $deploy_cfg \
  $model_cfg \
  --model $checkpoint \
  --json-file $json_file\
  --device cuda \
  --metrics mIoU
