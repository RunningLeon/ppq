#!/bin/sh
set -e
qtype=$1
bash_dir=$(cd `dirname $0`; pwd)
proj_dir=$bash_dir/../..
date_today=`date +'%Y-%m-%d'`

config=$proj_dir/examples/configs/ptq/$qtype.yaml

for model in fcn pspnet stdc bisenetv2;
do
  logdir=$proj_dir/../workdir/$model/$qtype
  mkdir -p $logdir
  python $proj_dir/examples/run.py $config --model $model | tee $logdir/run.log 2>&1
done
