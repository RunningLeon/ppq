model=$1
echo $model

python examples/convert_excel.py \
	--files trt-fp32:../workdir/$model/trt/eval_fp32.json \
	trt-fp16:../workdir/$model/trt/eval_fp16.json \
	trt-int8:../workdir/$model/trt/eval_quant128.json \
	default:../workdir/$model/default/eval.json \
	default_dynamic:../workdir/$model/default/eval_dynamic_range.json \
       	--output examples/results/$model.xlsx
