from ppq.api import QuantizationSettingFactory,TargetPlatform, NetworkFramework

# modify configuration below:
TARGET_PLATFORM = TargetPlatform.TRT_INT8
MODEL_TYPE = NetworkFramework.ONNX
TRAINING_YOUR_NETWORK = True

# -------------------------------------------------------------------
# SETTING 对象用于控制 PPQ 的量化逻辑，主要描述了图融合逻辑、调度方案、量化细节策略等
# 当你的网络量化误差过高时，你需要修改 SETTING 对象中的属性来进行特定的优化
# -------------------------------------------------------------------
QS = QuantizationSettingFactory.trt_setting()

# -------------------------------------------------------------------
# 下面向你展示了如何使用 finetuning 过程提升量化精度
# 在 PPQ 中我们提供了十余种算法用来帮助你恢复精度
# 开启他们的方式都是 QS.xxxx = True
# 按需使用，不要全部打开，容易起飞
# -------------------------------------------------------------------
if TRAINING_YOUR_NETWORK:
    QS.lsq_optimization = True                                      # 启动网络再训练过程，降低量化误差
    QS.lsq_optimization_setting.steps = 500                         # 再训练步数，影响训练时间，500 步大概几分钟
    QS.lsq_optimization_setting.collecting_device = 'cuda'          # 缓存数据放在那，cuda 就是放在gpu，如果显存超了你就换成 'cpu'

# -------------------------------------------------------------------
# 你可以把量化很糟糕的算子送回 FP32
# 当然你要先确认你的硬件支持 fp32 的执行
# 你可以使用 layerwise_error_analyse 来找出那些算子量化的很糟糕
# -------------------------------------------------------------------
QS.dispatching_table.append(operation='OP NAME', platform=TargetPlatform.FP32)
