from ppq.api import QuantizationSettingFactory,TargetPlatform, NetworkFramework

# modify configuration below:
TARGET_PLATFORM = TargetPlatform.TRT_INT8
MODEL_TYPE = NetworkFramework.ONNX

# -------------------------------------------------------------------
# SETTING 对象用于控制 PPQ 的量化逻辑，主要描述了图融合逻辑、调度方案、量化细节策略等
# 当你的网络量化误差过高时，你需要修改 SETTING 对象中的属性来进行特定的优化
# -------------------------------------------------------------------
QS = QuantizationSettingFactory.trt_setting()
QS.dispatcher = 'pursus'

# QS.graph_format_setting = GraphFormatSetting()
# 有一些平台不支持Constant Op，这个pass会尝试将 Constant Operation 的输入转变为 Parameter Variable
# Some deploy platform does not support constant operation,
# this pass will convert constant operation into parameter variable(if can be done).
QS.graph_format_setting.format_constant_op = True

# 融合Conv和Batchnorm
# Fuse Conv and Batchnorm Layer. This pass is necessary and crucial.
QS.graph_format_setting.fuse_conv_bn = True

# 将所有的parameter variable进行分裂，使得每个variable具有至多一个输出算子
# Split all parameter variables, making all variable has at most 1 output operation.
# This pass is necessary and crucial.
QS.graph_format_setting.format_paramters = True

# 一个你必须要启动的 Pass
# This pass is necessary and crucial.
QS.graph_format_setting.format_cast = True

# 尝试从网络中删除所有与输出无关的算子和 variable
# Remove all unnecessary operations and variables(which are not link to graph output) from current graph,
# notice that some platform use unlinked variable as quantization parameters, do not set this as true if so.
QS.graph_format_setting.delete_isolate = True
# ----------------------------------------------------------------------------------------------
# ssd with loss check equalization 相关设置
# may take longer time(about 30min for default 3 iterations), but guarantee better result than baseline
# should not be followed by a plain equalization when turned on
QS.ssd_equalization = True
# QS.ssd_setting = SSDEqualizationSetting()
# Equalization 优化级别，目前只支持level 1，对应 Conv--Relu--Conv 和 Conv--Conv 的拉平
# optimization level, only support level 1 for now
# you shouldn't modify this
QS.ssd_setting.opt_level = 1

# 在计算scale的时候，所有低于 channel_ratio * max(W) 的值会被裁减到 channel_ratio * max(W)
# channel ratio used to calculate equalization scale
# all values below this ratio of the maximum value of corresponding weight
# will be clipped to this ratio when calculating scale
QS.ssd_setting.channel_ratio = 0.5

# loss的降低阈值，优化后的loss低于 原来的loss * 降低阈值, 优化才会发生
# optimized loss must be below this threshold of original loss for algo to take effect
QS.ssd_setting.loss_threshold = 0.8

# 是否对权重进行正则化
# whether to apply layer normalization to weights
QS.ssd_setting.layer_norm = False

# 算法迭代次数，3次对于大部分网络足够
# num of iterations, 3 would be enough for most networks
# it takes about 10 mins for one iteration
QS.ssd_setting.iteration = 3

# ----------------------------------------------------------------------------------------------
# layer wise equalizition 与相关配置
QS.equalization = False
# QS.equalization_setting = EqualizationSetting()
# Equalization 优化级别，如果选成 1 则不进行多分支拉平，如果选成 2，则进行跨越 add, sub 的多分支拉平
# 不一定哪一个好，你自己试试
# optimization level of layerwise equalization
# 1 - single branch equalization(can not cross add, sub)
# 2 - multi branch equalization(equalization cross add, sub)
# don't know which one is better, try it by yourQS.equalization_setting.
QS.equalization_setting.opt_level = 1

# Equalization 迭代次数，试试 1，2，3，10，100
# algorithm iteration times, try 1, 2, 3, 10, 100
QS.equalization_setting.iterations = 10

# Equalization 权重阈值，试试 0.5, 2
# 这是个十分重要的属性，所有小于该值的权重不会参与运算
# value threshold of equalization, try 0.5 and 2
# it is a curical setting of equalization, value below this threshold won't get included in this optimizition.
QS.equalization_setting.value_threshold = .5  # try 0.5 and 2, it matters.

# 是否在 Equalization 中拉平 bias
# whether to equalize bias as well as weight
QS.equalization_setting.including_bias = False
QS.equalization_setting.bias_multiplier = 0.5

# 是否在 Equalization 中拉平 activation
# whether to equalize activation as well as weight
QS.equalization_setting.including_act = False
QS.equalization_setting.act_multiplier = 0.5
# ----------------------------------------------------------------------------------------------
QS.weight_split = False
# QS.weight_split_setting = WeightSplitSetting()
# 所有需要分裂的层的名字，Weight Split 会降低网络执行的性能，你必须手动指定那些层要被分裂
# Weight Split 和 Channel Split 都是一种以计算时间作为代价，提高量化精度的方法
# 这些方法主要适用于 per-tensor 的量化方案
# computing layers which are intended to be splited
QS.weight_split_setting.interested_layers = []

# 所有小于阈值的权重将被分裂
QS.weight_split_setting.value_threshold = 2.0

# 分裂方式，可以选 balance(平均分裂), random(随机分裂)
QS.weight_split_setting.method = 'balance'
# ----------------------------------------------------------------------------------------------
# OCS channel split configuration
QS.channel_split = False
# QS.channel_split_setting = ChannelSplitSetting()
# ----------------------------------------------------------------------------------------------
# Matrix Factorization Split. (Experimental method)
QS.matrix_factorization = False
# QS.matrix_factorization_setting = MatrixFactorizationSetting()
# ----------------------------------------------------------------------------------------------
# activation 量化与相关配置
QS.quantize_activation = True
# QS.quantize_activation_setting = ActivationQuantizationSetting()
# 激活值校准算法，不区分大小写，可以选择 minmax, kl, percentile, MSE, None
# 选择 None 时，将由 quantizer 指定量化算法
# activation calibration method
QS.quantize_activation_setting.calib_algorithm = None
# ----------------------------------------------------------------------------------------------
# 参数量化与相关配置
QS.quantize_parameter = True
# QS.quantize_parameter_setting = ParameterQuantizationSetting()
# 参数校准算法，不区分大小写，可以选择 minmax, percentile, kl, MSE, None
# parameter calibration method
QS.quantize_parameter_setting.calib_algorithm = None

# 是否处理被动量化参数
# whether to process passive parameters
QS.quantize_parameter_setting.quantize_passive_parameter = True

# 是否执行参数烘焙
# whether to bake quantization on parameter.
QS.quantize_parameter_setting.baking_parameter = True
# ----------------------------------------------------------------------------------------------
# 是否执行网络微调
QS.lsq_optimization = False
# QS.lsq_optimization_setting = LSQSetting()
QS.lsq_optimization_setting.interested_layers = []

# initial learning rate, by default Adam optimizer and a multistep scheduler with 0.1 decay
# are used for convergence
QS.lsq_optimization_setting.lr = 1e-5

# collecting device for block input and output
# turn this to cpu if CUDA OOM Error
QS.lsq_optimization_setting.collecting_device = 'cpu'

# num of training steps, please adjust it to your needs
QS.lsq_optimization_setting.steps = 500

# is scale trainable
QS.lsq_optimization_setting.is_scale_trainable = True

# regularization term
QS.lsq_optimization_setting.gamma = 0.0

# block size of graph cutting.
QS.lsq_optimization_setting.block_size = 4

# ----------------------------------------------------------------------------------------------
QS.blockwise_reconstruction = False
QS.blockwise_reconstruction_setting.interested_layers = []

# scale 是否可以训练
QS.blockwise_reconstruction_setting.is_scale_trainable = False

# 学习率
QS.blockwise_reconstruction_setting.lr = 1e-3

# 学习步数
QS.blockwise_reconstruction_setting.steps = 5000

# 正则化参数
QS.blockwise_reconstruction_setting.gamma = 1.0

# 缓存设备
QS.blockwise_reconstruction_setting.collecting_device = 'cuda'

# 区块大小
QS.blockwise_reconstruction_setting.block_size = 4
# ----------------------------------------------------------------------------------------------
# 是否启动 bias correction pass
QS.bias_correct = False
# QS.bias_correct_setting = BiasCorrectionSetting()
# 指定所有需要执行 BiasCorrection 的层的名字，不写就是所有层全部进行 bias correction
QS.bias_correct_setting.interested_layers = []

# 指定 BiasCorrection 的区块大小，越大越快，但也越不精准
QS.bias_correct_setting.block_size = 4

# 指定 bias 的统计步数，越大越慢，越小越不精准
QS.bias_correct_setting.steps = 32

# 缓存数据放在哪
QS.bias_correct_setting.collecting_device = 'executor'
# ----------------------------------------------------------------------------------------------
# 量化融合相关配置
QS.fusion = True
# QS.fusion_setting = QuantizationFusionSetting()

# 算子调度表，你可以编辑它来手动调度算子。
# QS.dispatching_table = DispatchingTable()
# QS.dispatching_table.dispatchings = {
#     'YOUR OEPRATION NAME': 'TARGET PLATFORM(INT)',
#     'FP32 OPERATION NAME': TargetPlatform.FP32.value,
#     'SOI OPERATION NAME': TargetPlatform.SHAPE_OR_INDEX.value,
#     'DSP INT8 OPERATION NAME': TargetPlatform.PPL_DSP_INT8.value,
#     'TRT INT8 OPERATION NAME': TargetPlatform.TRT_INT8.value,
#     'NXP INT8 OPERATION NAME': TargetPlatform.NXP_INT8.value,
#     'PPL INT8 OPERATION NAME': TargetPlatform.PPL_CUDA_INT8.value
# }
QS.dispatching_table.append(operation='OP NAME', platform=TargetPlatform.FP32)
