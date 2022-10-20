import argparse
import logging
import os
import os.path as osp
import shutil

import torch
import yaml
from easydict import EasyDict as edict
from src import (build_mmseg_dataloader, evaluate_model,
                 parse_quantization_config, print_quant_settings, run_cmd)

from ppq.api import (ENABLE_CUDA_KERNEL, TargetPlatform, TorchExecutor,
                     export_ppq_graph, load_onnx_graph, ppq_warning,
                     quantize_native_model)
from ppq.quantization.analyse import (graphwise_error_analyse,
                                      layerwise_error_analyse,
                                      parameter_analyse, statistical_analyse,
                                      variable_analyse)


def parse_args():
    parser = argparse.ArgumentParser(description='test int8')
    parser.add_argument('config', help='yaml config')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    QS = parse_quantization_config(config.quant_settings)
    print_quant_settings(QS)
    WORKING_DIRECTORY = config.model.working_dir
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    shutil.copy(args.config,
                osp.join(WORKING_DIRECTORY,
                         osp.split(args.config)[1]))
    PPQ_ONNX_INT8_FILE = os.path.join(WORKING_DIRECTORY, 'ppq-int8.onnx')
    PPQ_ONNX_INT8_CONFIG = os.path.join(WORKING_DIRECTORY, 'ppq-int8.json')
    PPQ_TRT_INT8_FILE = os.path.join(WORKING_DIRECTORY, 'ppq-int8.engine')

    # DEPLOY_CFG_PATH = osp.join(config.mmdeploy_dir, config.model.deploy_cfg)
    DEPLOY_CFG_INT8_PATH = osp.join(config.mmdeploy_dir,
                                    config.model.deploy_cfg_int8)
    MODEL_CFG_PATH = osp.join(config.mmseg_dir, config.model.model_cfg)

    NETWORK_INPUTSHAPE = [
        config.calib.batch_size, 3, config.calib.input_height,
        config.calib.input_width
    ]
    print(TargetPlatform)
    TARGET_PLATFORM = eval(config.calib.target_platform)
    collate_fn = lambda x: x.to(config.calib.device)  # noqa: E731
    calib_dataloader = build_mmseg_dataloader(MODEL_CFG_PATH, 'train',
                                              config.calib.calibration_file)

    with ENABLE_CUDA_KERNEL():
        graph = load_onnx_graph(onnx_import_file=config.model.onnx_file)
        print('网络正量化中，根据你的量化配置，这将需要一段时间:')
        quantized = quantize_native_model(
            setting=QS,  # setting 对象用来控制标准量化逻辑
            model=graph,
            calib_dataloader=calib_dataloader,
            calib_steps=config.calib.calib_steps,
            input_shape=NETWORK_INPUTSHAPE,
            inputs=None,
            collate_fn=collate_fn,
            platform=TARGET_PLATFORM,
            device=config.calib.device,
            do_quantize=True)

        if config.calib.analysis.test_executor:
            # -------------------------------------------------------------------
            # 如果你需要执行量化后的神经网络并得到结果，则需要创建一个 executor
            # 这个 executor 的行为和 torch.Module 是类似的，你可以利用这个东西来获取执行结果
            # 请注意，必须在 export 之前执行此操作。
            # -------------------------------------------------------------------
            executor = TorchExecutor(graph=quantized,
                                     device=config.calib.device)
            val_dataloader = build_mmseg_dataloader(MODEL_CFG_PATH, 'val')
            json_file = osp.join(WORKING_DIRECTORY, 'ppq_executor_val.json')
            print(100 * '--')
            print('evaluate val dataset')
            evaluate_model(executor, val_dataloader, json_file)

        # -------------------------------------------------------------------
        # PPQ 计算量化误差时，使用信噪比的倒数作为指标，即噪声能量 / 信号能量
        # 量化误差 0.1 表示在整体信号中，量化噪声的能量约为 10%
        # 你应当注意，在 graphwise_error_analyse 分析中，我们衡量的是累计误差
        # 网络的最后一层往往都具有较大的累计误差，这些误差是其前面的所有层所共同造成的
        # 你需要使用 layerwise_error_analyse 逐层分析误差的来源
        # -------------------------------------------------------------------
        if config.calib.analysis.graphwise:
            print('正计算网络量化误差(SNR)，最后一层的误差应小于 0.1 以保证量化精度:')
            reports = graphwise_error_analyse(
                graph=quantized,
                running_device=config.calib.device,
                steps=config.calib.calib_steps,
                dataloader=calib_dataloader,
                collate_fn=collate_fn)
            for op, snr in reports.items():
                if snr > 0.1:
                    ppq_warning(f'层 {op} 的累计量化误差显著，请考虑进行优化')

        if config.calib.analysis.layerwise:
            print('正计算逐层量化误差(SNR)，每一层的独立量化误差应小于 0.1 以保证量化精度:')
            layerwise_error_analyse(graph=quantized,
                                    running_device=config.calib.device,
                                    interested_outputs=None,
                                    dataloader=calib_dataloader,
                                    collate_fn=collate_fn)
        if config.calib.analysis.parameter:
            print('--- parameter_analyse')
            parameter_analyse(graph=quantized)
            print('--- variable_analyse')
            variable_analyse(quantized,
                             dataloader=calib_dataloader,
                             interested_outputs=[],
                             collate_fn=collate_fn,
                             running_device=config.calib.device,
                             samples_per_step=65536,
                             steps=8,
                             dequantize=False)
            records = statistical_analyse(quantized,
                                          running_device=config.calib.device,
                                          dataloader=calib_dataloader,
                                          collate_fn=collate_fn,
                                          steps=8)
            print(records)

        print('网络量化结束，正在生成目标文件:')
        export_ppq_graph(graph=quantized,
                         platform=TARGET_PLATFORM,
                         graph_save_to=PPQ_ONNX_INT8_FILE,
                         config_save_to=PPQ_ONNX_INT8_CONFIG)

    torch.cuda.empty_cache()
    cmd_lines = [
        'python',
        osp.join(config.mmdeploy_dir, 'tools/onnx2tensorrt.py'),
        DEPLOY_CFG_INT8_PATH,
        PPQ_ONNX_INT8_FILE,
        osp.splitext(PPQ_TRT_INT8_FILE)[0],
    ]
    log_path = osp.join(WORKING_DIRECTORY, 'ppq_onnx2tensorrt.log')
    run_cmd(cmd_lines, log_path)

    cmd_lines = [
        'python',
        osp.join(config.mmdeploy_dir,
                 'tools/test.py'), DEPLOY_CFG_INT8_PATH, MODEL_CFG_PATH,
        '--device cuda:0', f'--model {PPQ_TRT_INT8_FILE}', '--metrics mIoU'
    ]
    log_path = osp.join(WORKING_DIRECTORY, 'test_ppq_trt_int8.log')
    run_cmd(cmd_lines, log_path)

    logging.info(f'Saved results to {WORKING_DIRECTORY}')


if __name__ == '__main__':
    main()
