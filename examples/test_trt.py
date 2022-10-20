import argparse
import os
import os.path as osp

import yaml
from easydict import EasyDict as edict
from src import build_mmseg_dataloader, create_calib_input_data, run_cmd


def parse_args():
    parser = argparse.ArgumentParser(description='test trt int8')
    parser.add_argument('config', help='yaml config')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    WORKING_DIRECTORY = config.model.working_dir
    MMDEPLOY_DIR = config.mmdeploy_dir
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    ONNX_MODEL_FILE = osp.join(WORKING_DIRECTORY, 'end2end.onnx')
    DEPLOY_CFG_PATH = osp.join(config.mmdeploy_dir, config.model.deploy_cfg)
    DEPLOY_CFG_PATH_INT8 = osp.join(config.mdeploy_dir,
                                    config.model.deploy_cfg_int8)
    MODEL_CFG_PATH = osp.join(config.mmseg_dir, config.model.model_cfg)
    TEST_IMAGE = osp.join(config.mmseg_dir, 'demo/demo.png')

    PYTORCH_CHECKPOINT = config.model.checkpoint
    # torch2onnx
    cmd_lines = [
        'python',
        osp.join(MMDEPLOY_DIR, 'tools/deploy.py'),
        DEPLOY_CFG_PATH,
        MODEL_CFG_PATH,
        PYTORCH_CHECKPOINT,
        TEST_IMAGE,
        '--device cuda:0',
        f'--work-dir {WORKING_DIRECTORY}',
    ]
    log_path = osp.join(WORKING_DIRECTORY, 'torch2onnx.log')
    run_cmd(cmd_lines, log_path)

    # test trt fp32
    cmd_lines = [
        'python',
        osp.join(MMDEPLOY_DIR, 'tools/test.py'), DEPLOY_CFG_PATH,
        MODEL_CFG_PATH, '--device cuda:0',
        f'--model {osp.join(WORKING_DIRECTORY, "end2end.engine")}',
        '--metrics mIoU'
    ]
    log_path = osp.join(WORKING_DIRECTORY, 'test_trt_fp32.log')
    run_cmd(cmd_lines, log_path)

    # test trt int8 of original trt
    print('test original trt int8')
    # create calib file
    h5_calibe_file = config.calib.calibration_h5file
    if not osp.exists(h5_calibe_file):
        calib_dataloader = build_mmseg_dataloader(
            config.model.model_cfg, 'train', config.calib.calibration_file)
        create_calib_input_data(h5_calibe_file, calib_dataloader)
    trt_int8_engine = osp.join(WORKING_DIRECTORY, 'end2end-int8.engine')
    cmd_lines = [
        'python',
        osp.join(MMDEPLOY_DIR, 'tools/onnx2tensorrt.py'), DEPLOY_CFG_PATH_INT8,
        ONNX_MODEL_FILE,
        osp.splitext(trt_int8_engine)[0], f'--calib-file {h5_calibe_file}'
    ]
    log_path = osp.join(WORKING_DIRECTORY, 'original_onnx2tensorrt-int8.log')
    run_cmd(cmd_lines, log_path)

    cmd_lines = [
        'python',
        osp.join(MMDEPLOY_DIR,
                 'tools/test.py'), DEPLOY_CFG_PATH_INT8, MODEL_CFG_PATH,
        '--device cuda:0', f'--model {trt_int8_engine}', '--metrics mIoU'
    ]
    log_path = osp.join(WORKING_DIRECTORY, 'test_trt_int8.log')
    run_cmd(cmd_lines, log_path)


if __name__ == '__main__':
    main()
