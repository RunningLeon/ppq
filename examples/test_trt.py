import argparse
import os
import os.path as osp

import yaml
from easydict import EasyDict as edict
from src import build_mmseg_dataloader, create_calib_input_data, run_cmd


def parse_args():
    parser = argparse.ArgumentParser(description='test trt int8')
    parser.add_argument('config', help='yaml config')
    parser.add_argument('--model', default=None, nargs='+')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    models = config.models
    nrof_model = len(models)
    MMDEPLOY_DIR = config.mmdeploy_dir

    if args.model is not None:
        models = [m for m in models if m.name in args.model]
    for idx, model in enumerate(models):
        print(f'Processing {idx}/{nrof_model}\n'
              f'model={model.name}\nconfig={model.model_cfg}')
        print(25 * '--')
        output_dir = osp.join(config.workspace, model.name)
        engine_dir = osp.join(output_dir, 'trt')
        os.makedirs(engine_dir, exist_ok=True)
        ONNX_MODEL_FILE = osp.join(output_dir, 'end2end.onnx')
        TRT_FP32_ENGINE = osp.join(output_dir, 'end2end.engine')
        TRT_FP16_ENGINE = osp.join(engine_dir, 'fp16.engine')
        FP32_JSON_FILE = osp.join(engine_dir, 'eval_fp32.json')

        if not osp.exists(ONNX_MODEL_FILE) or config.pipeline.torch2onnx:
            # torch2onnx
            cmd_lines = [
                'python',
                osp.join(MMDEPLOY_DIR, 'tools/deploy.py'),
                config.deploy.trt_fp32,
                model.model_cfg,
                model.checkpoint,
                config.deploy.image,
                '--device cuda:0',
                f'--work-dir {output_dir}',
            ]
            log_path = osp.join(output_dir, 'torch2onnx.log')
            run_cmd(cmd_lines, log_path)

        if config.pipeline.trt_fp32.eval:
            # test trt fp32
            cmd_lines = [
                'python',
                osp.join(MMDEPLOY_DIR, 'tools/test.py'),
                config.deploy.trt_fp32, model.model_cfg, '--device cuda:0',
                f'--model {TRT_FP32_ENGINE}', '--metrics mIoU',
                f'--json-file {FP32_JSON_FILE}'
            ]
            log_path = osp.join(engine_dir, 'eval_fp32.log')
            ret_code = run_cmd(cmd_lines, log_path)
            if ret_code == 0:
                os.system(f'cat {FP32_JSON_FILE}')

        if config.pipeline.trt_fp32.speed:
            cmd_lines = [
                'python',
                osp.join(MMDEPLOY_DIR, 'tools/profiler.py'),
                config.deploy.trt_fp32,
                model.model_cfg,
                config.deploy.eval_img_dir,
                '--device cuda:0',
                f'--model {TRT_FP32_ENGINE}',
            ]
            log_path = osp.join(engine_dir, 'speed_fp32.log')
            ret_code = run_cmd(cmd_lines, log_path)
            if ret_code == 0:
                os.system(f'cat {log_path}')

        if not osp.exists(TRT_FP16_ENGINE) or config.pipeline.trt_fp16.convert:
            cmd_lines = [
                'python',
                osp.join(MMDEPLOY_DIR, 'tools/onnx2tensorrt.py'),
                config.deploy.trt_fp16, ONNX_MODEL_FILE,
                osp.splitext(TRT_FP16_ENGINE)[0]
            ]
            log_path = osp.join(engine_dir, 'onnx2tensorrt-fp16.log')
            run_cmd(cmd_lines, log_path)

        if config.pipeline.trt_fp16.eval:
            trt_fp16_json = osp.join(engine_dir, 'eval_fp16.json')
            cmd_lines = [
                'python',
                osp.join(MMDEPLOY_DIR, 'tools/test.py'),
                config.deploy.trt_fp16, model.model_cfg, '--device cuda:0',
                f'--model {TRT_FP16_ENGINE}', '--metrics mIoU',
                f'--json-file {trt_fp16_json}'
            ]
            log_path = trt_fp16_json.replace('.json', '.log')
            ret_code = run_cmd(cmd_lines, log_path)

        if config.pipeline.trt_fp16.speed:
            cmd_lines = [
                'python',
                osp.join(MMDEPLOY_DIR, 'tools/profiler.py'),
                config.deploy.trt_fp16,
                model.model_cfg,
                config.deploy.eval_img_dir,
                '--device cuda:0',
                f'--model {TRT_FP16_ENGINE}',
            ]
            log_path = osp.join(engine_dir, 'speed_fp16.log')
            ret_code = run_cmd(cmd_lines, log_path)

        for num_quant in config.pipeline.trt_int8.num_quant:
            TRT_INT8_ENGINE = osp.join(engine_dir, f'quant{num_quant}.engine')
            INT8_JSON_FILE = osp.join(engine_dir,
                                      f'eval_quant{num_quant}.json')
            # test trt int8 of original trt
            print('test original trt int8')
            if config.pipeline.trt_int8.convert:
                # create calib file
                h5_calibe_file = config.calib.calibration_h5file.format(
                    num_quant)
                calibration_file = config.calib.calibration_file.format(
                    num_quant)
                if not osp.exists(h5_calibe_file):
                    data_dir, _ = osp.split(h5_calibe_file)
                    os.makedirs(data_dir, exist_ok=True)
                    calib_dataloader = build_mmseg_dataloader(
                        model.model_cfg, 'train', calibration_file)
                    create_calib_input_data(h5_calibe_file, calib_dataloader)

                cmd_lines = [
                    'python',
                    osp.join(MMDEPLOY_DIR, 'tools/onnx2tensorrt.py'),
                    config.deploy.trt_int8, ONNX_MODEL_FILE,
                    osp.splitext(TRT_INT8_ENGINE)[0],
                    f'--calib-file {h5_calibe_file}'
                ]
                log_path = osp.join(engine_dir,
                                    f'onnx2tensorrt-quant{num_quant}.log')
                run_cmd(cmd_lines, log_path)
            if config.pipeline.trt_int8.eval:
                cmd_lines = [
                    'python',
                    osp.join(MMDEPLOY_DIR, 'tools/test.py'),
                    config.deploy.trt_int8, model.model_cfg, '--device cuda:0',
                    f'--model {TRT_INT8_ENGINE}', '--metrics mIoU',
                    f'--json-file {INT8_JSON_FILE}'
                ]
                log_path = INT8_JSON_FILE.replace('.json', '.log')
                ret_code = run_cmd(cmd_lines, log_path)
                if ret_code == 0:
                    os.system(f'cat {INT8_JSON_FILE}')

            if config.pipeline.trt_int8.speed:
                cmd_lines = [
                    'python',
                    osp.join(MMDEPLOY_DIR, 'tools/profiler.py'),
                    config.deploy.trt_int8,
                    model.model_cfg,
                    config.deploy.eval_img_dir,
                    '--device cuda:0',
                    f'--model {TRT_INT8_ENGINE}',
                ]
                log_path = osp.join(engine_dir, f'speed_quant{num_quant}.log')
                ret_code = run_cmd(cmd_lines, log_path)

    print('all done for test trt')


if __name__ == '__main__':
    main()
