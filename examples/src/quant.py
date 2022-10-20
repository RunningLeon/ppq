import logging
from pprint import pprint

from ppq.api.setting import QuantizationSettingFactory


def parse_quantization_config(config):
    QS = QuantizationSettingFactory.default_setting()
    for k, v in config.items():
        try:
            value = eval(f'QS.{k}')
            if value != v:
                if isinstance(v, str):
                    cmd = f'QS.{k} = "{v}"'
                else:
                    cmd = f'QS.{k} = {v}'
                exec(cmd)
        except AttributeError:
            logging.error(f'{k} not found in QS setting.')
            continue
    return QS


def print_quant_settings(qs):
    stats = {}

    def trans(obj, prefix):
        if hasattr(obj, '__dict__'):
            keys = sorted(obj.__dict__.keys())
            for k in keys:
                v = obj.__dict__[k]
                trans(v, f'{prefix}.{k}')
        else:
            stats[prefix] = obj

    trans(qs, 'QS')
    stats.pop('QS.extension_setting.my_first_parameter')
    stats.pop('QS.extension')
    stats.pop('QS.signature')
    stats.pop('QS.dispatching_table.intro_0')
    stats.pop('QS.dispatching_table.intro_1')
    stats.pop('QS.dispatching_table.intro_2')

    print(100 * '-')
    pprint(stats)
    print(100 * '-')
