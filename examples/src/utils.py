import os
import os.path as osp
import subprocess


def run_cmd(cmd_lines, log_path):
    """
    Args:
        cmd_lines: (list[str]): A command in multiple line style.
        log_path (str): Path to log file.

    Returns:
        int: error code.
    """
    sep = '\\'
    cmd_for_run = f' {sep}\n'.join(cmd_lines) + '\n'
    parent_path = osp.split(log_path)[0]
    os.makedirs(parent_path, exist_ok=True)

    print(100 * '-')
    print(f'Start running cmd\n{cmd_for_run}')
    print(f'Logging log to \n{log_path}')

    with open(log_path, 'w', encoding='utf-8') as file_handler:
        # write cmd
        file_handler.write(f'Command:\n{cmd_for_run}\n')
        file_handler.flush()
        process_res = subprocess.Popen(cmd_for_run,
                                       cwd=os.getcwd(),
                                       shell=True,
                                       stdout=file_handler,
                                       stderr=file_handler)
        process_res.wait()
        return_code = process_res.returncode

    if return_code != 0:
        print(f'Got shell return code={return_code}')
        with open(log_path, 'r') as f:
            content = f.read()
            print(f'Log message\n{content}')
    return return_code
