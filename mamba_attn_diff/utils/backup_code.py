import os
import sys
import glob
import torch
import shutil
import logging
import datetime
from torch.utils.tensorboard import SummaryWriter

UNSAVED_DIRS = ['outputs', 'checkpoint', 'checkpoints', 'workdir', 'build', '.git', '__pycache__', 'assets', 'samples']

def backup_code(work_dir, verbose=False):
    base_dir = './' #os.path.dirname(os.path.abspath(__file__))

    dir_list = ["*.py", ]
    for file in os.listdir(base_dir):
        sub_dir = os.path.join(base_dir, file)
        if os.path.isdir(sub_dir):
            if file in UNSAVED_DIRS:
                continue

            for root, dirs, files in os.walk(sub_dir):
                for dir_name in dirs:
                    dir_list.append(os.path.join(root, dir_name)+"/*.py")

        elif file.split('.')[-1] == 'py':
            pass

    # print(dir_list)

    for pattern in dir_list:
        for file in glob.glob(pattern):
            src = os.path.join(base_dir, file)
            dst = os.path.join(work_dir, 'backup', os.path.dirname(file))
            # print(base_dir, src, dst)

            if verbose:
                logging.info('Copying %s -> %s' % (os.path.relpath(src), os.path.relpath(dst)))
            
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, dst)


