import os
import shutil

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def rmdir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
