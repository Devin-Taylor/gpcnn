import os
import shutil

def mkdir(path: str):
    if not os.path.exists(path):
        os.mkdir(path)

def rmdir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
