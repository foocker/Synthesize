import os
from os.path import getsize, join

def remove_dir(top_dir):
    if not os.path.exists(top_dir):
        print('not exists')
        return
    if not os.path.isdir(top_dir):
        print('not a dir')
        return
    for dir_p, _, files in os.walk(top_dir, topdown=False, onerror=None):
        for file in files:
            file_p = join(dir_p, file)
            os.remove(file_p)
        os.rmdir(dir_p)


def get_dir_size(dir_top):
    size = 0
    for dir_p, _, files in os.walk(dir_top):
        for file in files:
            size += getsize(join(dir_p, file))
    
    return size
