import os
import sys


def get_path_root():
    '''os.path.dirname(os.path.abspath(__file__))'''
    debug_vars = dict((a, b) for a, b in os.environ.items() if a.find('IPYTHONENABLE') >= 0)
    # 根据不同场景获取根目录
    if len(debug_vars) > 0:
        """当前为debug运行时"""
        path_root = sys.path[2]
    elif getattr(sys, 'frozen', False):
        """当前为exe运行时"""
        path_root = os.getcwd()
    else:
        """正常执行"""
        path_root = sys.path[1]
    path_root = path_root.replace("\\", "/")  # 替换斜杠
    return path_root
