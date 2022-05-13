# -*- coding: utf-8 -*-

# import tarfile
# import os
#
# import tensorflow as tf
# import keras
# import platform

import global_params

import sys
sys.dont_write_bytecode = True

CHESS_TABLE = [
    "b_gen_",
    "b_adv_",
    "b_ele_",
    "b_hor_",
    "b_cha_",
    "b_can_",
    "b_sol_",
    "r_gen_",
    "r_adv_",
    "r_ele_",
    "r_hor_",
    "r_cha_",
    "r_can_",
    "r_sol_",
]

CHESS_CN = [
    "一",
    "将",
    "士",
    "象",
    "馬",
    "車",
    "砲",
    "卒",
    "帅",
    "仕",
    "相",
    "傌",
    "俥",
    "炮",
    "兵"
]


def str2int(string):
    return CHESS_TABLE.index(string)


def int2str(index):
    return CHESS_TABLE[index]


def int2cn(index):
    return CHESS_CN[index]


class ChessStep:
    def __init__(self, chess_num, pos_x, pos_y, tar_x, tar_y, kill = False, kill_num = 0):
        self.chess_num = chess_num
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.tar_x = tar_x
        self.tar_y = tar_y
        self.kill = kill
        self.kill_num = kill_num

    def generate_msg(self):
        msg = str(self.chess_num) + "," + \
              str(self.pos_x) + str(self.pos_y) + str(self.tar_x) + str(self.tar_y)
        if self.kill:
            msg += "1" + "," + str(self.kill_num)
        else:
            msg += "0,0"
        return msg


if __name__ == '__main__':
    step = ChessStep(1, 4, 0, 4, 1)
    print(step.generate_msg())

# def docuChessInfo(pathChessChoose):
#     # environment
#     print("tensorflow->", tf.__version__)
#     print("keras     ->", keras.__version__)
#     print("python    ->", platform.python_version())
#
#     # 遍历数据集，并存入inf列表
#     # 建立txt文件记录数据
#     chess_info_txt = global_params.M_chess_info_txt
#     f = open(chess_info_txt, 'w')
#
#     # 遍历
#     count = 0
#     files = os.listdir(pathChessChoose)
#     for filename in files:
#         printFilename = "Name -> " + str(count) + " -> " + filename
#         print(printFilename)
#         count += 1
#
#     index = 0
#
#     all_chess_data_path = []
#
#     # 写入txt文件中
#     for filename in files:
#         tempDir = pathChessChoose + "/" + filename
#         tempFiles = os.listdir(tempDir)
#         for tempFilename in tempFiles:
#             all_chess_data_path.append(os.path.join(tempDir, tempFilename))
#             fileDir = str(index) + " " + filename + " " + tempFilename + "\n"
#             f.write(fileDir) # 编号 姓名 图片路径与图片名
#         index += 1 # 编号+1
#
#     print("chess information documentation done!")
#     return all_chess_data_path
#
#
# def main(): # 主函数
#     pathChessChoose = global_params.M_data_360_path  # 人为选择的数据的路径
#     all_chess_data_path = docuChessInfo(pathChessChoose)
