# -*- coding: utf-8 -*-

"""
file: service.py
socket service
"""

import socket
import threading
import time
import sys
import numpy as np

import algorithm
import load_data
from logging import getLogger
from matrix_to_move import matrix_to_move
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.config import Config, PlayWithHumanConfig
from cchess_alphazero.play_games.play_Qt import config_play, string_to_list, trans_move

logger = getLogger(__name__)

HOST = "127.0.0.1"
# PORT = 6666
# HOST = "172.20.10.5"
# HOST = "10.162.26.114"
# HOST = "192.168.43.121"
PORT = 9999
global rec_move, send_move
global last_matrix, current_matrix


# __isStepGenerate = False


def socket_service():
    # __isStepGenerate = False
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # init
        # 防止socket server重启后端口被占用（socket.error: [Errno 98] Address already in use）
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))  # bonding
        s.listen(10)  # monitor, backlog = 10
        # s.setblocking(True)
        # default value is True, if False, program throw an error if accept and recv have no data

    except socket.error as msg:
        print(msg)
        sys.exit(1)

    print("Waiting connection...")

    while True:
        conn, addr = s.accept()
        # return (conn, address)
        # s.accept_ex(), have a return value, 0 is success and code if failure
        t = threading.Thread(target=deal_data, args=(conn, addr))
        t.start()
        # start a new thread, this line only execute once


def deal_data(conn, addr):
    global rec_move, send_move
    global last_matrix, current_matrix
    print("Accept new connection from {0}".format(addr))
    databack = ("Hello, client from {0}".format(addr)).encode()
    # conn.send(databack)

    # mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    # from cchess_alphazero import manager
    # manager.start()

    play = config_play()
    play.start()
    last_chess_board_string = None

    while True:
        print("ready_to_receive")
        data = conn.recv(1024)  # 1024 is the longest length of string
        print('{0} client send data is {1}'.format(addr, data.decode()))
        time.sleep(0.5)

        if data.decode() == "exit" or not data:
            print('{0} connection close'.format(addr))
            conn.send(bytes('Connection closed!'.encode("UTF-8")))
            break
        chess_board_string = data.decode()
        print(data.decode())

        if last_chess_board_string is not None and last_chess_board_string[11:] == chess_board_string[11:]:
            print(
                "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
            backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
            conn.send(backMsg)
            continue
        last_chess_board_string = chess_board_string

        if data.decode() == 'connection test, hello pycharm! (RL)':
            print('=' * 80)

            print(
                "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
            backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
            # conn.send(backMsg)
            print("conn.send(backMsg)", conn.send(backMsg))
            continue

        if chess_board_string == "stop":
            play.ai.close()
            # print(f"胜者是 is {play.env.board.winner} !!!")
            play.env.board.print_record()

            print(
                "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
            backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
            conn.send(backMsg)
            continue

        current_matrix, red_turn = algorithm.string2matrix(str(chess_board_string))

        # print('matrix:', matrix, 'turn:', turn)
        matrix = current_matrix[:-1]  # 未开辟新内存空间
        # print('np.array(matrix).shape:', np.array(matrix).shape)

        print("matrix: ", matrix)
        col = 10
        for row in matrix:
            for i in range(col // 2):
                row[i], row[col - 1 - i] = row[col - 1 - i], row[i]
        print("matrix: ", matrix)

        action = play.get_action(matrix, red_turn)

        move = string_to_list(action)
        send_move = [matrix[move[0]][move[1]]]

        flip_move = trans_move(move)
        send_move.extend(flip_move)

        print("will_send_move")
        # step = load_data.ChessStep(0, 1, 2, 3, 4)
        step = load_data.ChessStep(send_move[0], send_move[1], send_move[2], send_move[3], send_move[4])
        backMsg_example = step.generate_msg()
        print("backMsg_example ", backMsg_example)
        backMsg = bytes(backMsg_example.encode("UTF-8"))
        # conn.send(backMsg)
        print("conn.send(backMsg)", conn.send(backMsg))
        print("\n==================================\n" + "send_move", send_move,
              chess_board_string + "\n===========================================\n")
        # print("send_move", send_move)
        # conn.send(bytes(backMsg, "UTF-8"))
        # conn.send(bytes("Hello QT, from: {0}", "UTF-8"))
    conn.close()


if __name__ == '__main__':
    socket_service()
