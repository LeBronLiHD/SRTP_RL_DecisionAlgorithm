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
from cchess_alphazero.play_games.play_Qt_current import PlayWithHuman, string_to_list, trans_move

logger = getLogger(__name__)

HOST = "127.0.0.1"
# PORT = 6666
# HOST = "172.20.10.5"
# HOST = "10.162.26.114"
# HOST = "192.168.43.121"
# HOST = "172.20.10.5"
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

    # ai_move_first = False
    # human_move_first = True
    config_type = "mini"
    config = Config(config_type=config_type)
    config.opts.light = False
    pwhc = PlayWithHumanConfig()
    pwhc.update_play_config(config.play)
    # logger.info(f"AI move first : {ai_move_first}")

    play_r = PlayWithHuman(config)
    play_b = PlayWithHuman(config)

    last_exist = False
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

        if chess_board_string[11:] == "5,00;0,01;0,02;7,03;0,04;0,05;14,06;0,07;0,08;12,09;4,10;0,11;6,12;0," \
                                      "13;0,14;0,15;0,16;13,17;0,18;11,19;3,20;0,21;0,22;7,23;0,24;0,25;14,26;0,27;0," \
                                      "28;10,29;2,30;0,31;0,32;0,33;0,34;0,35;0,36;0,37;0,38;9,39;1,40;0,41;0,42;7," \
                                      "43;0," \
                                      "44;0,45;14,46;0,47;0,48;8,49;2,50;0,51;0,52;0,53;0,54;0,55;0,56;0,57;0,58;9," \
                                      "59;3," \
                                      "60;0,61;0,62;7,63;0,64;0,65;14,66;0,67;0,68;10,69;4,70;0,71;6,72;0,73;0,74;0," \
                                      "75;0," \
                                      "76;13,77;0,78;11,79;5,80;0,81;0,82;7,83;0,84;0,85;14,86;0,87;0,88;12,89;":
            print("game start")
            set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True,
                               device_list=config.opts.device_list)
            play_r.play_start(human_first=False)
            play_b.play_start(human_first=True)

            current_matrix, red_turn = algorithm.string2matrix(str(chess_board_string))
            last_exist = True

            rec_move = [0, 0, 0, 0]
            send_move = [0, 0, 0, 0, 0]

            print("Initialized")

            action = play_r.ai_step()
            move = string_to_list(action)
            flip_move = trans_move(move)
            send_move = [current_matrix[flip_move[0]][flip_move[1]]]
            send_move.extend(flip_move)
            print("send_move", send_move)

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
            continue

        if chess_board_string == "stop":
            play_r.ai.close()
            # print(f"胜者是 is {play.env.board.winner} !!!")
            play_r.env.board.print_record()

            play_b.ai.close()
            # print(f"胜者是 is {play.env.board.winner} !!!")
            play_b.env.board.print_record()

            print(
                "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
            backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
            conn.send(backMsg)
            continue

        print("last_exist", last_exist)

        if last_exist:
            last_matrix = current_matrix

        current_matrix, red_turn = algorithm.string2matrix(str(chess_board_string))

        if not last_exist:
            last_exist = True

        if last_exist and not red_turn:
            rec_move = matrix_to_move(np.array(last_matrix), np.array(current_matrix))
            print("ori_rec_move", rec_move)
            rec_move = trans_move(rec_move)
            print("rec_move", rec_move)

            if play_b.env.red_to_move:
                play_b.human_step(rec_move)
                if play_b.env.board.is_end():
                    play_b.ai.close()
                    print(f"胜者是 is {play_b.env.board.winner} !!!")
                    play_b.env.board.print_record()

                    print(
                        "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
                    backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
                    conn.send(backMsg)
                    continue

            if not play_b.env.red_to_move:
                action = play_b.ai_step()
                move = string_to_list(action)
                flip_move = trans_move(move)
                send_move = [current_matrix[flip_move[0]][flip_move[1]]]
                send_move.extend(flip_move)
                print("send_move", send_move)

        elif last_exist and red_turn:
            rec_move = matrix_to_move(np.array(last_matrix), np.array(current_matrix))
            print("ori_rec_move", rec_move)
            rec_move = trans_move(rec_move)
            print("rec_move", rec_move)

            if not play_r.env.red_to_move:
                play_r.human_step(rec_move)
                if play_r.env.board.is_end():
                    play_r.ai.close()
                    print(f"胜者是 is {play_r.env.board.winner} !!!")
                    play_r.env.board.print_record()

                    print(
                        "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
                    backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
                    conn.send(backMsg)
                    continue

            if play_r.env.red_to_move:
                action = play_r.ai_step()
                move = string_to_list(action)
                flip_move = trans_move(move)
                send_move = [current_matrix[flip_move[0]][flip_move[1]]]
                send_move.extend(flip_move)
                print("send_move", send_move)

        if play_r.env.board.is_end():
            play_r.ai.close()
            print(f"胜者是 is {play_r.env.board.winner} !!!")
            play_r.env.board.print_record()

            print(
                "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
            backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
            conn.send(backMsg)
            continue

        if play_b.env.board.is_end():
            play_b.ai.close()
            print(f"胜者是 is {play_b.env.board.winner} !!!")
            play_b.env.board.print_record()

            print(
                "\n==================================\n" + "SHENYIPENG NB" + chess_board_string + "\n===========================================\n")
            backMsg = bytes("SHENYIPENG NB".encode("UTF-8"))
            conn.send(backMsg)
            continue

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
