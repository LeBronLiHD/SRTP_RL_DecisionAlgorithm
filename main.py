# -*- coding: utf-8 -*-

import argparse


def get_user_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'play'], type=str, help='train model or play chinese-chess')
    parser.add_argument('--ai_count', default=1, choices=[1, 2], type=int, help='choose the number of AI players')
    parser.add_argument('--ai_function', default='mcts', choices=['mcts', 'net'], type=str, help='MCTS (slower but smarter) or net (faster)')
    parser.add_argument('--train_playout', default=400, type=int, help='MCTS training playout')
    parser.add_argument('--batch_size', default=512, type=int, help='training batch_size')
    parser.add_argument('--play_playout', default=400, type=int, help='MCTS play playout')
    parser.add_argument('--delay', dest='delay', action='store',
                        nargs='?', default=3, type=float, required=False,
                        help='set how many seconds you want to delay between each move')
    parser.add_argument('--end_delay', dest='end_delay', action='store',
                        nargs='?', default=3, type=float, required=False,
                        help='set how many seconds you want to delay after the end of game')
    parser.add_argument('--search_threads', default=16, type=int, help='the number of threads during searching')
    parser.add_argument('--processor', default='cpu', choices=['cpu', 'gpu'], type=str, help='cpu or gpu')
    parser.add_argument('--num_gpus', default=1, type=int, help='the number of gpu cores')
    parser.add_argument('--res_block_nums', default=7, type=int, help='the number of blocks in res-net')
    parser.add_argument('--human_color', default='b', choices=['w', 'b'], type=str, help='the turn of human (w or b)')
    return parser.parse_args()

if __name__ == '__main__':
    user_input = get_user_input()
