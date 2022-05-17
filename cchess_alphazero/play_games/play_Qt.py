from collections import defaultdict
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.config import Config, PlayWithHumanConfig
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight

import algorithm
import numpy as np

# logger = getLogger(__name__)

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

fen_list = ['', 's', 'm', 'e', 'k', 'r', 'c', 'p', 'K', 'M', 'E', 'S', 'R', 'C', 'P']
# fen_list = ['', 'K', 'M', 'E', 'S', 'R', 'C', 'P', 's', 'm', 'e', 'k', 'r', 'c', 'p']


def string_to_list(action):
    row1 = int(action[0])
    col1 = int(action[1])
    row2 = int(action[2])
    col2 = int(action[3])
    move = [row1, col1, row2, col2]
    return move


def trans_move(move):
    return [move[0], 9 - move[1], move[2], 9 - move[3]]


def config_play():
    # set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    config_type = "mini"
    config = Config(config_type=config_type)
    pwhc = PlayWithHumanConfig()
    pwhc.update_play_config(config.play)  # 更新play参数，替换config中默认参数
    play = PlayWithHuman(config)
    return play


class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.state = None
        self.model = None
        self.pipe = None
        self.ai = None
        self.is_red_turn = None

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def FENboard(self):
        '''
        FEN board representation
        rules: https://www.xqbase.com/protocol/pgnfen2.htm
        '''
        cnt = 0
        fen = ''
        for i in range(9, -1, -1):
            cnt = 0
            for j in range(9):
                chessman = self.state[j][i]
                if chessman == 0:
                    cnt += 1
                else:
                    if cnt > 0:
                        fen = fen + str(cnt)
                    fen = fen + fen_list[chessman]
                    cnt = 0
            if cnt > 0:
                fen = fen + str(cnt)
            if i > 0:
                fen = fen + '/'
        fen += ' r'
        fen += ' - - 0 1'
        return fen

    def fliped_FENboard(self):
        fen = self.FENboard()
        foo = fen.split(' ')
        rows = foo[0].split('/')

        def swapcase(a):
            if a.isalpha():
                return a.lower() if a.isupper() else a.upper()
            return a

        def swapall(aa):
            return "".join([swapcase(a) for a in aa])

        return "/".join([swapall(reversed(row)) for row in reversed(rows)]) \
               + " " + ('r' if foo[1] == 'b' else 'b') \
               + " " + foo[2] \
               + " " + foo[3] + " " + foo[4] + " " + foo[5]

    def observation(self):
        if self.is_red_turn:
            fen = self.FENboard()
        else:
            fen = self.fliped_FENboard()
        return fen

    def get_state(self):
        fen = self.observation()
        foo = fen.split(' ')
        return foo[0]

    def start(self):

        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                               enable_resign=True, debugging=False)

    def get_action(self, state, red_to_move):

        self.state = state
        self.is_red_turn = red_to_move

        print('state:', self.get_state())
        action, policy = self.ai.action(self.get_state(), 0)
        if not self.is_red_turn:
            action = flip_move(action)
        if action is None:
            print("AI投降了!")

        print(f"AI选择移动 {action}")

        # self.env.board.print_to_cl()
        # self.ai.close()

        return action


if __name__ == "__main__":
    # mp.set_start_method('spawn')
    # sys.setrecursionlimit(10000)
    # start()

    matrix, turn = algorithm.string2matrix('red-conn<1>5,00;0,01;0,02;7,03;0,04;0,05;14,06;0,07;0,08;12,09;4,10;0,'
                                           '11;6,12;0,13;0,14;0,15;0,16;13,17;0,18;11,19;3,20;0,21;0,22;7,23;0,24;0,'
                                           '25;14,26;0,27;0,28;10,29;2,30;0,31;0,32;0,33;0,34;0,35;0,36;0,37;0,38;9,'
                                           '39;1,40;0,41;0,42;7,43;0,44;0,45;14,46;0,47;0,48;8,49;2,50;0,51;0,52;0,'
                                           '53;0,54;0,55;0,56;0,57;0,58;9,59;3,60;0,61;0,62;7,63;0,64;0,65;14,66;0,'
                                           '67;0,68;10,69;4,70;0,71;6,72;0,73;0,74;0,75;0,76;13,77;0,78;11,79;5,80;0,'
                                           '81;0,82;7,83;0,84;0,85;14,86;0,87;0,88;12,89;')

    # print('matrix:', matrix, 'turn:', turn)
    matrix = matrix[:-1]
    # print('np.array(matrix).shape:', np.array(matrix).shape)

    col = 10
    for row in matrix:
        for i in range(col // 2):
            row[i], row[col - 1 - i] = row[col - 1 - i], row[i]

    # print('np.array(matrix).shape:', np.array(matrix).shape)
    # print('matrix:', matrix, 'turn:', turn)

    play = config_play()
    play.start()

    action = play.get_action(matrix, True)

    move = string_to_list(action)
    send_move = [matrix[move[0]][move[1]]]

    flip_move = trans_move(move)
    send_move.extend(flip_move)

    print(action)
    print(move)
    print(send_move)
