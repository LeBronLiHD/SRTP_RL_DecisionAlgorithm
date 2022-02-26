# Note of `cchess_zero`

&copy; LeBronLiHD

[TOC]

### original repository

- [GitHub link](https://github.com/chengstone/cchess-zero)

### data type of Chinese chess

- lowercase for black
- uppercase for red

| letter | chess type |
| :----: | :--------: |
|  K (King)      |       将/帅     |
|A (Advisor) |士 |
|E (Elephant) | 象/相|
|H (Horse) | 马|
|R (Race) | 车|
|C (Cannon) |炮 |
|P (Person) | 卒/兵|

### formulas of `leaf_node`

- get_Q_plus_U_new

$$
U = c\_puct * self.P *\displaystyle \frac{\sqrt{self.parent.N}}{1 + self.N}
$$

- get_Q_plus_U

$$
self.U = c\_puct * self.P *\displaystyle \frac{\sqrt{self.parent.N}}{1 + self.N}
$$

### data in `leaf_node`

```python
def __init__(self, in_parent, in_prior_p, in_state):
    self.P = in_prior_p
    self.Q = 0
    self.N = 0
    self.v = 0
    self.U = 0
    self.W = 0
    self.parent = in_parent
    self.child = {}
    self.state = in_state
    
self.root = leaf_node(None,		# in_parent 
                      self.p_,	# in_prior_p
                      "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr")
```

### data in `MCTS_tree`

```python
def __init__(self, in_state, in_forward, search_threads):
    self.noise_eps = 0.25
    self.dirichlet_alpha = 0.3    #0.03
    self.p_ = (1 - self.noise_eps) * 1 + self.noise_eps * np.random.dirichlet([self.dirichlet_alpha])
    self.root = leaf_node(None, self.p_, in_state)
    self.c_puct = 5    #1.5
    # self.policy_network = in_policy_network
    self.forward = in_forward
    self.node_lock = defaultdict(Lock)

    self.virtual_loss = 3
    self.now_expanding = set()
    self.expanded = set()
    self.cut_off_depth = 30
    # self.QueueItem = namedtuple("QueueItem", "feature future")
    self.sem = asyncio.Semaphore(search_threads)
    self.queue = Queue(search_threads)
    self.loop = asyncio.get_event_loop()
    self.running_simulation_num = 0
```

### data in `GameBoard`

```python
board_pos_name = np.array(create_position_labels()).reshape(9,10).transpose()
Ny = 10
Nx = 9

def __init__(self):
    self.state = "RNBAKABNR/9/1C5C1/P1P1P1P1P/9/9/p1p1p1p1p/1c5c1/9/rnbakabnr"
    self.round = 1
    # self.players = ["w", "b"]
    self.current_player = "w"
    self.restrict_round = 0
```

### data in `cchess_main`

```python
def __init__(self, playout=400, in_batch_size=128, exploration = True, in_search_threads = 16, processor = "cpu", num_gpus = 1, res_block_nums = 7, human_color = 'b'):
    self.epochs = 5
    self.playout_counts = playout    #400    #800    #1600    200
    self.temperature = 1    #1e-8    1e-3
    # self.c = 1e-4
    self.batch_size = in_batch_size    #128    #512
    # self.momentum = 0.9
    self.game_batch = 400    #  Evaluation each 400 times
    # self.game_loop = 25000
    self.top_steps = 30
    self.top_temperature = 1    #2
    # self.Dirichlet = 0.3    # P(s,a) = (1 - ϵ)p_a  + ϵη_a    #self-play chapter in the paper
    self.eta = 0.03
    # self.epsilon = 0.25
    # self.v_resign = 0.05
    # self.c_puct = 5
    self.learning_rate = 0.001    #5e-3    #    0.001
    self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
    self.buffer_size = 10000
    self.data_buffer = deque(maxlen=self.buffer_size)
    self.game_borad = GameBoard()
    # self.current_player = 'w'    #“w”表示红方，“b”表示黑方。
    self.policy_value_netowrk = policy_value_network(res_block_nums) if processor == 'cpu' else policy_value_network_gpus(num_gpus, res_block_nums)
    self.search_threads = in_search_threads
    self.mcts = MCTS_tree(self.game_borad.state, self.policy_value_netowrk.forward, self.search_threads)
    self.exploration = exploration
    self.resign_threshold = -0.8    #0.05
    self.global_step = 0
    self.kl_targ = 0.025
    self.log_file = open(os.path.join(os.getcwd(), 'log_file.txt'), 'w')
    self.human_color = human_color
```

### network training equations

```python
kl_tmp = old_probs * (np.log((old_probs + 1e-10) / (new_probs + 1e-10)))
```

![image-20220226192731679](C:/Users/87236/AppData/Roaming/Typora/typora-user-images/image-20220226192731679.png)

### data in `ChessGame`

```python
board = None    #ChessBoard()
cur_round = 1
game_mode = 1  # 0:HUMAN VS HUMAN 1:HUMAN VS AI 2:AI VS AI
time_red = []
time_green = []

def __init__(self, in_ai_count, in_ai_function, in_play_playout, in_delay, in_end_delay, batch_size, search_threads,
             processor, num_gpus, res_block_nums, human_color = "b"):
    self.human_color = human_color
    self.current_player = "w"
    self.players = {}
    self.players[self.human_color] = "human"
    ai_color = "w" if self.human_color == "b" else "b"
    self.players[ai_color] = "AI"

    ChessGame.board = ChessBoard(self.human_color == 'b')
    self.view = ChessView(self, board=ChessGame.board)
    self.view.showMsg("Loading Models...")    #"Red"    player_color
    self.view.draw_board(self.board)
    ChessGame.game_mode = in_ai_count
    self.ai_function = in_ai_function
    self.play_playout = in_play_playout
    self.delay = in_delay
    self.end_delay = in_end_delay

    self.win_rate = {}
    self.win_rate['w'] = 0.0
    self.win_rate['b'] = 0.0

    self.view.root.update()
    self.cchess_engine = cchess_main(playout=self.play_playout, in_batch_size=batch_size, exploration=False, in_search_threads=search_threads, processor=processor, num_gpus=num_gpus, res_block_nums=res_block_nums, human_color=human_color)

```

