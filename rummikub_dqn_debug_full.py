# ============================================================
# Rummikub DQN - 1차 동작 확인용
#
# 전제:
#   import rummikub_env 가 이미 성공해야 함
#   STATE_DIM  = 702
#   ACTION_DIM = 699
#
# DQN Player : 0
# Opponent   : Random Agent (Player 1)
#
# 추가 기능:
#   1. BEFORE / AFTER 평가 진행상황 출력
#   2. TRAIN episode별 승패 출력
#   3. 최근 20 episode 승률 출력
#   4. 학습/평가 데이터를 CSV로 저장
#   5. 실험 설정을 JSON으로 저장
#
# 주의:
#   NUM_PLAYERS
#   DQN_PLAYER
#   OPPONENT_PLAYER
#   reset_fixed_game()
#
# 위 항목들은 기존 코드와 마찬가지로
# 앞선 셀에서 이미 정의되어 있다고 가정한다.
# ============================================================


import math
import random
import time
import csv
import json
import os

from datetime import datetime
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import rummikub_env


# ============================================================
# 0. 기본 설정
# ============================================================

STATE_DIM = rummikub_env.STATE_DIM
ACTION_DIM = rummikub_env.ACTION_DIM

assert STATE_DIM == 702
assert ACTION_DIM == 699

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("DEVICE =", DEVICE)
print("STATE_DIM =", STATE_DIM)
print("ACTION_DIM =", ACTION_DIM)


# ------------------------------------------------------------
# 처음 정의했던 보상함수
# ------------------------------------------------------------

REWARD_WEIGHTS = {
    "hand_normal": 1.0,
    "hand_2_12": 2.0,
    "hand_1_13": 3.0,

    "table_len3": 2.0,
    "table_len4": 1.0,
    "table_len5plus": 0.0,

    "win": 1000.0,
    "loss": -1000.0,
    "tie": 0.0,
}


# ------------------------------------------------------------
# 1차 테스트에서는 행동 생성 수를 제한한다.
#
# 이유:
# 완전 재배열 때문에 어떤 상태에서는 행동 후보가 매우 많아질 수 있음.
#
# 이건 DQN 코드가 작동하는지 검증하기 위한 값.
# 최종 연구에서는 반드시 별도로 조정해야 함.
# ------------------------------------------------------------

MAX_LEGAL_ACTIONS = 64


# ------------------------------------------------------------
# DQN hyperparameter
# ------------------------------------------------------------

GAMMA = 0.99

LEARNING_RATE = 3e-4

BATCH_SIZE = 32
REPLAY_CAPACITY = 5000
MIN_REPLAY_SIZE = 500

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 30000

TAU = 0.005


# 우선 동작 확인
TRAIN_EPISODES = 500


# 학습 전/후 평가
PRE_EVAL_GAMES = 100
POST_EVAL_GAMES = 100


# ============================================================
# 1. 로그 저장 설정
# ============================================================

# 실행마다 별도 폴더 생성
RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

LOG_DIR = f"dqn_run_{RUN_ID}"

os.makedirs(
    LOG_DIR,
    exist_ok=True
)


TRAIN_CSV_PATH = os.path.join(
    LOG_DIR,
    "train_log.csv"
)

BEFORE_EVAL_CSV_PATH = os.path.join(
    LOG_DIR,
    "eval_before.csv"
)

AFTER_EVAL_CSV_PATH = os.path.join(
    LOG_DIR,
    "eval_after.csv"
)

SUMMARY_CSV_PATH = os.path.join(
    LOG_DIR,
    "summary.csv"
)

CONFIG_JSON_PATH = os.path.join(
    LOG_DIR,
    "run_config.json"
)


print("LOG_DIR =", LOG_DIR)


# ------------------------------------------------------------
# CSV 컬럼 정의
# ------------------------------------------------------------

TRAIN_CSV_FIELDS = [

    "episode",

    "global_step",

    "result",
    "utility",

    "episode_reward",
    "reward_ma20",

    "win_rate20",

    "epsilon",

    "replay_size",

    "episode_loss",
    "loss_ma100",

    "episode_dqn_steps",

    "episode_action_generations",
    "episode_truncated_generations",
    "episode_truncation_rate",

    "cumulative_action_generations",
    "cumulative_truncated_generations",
    "cumulative_truncation_rate",

    "optimization_count",

    "elapsed_sec",
]


EVAL_CSV_FIELDS = [

    "phase",

    "game",

    "result",
    "utility",

    "turns",

    "action_generations",
    "truncated_generations",
    "truncation_rate",

    "cumulative_wins",
    "cumulative_losses",
    "cumulative_ties",

    "cumulative_win_rate",

    "elapsed_sec",
]


SUMMARY_CSV_FIELDS = [

    "phase",

    "games",

    "wins",
    "losses",
    "ties",

    "win_rate",
    "loss_rate",
    "tie_rate",

    "action_generations",
    "truncated_generations",
    "truncation_rate",

    "elapsed_sec",
]


# ------------------------------------------------------------
# CSV 초기화
# ------------------------------------------------------------

def initialize_csv(
    path,
    fieldnames
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()


# ------------------------------------------------------------
# CSV에 한 행 추가
# ------------------------------------------------------------

def append_csv_row(
    path,
    fieldnames,
    row
):

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writerow(row)


# CSV 파일 생성 + header 기록

initialize_csv(
    TRAIN_CSV_PATH,
    TRAIN_CSV_FIELDS
)

initialize_csv(
    BEFORE_EVAL_CSV_PATH,
    EVAL_CSV_FIELDS
)

initialize_csv(
    AFTER_EVAL_CSV_PATH,
    EVAL_CSV_FIELDS
)

initialize_csv(
    SUMMARY_CSV_PATH,
    SUMMARY_CSV_FIELDS
)


# ------------------------------------------------------------
# 실험 설정 JSON 저장
# ------------------------------------------------------------

run_config = {

    "state_dim":
        STATE_DIM,

    "action_dim":
        ACTION_DIM,

    "device":
        str(DEVICE),

    "max_legal_actions":
        MAX_LEGAL_ACTIONS,

    "train_episodes":
        TRAIN_EPISODES,

    "pre_eval_games":
        PRE_EVAL_GAMES,

    "post_eval_games":
        POST_EVAL_GAMES,

    "gamma":
        GAMMA,

    "learning_rate":
        LEARNING_RATE,

    "batch_size":
        BATCH_SIZE,

    "replay_capacity":
        REPLAY_CAPACITY,

    "min_replay_size":
        MIN_REPLAY_SIZE,

    "eps_start":
        EPS_START,

    "eps_end":
        EPS_END,

    "eps_decay":
        EPS_DECAY,

    "tau":
        TAU,

    "reward_weights":
        REWARD_WEIGHTS,

    # 원본 코드의 evaluate()와 동일하게
    # seed_start 인자는 실제 RNG seed 설정에 사용하지 않는다.
    "before_eval_seed_start_argument":
        200000,

    "after_eval_seed_start_argument":
        300000,

    "seed_note":
        (
            "seed_start is retained from the original code, "
            "but evaluate() does not apply it to RNG."
        )
}


with open(
    CONFIG_JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        run_config,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 2. 환경 생성
# ============================================================

def make_env():

    env = rummikub_env.RummikubEnv(
        players=NUM_PLAYERS,
        max_legal_actions=MAX_LEGAL_ACTIONS
    )

    env.set_reward_weights(
        hand_normal=REWARD_WEIGHTS["hand_normal"],
        hand_2_12=REWARD_WEIGHTS["hand_2_12"],
        hand_1_13=REWARD_WEIGHTS["hand_1_13"],

        table_len3=REWARD_WEIGHTS["table_len3"],
        table_len4=REWARD_WEIGHTS["table_len4"],
        table_len5plus=REWARD_WEIGHTS["table_len5plus"],

        win=REWARD_WEIGHTS["win"],
        loss=REWARD_WEIGHTS["loss"],
        tie=REWARD_WEIGHTS["tie"]
    )

    return env


# ============================================================
# 3. DQN
#
# State  702 → encoder
# Action 699 → encoder
#
# 두 representation을 합쳐 Q(s,a) 하나를 출력
# ============================================================

class RummikubDQN(nn.Module):

    def __init__(self):

        super().__init__()

        self.state_encoder = nn.Sequential(
            nn.Linear(
                STATE_DIM,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(
                ACTION_DIM,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(
                256,
                256
            ),
            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                1
            )
        )


    def forward(
        self,
        state,
        action
    ):

        state_feature = self.state_encoder(
            state
        )

        action_feature = self.action_encoder(
            action
        )

        x = torch.cat(
            [
                state_feature,
                action_feature
            ],
            dim=-1
        )

        return self.q_head(
            x
        )


# ============================================================
# 4. Replay Buffer
#
# 다음 상태에서 가능한 행동도 Bellman target 계산에 필요.
#
# 1차 테스트에서는 MAX_LEGAL_ACTIONS=64이므로
# next_actions 자체를 저장한다.
#
# 메모리를 줄이기 위해 numpy float16으로 저장.
# 학습 직전에 float32로 변환.
# ============================================================

@dataclass
class Transition:

    state: np.ndarray
    action: np.ndarray

    reward: float

    next_state: object
    next_actions: object

    done: bool


class ReplayBuffer:

    def __init__(
        self,
        capacity
    ):

        self.buffer = deque(
            maxlen=capacity
        )


    def push(
        self,
        state,
        action,
        reward,
        next_state,
        next_actions,
        done
    ):

        self.buffer.append(
            Transition(

                state=np.asarray(
                    state,
                    dtype=np.float16
                ),

                action=np.asarray(
                    action,
                    dtype=np.float16
                ),

                reward=float(
                    reward
                ),

                next_state=(
                    None
                    if next_state is None
                    else np.asarray(
                        next_state,
                        dtype=np.float16
                    )
                ),

                next_actions=(
                    None
                    if next_actions is None
                    else np.asarray(
                        next_actions,
                        dtype=np.float16
                    )
                ),

                done=bool(
                    done
                )
            )
        )


    def sample(
        self,
        batch_size
    ):

        return random.sample(
            self.buffer,
            batch_size
        )


    def __len__(
        self
    ):

        return len(
            self.buffer
        )


# ============================================================
# 5. Network 생성
# ============================================================

policy_net = RummikubDQN().to(
    DEVICE
)

target_net = RummikubDQN().to(
    DEVICE
)

target_net.load_state_dict(
    policy_net.state_dict()
)

target_net.eval()


optimizer = optim.AdamW(
    policy_net.parameters(),
    lr=LEARNING_RATE,
    amsgrad=True
)


loss_function = nn.SmoothL1Loss()


memory = ReplayBuffer(
    REPLAY_CAPACITY
)


# ============================================================
# 6. epsilon
# ============================================================

global_step = 0


def get_epsilon():

    return EPS_END + (
        EPS_START - EPS_END
    ) * math.exp(
        -global_step / EPS_DECAY
    )


# ============================================================
# 7. 행동 선택
#
# epsilon 확률 → random
# 그 외 → argmax Q(s,a)
# ============================================================

def select_action(
    model,
    state,
    actions,
    epsilon
):

    # exploration
    if random.random() < epsilon:

        return random.randrange(
            len(actions)
        )


    # exploitation

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    )

    action_tensor = torch.tensor(
        np.asarray(
            actions
        ),
        dtype=torch.float32,
        device=DEVICE
    )

    action_count = (
        action_tensor.shape[0]
    )

    state_batch = (
        state_tensor
        .unsqueeze(0)
        .expand(
            action_count,
            -1
        )
    )


    with torch.no_grad():

        q_values = model(
            state_batch,
            action_tensor
        ).squeeze(-1)


    return int(
        torch.argmax(
            q_values
        ).item()
    )


# ============================================================
# 8. Random 상대
# ============================================================

def random_opponent_action(
    env
):

    actions = (
        env.legal_action_features()
    )

    if len(actions) == 0:

        raise RuntimeError(
            "상대 legal action이 0개입니다."
        )


    action_index = random.randrange(
        len(actions)
    )

    return env.step(
        action_index
    )


# ============================================================
# 9. Target network soft update
# ============================================================

def soft_update():

    with torch.no_grad():

        for (
            target_param,
            policy_param
        ) in zip(
            target_net.parameters(),
            policy_net.parameters()
        ):

            target_param.data.mul_(
                1.0 - TAU
            )

            target_param.data.add_(
                TAU
                * policy_param.data
            )


# ============================================================
# 10. DQN optimization
# ============================================================

def optimize_model():

    if len(memory) < max(
        BATCH_SIZE,
        MIN_REPLAY_SIZE
    ):

        return None


    batch = memory.sample(
        BATCH_SIZE
    )


    # --------------------------------------------------------
    # Q(s,a)
    # --------------------------------------------------------

    states = torch.tensor(

        np.stack([
            t.state
            for t in batch
        ]),

        dtype=torch.float32,
        device=DEVICE
    )


    actions = torch.tensor(

        np.stack([
            t.action
            for t in batch
        ]),

        dtype=torch.float32,
        device=DEVICE
    )


    rewards = torch.tensor(

        [
            t.reward
            for t in batch
        ],

        dtype=torch.float32,
        device=DEVICE
    )


    current_q = policy_net(
        states,
        actions
    ).squeeze(-1)


    # --------------------------------------------------------
    # max Q_target(s', a')
    #
    # 각 transition마다 가능한 action 수가 다르기 때문에
    # 개별적으로 계산한다.
    # --------------------------------------------------------

    next_q_values = torch.zeros(
        BATCH_SIZE,
        dtype=torch.float32,
        device=DEVICE
    )


    with torch.no_grad():

        for i, transition in enumerate(
            batch
        ):

            if transition.done:
                continue


            next_state = torch.tensor(
                transition.next_state,
                dtype=torch.float32,
                device=DEVICE
            )


            next_actions = torch.tensor(
                transition.next_actions,
                dtype=torch.float32,
                device=DEVICE
            )


            action_count = (
                next_actions.shape[0]
            )


            next_state_batch = (
                next_state
                .unsqueeze(0)
                .expand(
                    action_count,
                    -1
                )
            )


            q = target_net(
                next_state_batch,
                next_actions
            ).squeeze(-1)


            next_q_values[i] = (
                q.max()
            )


    # Bellman target

    target_q = (
        rewards
        + GAMMA
        * next_q_values
    )


    # --------------------------------------------------------
    # Huber loss
    # --------------------------------------------------------

    loss = loss_function(
        current_q,
        target_q
    )


    optimizer.zero_grad()

    loss.backward()


    # gradient 폭주 방지
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=10.0
    )


    optimizer.step()

    soft_update()


    return float(
        loss.item()
    )


# ============================================================
# 11. DQN 관점 terminal reward
#
# 상대가 게임을 끝낸 경우 C++ step 결과의 terminal_reward는
# 상대 관점이다.
#
# 그래서 DQN player=0 관점으로 다시 계산.
# ============================================================

def terminal_reward_for_dqn(
    env
):

    utility = env.terminal_utility(
        DQN_PLAYER
    )


    if utility > 0:

        return REWARD_WEIGHTS[
            "win"
        ]


    if utility < 0:

        return REWARD_WEIGHTS[
            "loss"
        ]


    return REWARD_WEIGHTS[
        "tie"
    ]


# ============================================================
# 12. 평가
#
# epsilon = 0
# DQN은 greedy
# 상대는 random
#
# 기존 평가 로직은 그대로 유지하면서
# 각 함수 호출 전/후에 DEBUG 출력을 넣어
# 어디에서 멈추는지 확인한다.
# ============================================================

def evaluate(
    model,
    games=100,
    seed_start=100000,
    phase="EVAL",
    csv_filename=None
):

    model.eval()

    wins = 0
    losses = 0
    ties = 0

    truncated_count = 0
    action_generation_count = 0

    eval_logs = []

    eval_start_time = time.perf_counter()

    for game in range(games):

        print(
            f"\n[DEBUG] ===== {phase} GAME {game + 1}/{games} START =====",
            flush=True
        )

        print(
            f"[DEBUG] game={game + 1} before make_env",
            flush=True
        )

        env = make_env()

        print(
            f"[DEBUG] game={game + 1} after make_env",
            flush=True
        )

        print(
            f"[DEBUG] game={game + 1} before reset_fixed_game",
            flush=True
        )

        reset_fixed_game(env)

        print(
            f"[DEBUG] game={game + 1} after reset_fixed_game",
            flush=True
        )

        game_turns = 0
        game_truncated_count = 0
        game_generation_count = 0

        while True:

            # ------------------------------------------------
            # terminal 검사
            # ------------------------------------------------

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} before terminal",
                flush=True
            )

            is_terminal = env.terminal()

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"after terminal={is_terminal}",
                flush=True
            )

            if is_terminal:
                break


            # ------------------------------------------------
            # 현재 플레이어 확인
            # ------------------------------------------------

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} before current_player",
                flush=True
            )

            player = env.current_player()

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"after current_player={player}",
                flush=True
            )


            # ------------------------------------------------
            # legal action 생성
            # ------------------------------------------------

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"player={player} "
                f"before legal_action_features",
                flush=True
            )

            actions = env.legal_action_features()

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"after legal_action_features "
                f"actions={len(actions)}",
                flush=True
            )

            action_generation_count += 1
            game_generation_count += 1


            # ------------------------------------------------
            # generation_truncated 확인
            # ------------------------------------------------

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"before generation_truncated",
                flush=True
            )

            is_truncated = env.generation_truncated()

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"after generation_truncated={is_truncated}",
                flush=True
            )

            if is_truncated:
                truncated_count += 1
                game_truncated_count += 1


            # ------------------------------------------------
            # 행동 선택
            # ------------------------------------------------

            if player == DQN_PLAYER:

                print(
                    f"[DEBUG] game={game + 1} "
                    f"turn={game_turns} "
                    f"before observation",
                    flush=True
                )

                state = env.observation(
                    DQN_PLAYER
                )

                print(
                    f"[DEBUG] game={game + 1} "
                    f"turn={game_turns} "
                    f"after observation",
                    flush=True
                )

                print(
                    f"[DEBUG] game={game + 1} "
                    f"turn={game_turns} "
                    f"before select_action",
                    flush=True
                )

                action_index = select_action(
                    model,
                    state,
                    actions,
                    epsilon=0.0
                )

                print(
                    f"[DEBUG] game={game + 1} "
                    f"turn={game_turns} "
                    f"after select_action "
                    f"selected={action_index}",
                    flush=True
                )

            else:

                print(
                    f"[DEBUG] game={game + 1} "
                    f"turn={game_turns} "
                    f"before random action selection",
                    flush=True
                )

                action_index = random.randrange(
                    len(actions)
                )

                print(
                    f"[DEBUG] game={game + 1} "
                    f"turn={game_turns} "
                    f"after random action selection "
                    f"selected={action_index}",
                    flush=True
                )


            # ------------------------------------------------
            # step
            # ------------------------------------------------

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"player={player} "
                f"before step",
                flush=True
            )

            env.step(
                action_index
            )

            print(
                f"[DEBUG] game={game + 1} "
                f"turn={game_turns} "
                f"player={player} "
                f"after step",
                flush=True
            )

            game_turns += 1


        # ----------------------------------------------------
        # 게임 종료 후 utility 계산
        # ----------------------------------------------------

        print(
            f"[DEBUG] game={game + 1} "
            f"before terminal_utility",
            flush=True
        )

        utility = env.terminal_utility(
            DQN_PLAYER
        )

        print(
            f"[DEBUG] game={game + 1} "
            f"after terminal_utility={utility}",
            flush=True
        )


        if utility > 0:
            wins += 1
            result = "W"

        elif utility < 0:
            losses += 1
            result = "L"

        else:
            ties += 1
            result = "T"


        elapsed = (
            time.perf_counter()
            - eval_start_time
        )


        eval_logs.append({
            "phase": phase,
            "game": game + 1,
            "result": result,
            "utility": float(utility),
            "turns": game_turns,

            "action_generations":
                game_generation_count,

            "truncated_generations":
                game_truncated_count,

            "truncation_rate":
                game_truncated_count
                / max(1, game_generation_count),

            "cumulative_wins":
                wins,

            "cumulative_losses":
                losses,

            "cumulative_ties":
                ties,

            "cumulative_win_rate":
                wins / (game + 1),

            "elapsed_sec":
                elapsed
        })


        print(
            f"[DEBUG] ===== {phase} GAME {game + 1}/{games} END "
            f"result={result} turns={game_turns} =====",
            flush=True
        )


        # 10게임마다 진행상황 출력
        if (
            game == 0
            or (game + 1) % 10 == 0
            or game + 1 == games
        ):

            print(
                f"[{phase}] "
                f"Game {game + 1:3d}/{games}"
                f" | W={wins:3d}"
                f" L={losses:3d}"
                f" T={ties:3d}"
                f" | win_rate={wins / (game + 1):.3f}"
                f" | trunc="
                f"{truncated_count / max(1, action_generation_count):.3f}",
                flush=True
            )


    model.train()


    # 평가 데이터 CSV 저장
    if csv_filename is not None:

        with open(
            csv_filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=eval_logs[0].keys()
            )

            writer.writeheader()

            writer.writerows(
                eval_logs
            )


    total_elapsed = (
        time.perf_counter()
        - eval_start_time
    )


    return {
        "phase": phase,
        "games": games,
        "win_rate": wins / games,
        "loss_rate": losses / games,
        "tie_rate": ties / games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "action_generations": action_generation_count,
        "truncated_generations": truncated_count,

        "truncation_rate": (
            truncated_count
            / max(
                1,
                action_generation_count
            )
        ),

        "elapsed_sec": total_elapsed
    }


# ============================================================
# 13. 학습 전 평가
# ============================================================

print(
    "\n===== BEFORE TRAINING ====="
)


before_result = evaluate(

    policy_net,

    games=PRE_EVAL_GAMES,

    seed_start=200000,

    phase="BEFORE",

    csv_filename=BEFORE_EVAL_CSV_PATH,

)


print(
    "\nBEFORE RESULT"
)

print(
    before_result
)


# ============================================================
# 14. 학습
# ============================================================

episode_rewards = []

loss_history = []

episode_results = []


truncated_count = 0
generation_count = 0


print(
    "\n===== TRAINING START ====="
)


training_start_time = (
    time.perf_counter()
)


for episode in range(
    TRAIN_EPISODES
):

    env = make_env()

    reset_fixed_game(
        env
    )


    # --------------------------------------------------------
    # episode별 통계
    # --------------------------------------------------------

    episode_generation_count = 0

    episode_truncated_count = 0

    episode_dqn_steps = 0

    episode_losses = []


    # --------------------------------------------------------
    # 상대가 먼저 시작하는 episode
    # --------------------------------------------------------

    if (
        not env.terminal()
        and env.current_player()
        == OPPONENT_PLAYER
    ):

        random_opponent_action(
            env
        )


    episode_reward = 0.0


    # 최초 DQN 상태와 행동 후보를 한 번만 생성

    state = env.observation(
        DQN_PLAYER
    )


    legal_actions = (
        env.legal_action_features()
    )


    generation_count += 1
    episode_generation_count += 1


    if env.generation_truncated():

        truncated_count += 1
        episode_truncated_count += 1


    if len(
        legal_actions
    ) == 0:

        raise RuntimeError(
            "DQN legal action이 0개입니다."
        )


    # --------------------------------------------------------
    # 한 episode
    # --------------------------------------------------------

    while not env.terminal():

        if (
            env.current_player()
            != DQN_PLAYER
        ):

            raise RuntimeError(
                "DQN 턴 동기화 오류"
            )


        # ====================================================
        # a_t
        # ====================================================

        epsilon = (
            get_epsilon()
        )


        action_index = select_action(
            policy_net,
            state,
            legal_actions,
            epsilon
        )


        chosen_action = legal_actions[
            action_index
        ]


        # ====================================================
        # DQN action
        # ====================================================

        own_result = env.step(
            action_index
        )


        reward = float(
            own_result.reward
        )


        global_step += 1
        episode_dqn_steps += 1


        # ====================================================
        # DQN 행동으로 게임 종료
        # ====================================================

        if own_result.done:

            memory.push(

                state=state,

                action=chosen_action,

                reward=reward,

                next_state=None,

                next_actions=None,

                done=True
            )


            loss = optimize_model()


            if loss is not None:

                loss_history.append(
                    loss
                )

                episode_losses.append(
                    loss
                )


            episode_reward += (
                reward
            )


            break


        # ====================================================
        # 상대 Random 행동
        # ====================================================

        if (
            env.current_player()
            != OPPONENT_PLAYER
        ):

            raise RuntimeError(
                "상대 턴 동기화 오류"
            )


        random_opponent_action(
            env
        )


        # ====================================================
        # 상대 행동으로 게임 종료
        # ====================================================

        if env.terminal():

            # DQN 관점 패배/승리/무승부 보상

            reward += (
                terminal_reward_for_dqn(
                    env
                )
            )


            memory.push(

                state=state,

                action=chosen_action,

                reward=reward,

                next_state=None,

                next_actions=None,

                done=True
            )


            loss = optimize_model()


            if loss is not None:

                loss_history.append(
                    loss
                )

                episode_losses.append(
                    loss
                )


            episode_reward += (
                reward
            )


            break


        # ====================================================
        # s_(t+1)
        #
        # 다시 DQN의 차례가 되어야 한다.
        # ====================================================

        if (
            env.current_player()
            != DQN_PLAYER
        ):

            raise RuntimeError(
                "next state가 DQN 턴이 아닙니다."
            )


        next_state = env.observation(
            DQN_PLAYER
        )


        next_actions = (
            env.legal_action_features()
        )


        generation_count += 1
        episode_generation_count += 1


        if env.generation_truncated():

            truncated_count += 1
            episode_truncated_count += 1


        # ====================================================
        # Replay Buffer
        # ====================================================

        memory.push(

            state=state,

            action=chosen_action,

            reward=reward,

            next_state=next_state,

            next_actions=next_actions,

            done=False
        )


        # ====================================================
        # DQN update
        # ====================================================

        loss = optimize_model()


        if loss is not None:

            loss_history.append(
                loss
            )

            episode_losses.append(
                loss
            )


        episode_reward += (
            reward
        )


        # 다음 반복에서 같은 상태의
        # legal actions를 다시 생성하지 않도록 재사용

        state = next_state
        legal_actions = next_actions


    # ========================================================
    # episode 종료 결과
    # ========================================================

    utility = env.terminal_utility(
        DQN_PLAYER
    )


    if utility > 0:

        episode_result = "W"
        episode_win_value = 1


    elif utility < 0:

        episode_result = "L"
        episode_win_value = 0


    else:

        episode_result = "T"
        episode_win_value = 0


    episode_rewards.append(
        episode_reward
    )


    episode_results.append(
        episode_win_value
    )


    # --------------------------------------------------------
    # 최근 20 episode reward
    # --------------------------------------------------------

    recent_reward = np.mean(
        episode_rewards[
            -20:
        ]
    )


    # --------------------------------------------------------
    # 최근 20 episode 승률
    # --------------------------------------------------------

    win_rate20 = np.mean(
        episode_results[
            -20:
        ]
    )


    # --------------------------------------------------------
    # 이번 episode의 평균 loss
    # --------------------------------------------------------

    episode_loss = (

        np.mean(
            episode_losses
        )

        if episode_losses

        else float(
            "nan"
        )
    )


    # --------------------------------------------------------
    # 최근 optimization 100회의 평균 loss
    # --------------------------------------------------------

    recent_loss = (

        np.mean(
            loss_history[
                -100:
            ]
        )

        if loss_history

        else float(
            "nan"
        )
    )


    # --------------------------------------------------------
    # truncation 통계
    # --------------------------------------------------------

    episode_truncation_rate = (

        episode_truncated_count
        / max(
            1,
            episode_generation_count
        )
    )


    cumulative_truncation_rate = (

        truncated_count
        / max(
            1,
            generation_count
        )
    )


    # --------------------------------------------------------
    # 시간
    # --------------------------------------------------------

    training_elapsed = (

        time.perf_counter()
        - training_start_time
    )


    # --------------------------------------------------------
    # CSV용 데이터
    # --------------------------------------------------------

    train_log = {

        "episode":
            episode + 1,

        "global_step":
            global_step,

        "result":
            episode_result,

        "utility":
            float(
                utility
            ),

        "episode_reward":
            episode_reward,

        "reward_ma20":
            float(
                recent_reward
            ),

        "win_rate20":
            float(
                win_rate20
            ),

        "epsilon":
            get_epsilon(),

        "replay_size":
            len(
                memory
            ),

        "episode_loss":
            float(
                episode_loss
            ),

        "loss_ma100":
            float(
                recent_loss
            ),

        "episode_dqn_steps":
            episode_dqn_steps,

        "episode_action_generations":
            episode_generation_count,

        "episode_truncated_generations":
            episode_truncated_count,

        "episode_truncation_rate":
            episode_truncation_rate,

        "cumulative_action_generations":
            generation_count,

        "cumulative_truncated_generations":
            truncated_count,

        "cumulative_truncation_rate":
            cumulative_truncation_rate,

        "optimization_count":
            len(
                loss_history
            ),

        "elapsed_sec":
            training_elapsed,
    }


    # --------------------------------------------------------
    # 매 episode마다 CSV에 즉시 저장
    # --------------------------------------------------------

    append_csv_row(
        TRAIN_CSV_PATH,
        TRAIN_CSV_FIELDS,
        train_log
    )


    # --------------------------------------------------------
    # 진행상황 출력
    #
    # 원본과 동일하게 매 episode 출력
    # --------------------------------------------------------

    if (
        episode == 0
        or (episode + 1) % 1 == 0
    ):

        print(
            f"[TRAIN] "
            f"Ep {episode + 1:4d}/{TRAIN_EPISODES}"
            f" | result={episode_result}"
            f" | win20={win_rate20:.3f}"
            f" | reward={episode_reward:8.2f}"
            f" | R20={recent_reward:8.2f}"
            f" | eps={get_epsilon():.3f}"
            f" | loss_ep={episode_loss:.4f}"
            f" | loss100={recent_loss:.4f}"
            f" | replay={len(memory):4d}"
        )


print(
    "\n===== TRAINING COMPLETE ====="
)


# ============================================================
# 15. 학습 후 평가
# ============================================================

print(
    "\n===== AFTER TRAINING ====="
)


after_result = evaluate(

    policy_net,

    games=POST_EVAL_GAMES,

    seed_start=300000,

    phase="AFTER",

    csv_filename=AFTER_EVAL_CSV_PATH,

)


print(
    "\nAFTER RESULT"
)

print(
    after_result
)


# ============================================================
# 16. Summary CSV 저장
# ============================================================

append_csv_row(

    SUMMARY_CSV_PATH,

    SUMMARY_CSV_FIELDS,

    before_result
)


append_csv_row(

    SUMMARY_CSV_PATH,

    SUMMARY_CSV_FIELDS,

    after_result
)


# ============================================================
# 17. 최종 요약
# ============================================================

print(
    "\n===================================="
)

print(
    "DQN TEST RESULT"
)

print(
    "===================================="
)


print(
    "Before training win rate =",
    f"{before_result['win_rate']:.3f}"
)


print(
    "After training win rate  =",
    f"{after_result['win_rate']:.3f}"
)


print(
    "Training action truncation rate =",
    f"{truncated_count / max(1, generation_count):.3f}"
)


print(
    "Evaluation truncation rate =",
    f"{after_result['truncation_rate']:.3f}"
)


print(
    "Replay buffer size =",
    len(
        memory
    )
)


print(
    "Optimization count =",
    len(
        loss_history
    )
)


print(
    "\n===================================="
)

print(
    "SAVED FILES"
)

print(
    "===================================="
)


print(
    "Config       :",
    CONFIG_JSON_PATH
)

print(
    "Train log    :",
    TRAIN_CSV_PATH
)

print(
    "Before eval  :",
    BEFORE_EVAL_CSV_PATH
)

print(
    "After eval   :",
    AFTER_EVAL_CSV_PATH
)

print(
    "Summary      :",
    SUMMARY_CSV_PATH
)