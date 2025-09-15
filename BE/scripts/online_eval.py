import json
import random
from typing import Dict

from gamebrain import GameBrain


def simulate_session(brain: GameBrain, uid: str, n_rounds: int = 200, human_type: str = "mixed") -> Dict:
    # simple stochastic opponents
    def human_move(ai_move: int, t: int) -> int:
        if human_type == "counter":
            return (ai_move + 1) % 3  # tries to beat us
        elif human_type == "sticky":
            return 0 if random.random() < 0.6 else random.randrange(3)
        else:  # mixed
            return random.randrange(3)

    A = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ]
    wins = 0
    for t in range(n_rounds):
        ai, meta = brain.predict(uid, None)
        hm = human_move(ai, t)
        res_m = A[ai][hm]
        res = 'win' if res_m == 1 else ('draw' if res_m == 0 else 'lose')
        if res == 'win':
            wins += 1
        brain.feedback(uid, ai, hm, dt_ms=350, result=res)
    return {"win_rate": wins / n_rounds}


def run_ab():
    # A: bandit off (alpha=0), B: bandit on
    A_brain = GameBrain(state_dir="./rps_state_A", remember_history=False)
    B_brain = GameBrain(state_dir="./rps_state_B", remember_history=False)
    # turn off bandit by setting alpha ~0
    for uid in ["A_user"]:
        A_brain._get_or_create_user(uid).bandit.alpha = 0.0

    ra = simulate_session(A_brain, "A_user", human_type="mixed")
    rb = simulate_session(B_brain, "B_user", human_type="mixed")
    print(json.dumps({"A": ra, "B": rb, "uplift": rb["win_rate"] - ra["win_rate"]}, indent=2))


if __name__ == "__main__":
    run_ab()
