import os
import shutil

from gamebrain import GameBrain


def setup_state_dir(tmp="./rps_state_test_alt"):
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    return tmp


def simulate_alternating(brain: GameBrain, uid: str, rounds: int = 40):
    pattern = [2, 1]  # user alternates: Scissors(2), Paper(1)
    wins = draws = losses = 0
    for t in range(rounds):
        ai, _ = brain.predict(uid, None)
        u = pattern[t % 2]
        # outcome from AI perspective
        if ai == u:
            res = 'draw'
            draws += 1
        elif (ai + 1) % 3 == u:
            # user beats AI -> AI lose
            res = 'lose'
            losses += 1
        else:
            res = 'win'
            wins += 1
        brain.feedback(uid, ai, u, 300, res)
    return wins, draws, losses


def test_ai_handles_two_move_alternation():
    state_dir = setup_state_dir()
    brain = GameBrain(state_dir=state_dir, remember_history=False)
    uid = "alt_user"
    wins, draws, losses = simulate_alternating(brain, uid, rounds=50)
    # After learning phase, AI should achieve strong win rate.
    # Allow some volatility but expect >= 70% wins ignoring draws.
    total_non_draw = max(1, wins + losses)
    assert wins / total_non_draw >= 0.7
