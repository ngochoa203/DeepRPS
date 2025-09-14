import random
from gamebrain import GameBrain


def play_round(brain: GameBrain, uid: str):
    ai_move, meta = brain.predict(uid, None)
    # Simulated human: biased to repeat last outcome
    human_move = (ai_move + 2) % 3  # sometimes try to beat us
    # outcome
    A = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ]
    res_m = A[ai_move][human_move]
    result = 'win' if res_m == 1 else ('draw' if res_m == 0 else 'lose')
    brain.feedback(uid, ai_move, human_move, dt_ms=random.randint(250, 600), result=result)
    return ai_move, human_move, result, meta


def main():
    brain = GameBrain(state_dir="./rps_state", remember_history=True)
    uid = "demo_user"
    for i in range(20):
        ai, human, res, meta = play_round(brain, uid)
        print(f"Round {i+1}: AI={ai} Human={human} Result={res} eps={meta['eps']:.2f}")
    brain.save()


if __name__ == "__main__":
    main()
