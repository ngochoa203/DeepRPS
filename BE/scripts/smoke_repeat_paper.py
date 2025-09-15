import sys
from gamebrain import GameBrain

# Simple smoke test: user always plays Paper (1)
brain = GameBrain(state_dir="./rps_state", remember_history=False)
uid = "smoke_paper"

wins = draws = losses = 0
for t in range(10):
    ai_move, meta = brain.predict(uid)
    user_move = 1  # Paper
    # Determine result from AI perspective
    if ai_move == 2:  # Scissors beats Paper
        result = "win"
        wins += 1
    elif ai_move == 1:  # Paper vs Paper
        result = "draw"
        draws += 1
    else:  # Rock loses to Paper
        result = "lose"
        losses += 1
    brain.feedback(uid, ai_move, user_move, 300, result)
    print(f"round={t+1} ai_move={ai_move} result={result} policy={meta.get('policy')} family={meta.get('family')}")

print(f"Summary: wins={wins} draws={draws} losses={losses}")
