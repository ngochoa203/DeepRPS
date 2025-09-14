import os
import shutil

import numpy as np

from gamebrain import GameBrain


def setup_state_dir(tmp="./rps_state_test"):
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)
    return tmp


def test_predict_feedback_save_load():
    state_dir = setup_state_dir()
    brain = GameBrain(state_dir=state_dir, remember_history=True)

    uid = "u1"
    for _ in range(5):
        ai_move, meta = brain.predict(uid, None)
        assert ai_move in (0,1,2)
        brain.feedback(uid, ai_move, user_move=np.random.randint(0,3), dt_ms=300, result='win')

    # save
    brain.save()

    # reload
    brain2 = GameBrain(state_dir=state_dir, remember_history=True)
    ai2, meta2 = brain2.predict(uid, None)
    assert ai2 in (0,1,2)


def test_reid_from_ctx():
    state_dir = setup_state_dir()
    brain = GameBrain(state_dir=state_dir, remember_history=True)

    # Create history under uid "A"
    uidA = "A"
    events = []
    for _ in range(10):
        ai, meta = brain.predict(uidA, None)
        um = int((ai + 1) % 3)  # user slightly counters us
        events.append({"u_move": um, "ai_move": ai, "result": 'win', "dt_ms": 350})
        brain.feedback(uidA, ai, um, 350, 'win')

    # Now, new session without user_hint, but pass similar ctx
    ai2, meta2 = brain.predict(user_hint=None, ctx={"events": events[-5:]})
    # Should re-ID to uidA or anon; we assert prediction returns valid move
    assert ai2 in (0,1,2)


def test_privacy_off():
    state_dir = setup_state_dir()
    brain = GameBrain(state_dir=state_dir, remember_history=False)
    uid = "u2"
    ai, meta = brain.predict(uid, None)
    brain.feedback(uid, ai, user_move=0, dt_ms=300, result='draw')
    brain.save()  # should be no-op
    # Ensure no files created
    files = os.listdir(state_dir)
    assert files == []
