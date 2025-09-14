import json
import math
import os
from collections import defaultdict

import numpy as np

from gamebrain import GameBrain


def log_loss(p, y):
    return -math.log(max(1e-8, p[y]))


def ece(probs_list, labels, n_bins=10):
    # Expected Calibration Error (multiclass, one-vs-max)
    confs = np.array([p.max() for p in probs_list])
    preds = np.array([int(np.argmax(p)) for p in probs_list])
    labels = np.array(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    total = len(labels)
    e = 0.0
    for i in range(n_bins):
        m = (confs > bins[i]) & (confs <= bins[i + 1])
        if not np.any(m):
            continue
        acc = np.mean(preds[m] == labels[m])
        conf = float(np.mean(confs[m]))
        e += (np.sum(m) / total) * abs(acc - conf)
    return float(e)


def run_offline(data_path: str):
    # data format: list of {user_id, events:[{u_move, ai_move?, outcome?, dt_ms?}, ...]}
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    brain = GameBrain(state_dir="./rps_state_eval", remember_history=False)
    probs = []
    labels = []

    for sess in data:
        uid = sess.get("user_id")
        events = sess.get("events", [])
        for e in events:
            _, meta = brain.predict(uid, None)
            probs.append(np.array(meta["p_opp"]))
            labels.append(int(e["u_move"]))
            # fake feedback to move window forward
            ai = int(np.argmax(meta["p_opp"]))
            res = 'draw'
            brain.feedback(uid, ai, int(e["u_move"]), int(e.get("dt_ms", 350)), res)

    ll = np.mean([log_loss(p, y) for p, y in zip(probs, labels)])
    e = ece(probs, labels)
    print(json.dumps({"log_loss": ll, "ece": e}, indent=2))


if __name__ == "__main__":
    import sys
    run_offline(sys.argv[1])
