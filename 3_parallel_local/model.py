from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass
class State:
    mailly: int
    moulin: int
    unmet_mailly: int = 0
    unmet_moulin: int = 0


def step(state: State, p1: float, p2: float, rng: np.random.Generator, metrics: Dict[str, int]) -> State:
    # Mailly -> Moulin
    if rng.random() < p1:
        if state.mailly > 0:
            state.mailly -= 1
            state.moulin += 1
        else:
            state.unmet_mailly += 1
            metrics["unmet_mailly"] += 1

    # Moulin -> Mailly
    if rng.random() < p2:
        if state.moulin > 0:
            state.moulin -= 1
            state.mailly += 1
        else:
            state.unmet_moulin += 1
            metrics["unmet_moulin"] += 1

    return state


def run_simulation(
    initial_mailly: int,
    initial_moulin: int,
    steps: int,
    p1: float,
    p2: float,
    seed: int,
) -> Dict[str, list]:
    rng = np.random.default_rng(seed)
    state = State(mailly=int(initial_mailly), moulin=int(initial_moulin))

    counters = {"unmet_mailly": 0, "unmet_moulin": 0}

    out: Dict[str, list] = {
        "time": [],
        "mailly": [],
        "moulin": [],
        "balance": [],
        "unmet_mailly": [],
        "unmet_moulin": [],
    }

    for t in range(steps + 1):
        out["time"].append(t)
        out["mailly"].append(state.mailly)
        out["moulin"].append(state.moulin)
        out["balance"].append(state.mailly - state.moulin)
        out["unmet_mailly"].append(counters["unmet_mailly"])
        out["unmet_moulin"].append(counters["unmet_moulin"])

        if t == steps:
            break

        step(state, p1, p2, rng, counters)

    return out
