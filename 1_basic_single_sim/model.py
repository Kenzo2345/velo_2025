from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd


@dataclass
class State:
    """Represents the state of bikes at two stations.

    Attributes:
        mailly: Number of bikes at Mailly station
        moulin: Number of bikes at Moulin station
    """

    mailly: int
    moulin: int
    unmet_mailly: int = 0
    unmet_moulin: int = 0


def step(
    state: State,
    p1: float,
    p2: float,
    rng: np.random.Generator,
    metrics: Dict[str, int],
) -> State:
    #Mailly -> Moulin
    if rng.random() < p1:
        if state.mailly > 0:
            state.mailly -= 1
            state.moulin += 1
        else:
            state.unmet_mailly += 1
            metrics["unmet_mailly"] += 1

    #Moulin -> Mailly
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
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rng = np.random.default_rng(seed)

    state = State(mailly=int(initial_mailly), moulin=int(initial_moulin))

    metrics: Dict[str, int] = {
        "unmet_mailly": 0,
        "unmet_moulin": 0,
    }

    rows = [{"time": 0, "mailly": state.mailly, "moulin": state.moulin}]

    for t in range(1, steps + 1):
        step(state, p1, p2, rng, metrics)
        rows.append({"time": t, "mailly": state.mailly, "moulin": state.moulin})

    df = pd.DataFrame(rows, columns=["time", "mailly", "moulin"])

    metrics["mailly"] = state.mailly
    metrics["moulin"] = state.moulin
    metrics["final_imbalance"] = state.mailly - state.moulin

    return df, metrics
