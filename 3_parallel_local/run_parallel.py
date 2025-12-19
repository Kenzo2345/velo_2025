import argparse
from pathlib import Path
import os
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt

from model import run_simulation


def parse_args():
    p = argparse.ArgumentParser(description="Parallel parameter sweep (multiprocessing)")
    p.add_argument("--params", required=True, help="CSV file with parameter combinations")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--workers", default="auto", help="Number of workers or 'auto'")
    p.add_argument("--plot", action="store_true", help="Generate plots")
    return p.parse_args()


def _one_run(args_tuple):
    run_id, row = args_tuple
    steps = int(row["steps"])
    p1 = float(row["p1"])
    p2 = float(row["p2"])
    init_mailly = int(row["init_mailly"])
    init_moulin = int(row["init_moulin"])
    seed = int(row["seed"])

    ts = run_simulation(
        initial_mailly=init_mailly,
        initial_moulin=init_moulin,
        steps=steps,
        p1=p1,
        p2=p2,
        seed=seed,
    )
    ts_df = pd.DataFrame(ts)
    ts_df["run_id"] = run_id
    ts_df["p1"] = p1
    ts_df["p2"] = p2

    metrics = {
        "run_id": run_id,
        "steps": steps,
        "p1": p1,
        "p2": p2,
        "init_mailly": init_mailly,
        "init_moulin": init_moulin,
        "seed": seed,
        "final_mailly": int(ts_df["mailly"].iloc[-1]),
        "final_moulin": int(ts_df["moulin"].iloc[-1]),
        "final_balance": int(ts_df["balance"].iloc[-1]),
        "unmet_mailly": int(ts_df["unmet_mailly"].iloc[-1]),
        "unmet_moulin": int(ts_df["unmet_moulin"].iloc[-1]),
    }
    return metrics, ts_df


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_params = pd.read_csv(args.params)

    if args.workers == "auto":
        n_workers = os.cpu_count() or 1
    else:
        n_workers = max(1, int(args.workers))

    jobs = list(df_params.iterrows())

    with mp.Pool(processes=n_workers) as pool:
        results = list(pool.imap_unordered(_one_run, jobs))

    metrics_rows = [m for (m, _) in results]
    ts_all = pd.concat([ts for (_, ts) in results], ignore_index=True)

    pd.DataFrame(metrics_rows).sort_values("run_id").to_csv(out_dir / "metrics.csv", index=False)

    if args.plot:
        plt.figure(figsize=(10, 6))
        for run_id, g in ts_all.groupby("run_id"):
            plt.plot(g["time"], g["mailly"], label=f"run{run_id}-mailly")
            plt.plot(g["time"], g["moulin"], label=f"run{run_id}-moulin")
            plt.plot(g["time"], g["balance"], linestyle="--", label=f"run{run_id}-balance")
        plt.xlabel("time")
        plt.ylabel("bikes / balance")
        plt.title("Mailly / Moulin / Balance (multiprocessing)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "metrics_3plot.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
