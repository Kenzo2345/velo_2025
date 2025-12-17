import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from model import State, run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Serial parameter sweep for the bike simulation")

    parser.add_argument("--params", type=str, required=True, help="CSV file with parameter combinations")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory (e.g. results/)")
    parser.add_argument("--plot", action="store_true", help="Generate plots after run")
    parser.add_argument("--smooth-window", type=int, default=1, help="Smoothing window (default: 1 = no smoothing)")

    return parser.parse_args()


def main():
    args = parse_args()

    params_path = Path(args.params)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_params = pd.read_csv(params_path)

    all_metrics_rows = []
    all_timeseries = []

    for run_id, row in df_params.iterrows():
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
        all_timeseries.append(ts_df)

        final_mailly = ts_df["mailly"].iloc[-1]
        final_moulin = ts_df["moulin"].iloc[-1]
        final_balance = ts_df["balance"].iloc[-1]
        unmet_mailly = ts_df["unmet_mailly"].iloc[-1]
        unmet_moulin = ts_df["unmet_moulin"].iloc[-1]

        all_metrics_rows.append(
            {
                "run_id": run_id,
                "steps": steps,
                "p1": p1,
                "p2": p2,
                "init_mailly": init_mailly,
                "init_moulin": init_moulin,
                "seed": seed,
                "final_mailly": final_mailly,
                "final_moulin": final_moulin,
                "final_balance": final_balance,
                "unmet_mailly": unmet_mailly,
                "unmet_moulin": unmet_moulin,
            }
        )

    metrics_df = pd.DataFrame(all_metrics_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    if args.plot:
        ts_all = pd.concat(all_timeseries, ignore_index=True)

        w = max(1, int(args.smooth_window))
        if w > 1:
            ts_all = ts_all.sort_values(["run_id", "time"])
            ts_all["mailly_s"] = ts_all.groupby("run_id")["mailly"].transform(lambda s: s.rolling(w, min_periods=1).mean())
            ts_all["moulin_s"] = ts_all.groupby("run_id")["moulin"].transform(lambda s: s.rolling(w, min_periods=1).mean())
            ts_all["balance_s"] = ts_all.groupby("run_id")["balance"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        else:
            ts_all["mailly_s"] = ts_all["mailly"]
            ts_all["moulin_s"] = ts_all["moulin"]
            ts_all["balance_s"] = ts_all["balance"]

        plt.figure(figsize=(10, 6))

        for run_id, g in ts_all.groupby("run_id"):
            plt.plot(g["time"], g["mailly_s"], alpha=0.8, label=f"run{run_id}-mailly")
            plt.plot(g["time"], g["moulin_s"], alpha=0.8, label=f"run{run_id}-moulin")
            plt.plot(g["time"], g["balance_s"], alpha=0.8, linestyle="--", label=f"run{run_id}-balance")

        plt.xlabel("time")
        plt.ylabel("bikes / balance")
        plt.title("Mailly / Moulin / Balance (serial sweep)")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "metrics_3plot.png", dpi=150)
        plt.close()
if __name__ == "__main__":
    main()
