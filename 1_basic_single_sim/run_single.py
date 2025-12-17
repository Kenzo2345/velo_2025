import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from model import State, run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single bike-sharing simulation.")

    parser.add_argument("--steps", type=int, required=True, help="Number of simulation steps")
    parser.add_argument("--p1", type=float, required=True, help="Prob Mailly -> Moulin")
    parser.add_argument("--p2", type=float, required=True, help="Prob Moulin -> Mailly")
    parser.add_argument("--init-mailly", dest="init_mailly", type=int, required=True, help="Initial bikes at Mailly")
    parser.add_argument("--init-moulin", dest="init_moulin", type=int, required=True, help="Initial bikes at Moulin")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--out-csv", dest="out_csv", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    return parser.parse_args()


def main():
    args = parse_args()

    out_csv = Path(args.out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df, metrics = run_simulation(
        initial_mailly=args.init_mailly,
        initial_moulin=args.init_moulin,
        steps=args.steps,
        p1=args.p1,
        p2=args.p2,
        seed=args.seed,
    )

    df.to_csv(out_csv, index=False)

    metrics_path = out_dir / f"{out_csv.stem}_metrics.tsv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}\t{v}\n")

    if args.plot:
        plt.figure()
        plt.plot(df["time"], df["mailly"], label="Mailly")
        plt.plot(df["time"], df["moulin"], label="Moulin")
        plt.xlabel("time")
        plt.ylabel("bikes")
        plt.legend()
        plot_path = out_dir / f"{out_csv.stem}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    main()
