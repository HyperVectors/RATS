import argparse
import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 55,
    'axes.labelsize': 50,
    'xtick.labelsize': 45,
    'ytick.labelsize': 45,
    'legend.fontsize': 30,
    'figure.titlesize': 55
})


def _first_present_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the expected columns were found. Expected one of: {candidates}, got: {list(df.columns)}")


def save_legend_strip(output_dir: pathlib.Path) -> None:
    legend_dir = output_dir / "legend"
    legend_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16, 2))
    ax = fig.add_subplot(111)
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="C0", lw=3, marker="o", markersize=10, label="RATSpy"),
        Line2D([0], [0], color="orange", lw=3, marker="x", markersize=12, label="tsaug"),
    ]
    ax.legend(handles=handles, loc="center", ncol=2, fontsize=50, frameon=True)
    fig.savefig(legend_dir / "legend_strip.eps", bbox_inches="tight")
    fig.savefig(legend_dir / "legend_strip.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_memory_vs_size(csv_path: pathlib.Path, dataset_name: str, output_dir: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)

    x_col = _first_present_column(df, ["Dataset Size", "Dataset_size", "Dataset_size".replace("_", " ")])
    rp_col = _first_present_column(df, ["RATSpy Peak Memory (MB)", "RATSpy_peak_mem_MB", "RATSpy_peak_mem_MB".replace("_", " ")])
    ts_col = _first_present_column(df, ["tsaug Peak Memory (MB)", "tsaug_peak_mem_MB", "tsaug_peak_mem_MB".replace("_", " ")])

    plt.figure(figsize=(24, 12))
    plt.plot(df[x_col], df[rp_col], label="RATSpy", marker="o", linewidth=3, markersize=10)
    if df[ts_col].notna().any():
        plt.plot(df[x_col], df[ts_col], label="tsaug", marker="x", linewidth=3, markersize=12, color="orange")

    plt.xlabel("Dataset Size", fontsize=50)
    plt.ylabel("Peak Memory Usage (MB)", fontsize=50)
    plt.title(f"Memory Usage vs Dataset Size - {dataset_name}", fontsize=55, pad=30)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.grid(alpha=0.3)
    # No legend; save standalone legend strip
    save_legend_strip(output_dir)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{dataset_name}_memory_vs_size.eps", bbox_inches="tight")
    plt.savefig(output_dir / f"{dataset_name}_memory_vs_size.pdf", bbox_inches="tight")
    plt.close()


def plot_time_vs_size(csv_path: pathlib.Path, dataset_name: str, output_dir: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)

    x_col = _first_present_column(df, ["Dataset Size", "Dataset_size", "Dataset_size".replace("_", " ")])
    rp_col = _first_present_column(df, ["RATSpy_time_sec", "RATSpy time sec", "RATSpy Time (sec)"])
    ts_col = _first_present_column(df, ["tsaug_time_sec", "tsaug time sec", "tsaug Time (sec)"])

    plt.figure(figsize=(24, 12))
    # Convert seconds to milliseconds for plotting
    rp_ms = df[rp_col] * 1000.0
    plt.plot(df[x_col], rp_ms, label="RATSpy", marker="o", linewidth=3, markersize=10)
    if df[ts_col].notna().any():
        ts_ms = df[ts_col] * 1000.0
        plt.plot(df[x_col], ts_ms, label="tsaug", marker="x", linewidth=3, markersize=12, color="orange")

    plt.xlabel("Dataset Size", fontsize=50)
    plt.ylabel("Time (ms)", fontsize=50)
    plt.title(f"Time vs Dataset Sizes - {dataset_name}", fontsize=55, pad=30)
    plt.xticks(fontsize=45)
    plt.yticks(fontsize=45)
    plt.grid(alpha=0.3)
    # No legend; save standalone legend strip
    save_legend_strip(output_dir)
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{dataset_name}_time_vs_size.eps", bbox_inches="tight")
    plt.savefig(output_dir / f"{dataset_name}_time_vs_size.pdf", bbox_inches="tight")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot memory/time vs dataset size from CSV files.")
    parser.add_argument("--dataset", type=str, default="Car", help="Dataset name for titles and filenames")
    parser.add_argument("--memory_csv", type=pathlib.Path, default=pathlib.Path("./Car_memory_vs_size.csv"),
                        help="Path to memory vs size CSV")
    parser.add_argument("--time_csv", type=pathlib.Path, default=pathlib.Path("./Car_time_vs_size.csv"),
                        help="Path to time vs size CSV")
    parser.add_argument("--output_dir", type=pathlib.Path, default=pathlib.Path("./results"),
                        help="Directory to save the generated plots")
    args = parser.parse_args()

    if args.memory_csv.exists():
        plot_memory_vs_size(args.memory_csv, args.dataset, args.output_dir)
    else:
        print(f"Skipping memory plot; file not found: {args.memory_csv}")

    if args.time_csv.exists():
        plot_time_vs_size(args.time_csv, args.dataset, args.output_dir)
    else:
        print(f"Skipping time plot; file not found: {args.time_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


