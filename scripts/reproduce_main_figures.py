"""Reproduce the five manuscript figures from publication-level aggregates only."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmrv_metrics import TIER_ORDER, information_saturation  # noqa: E402


DATA = ROOT / "data" / "cmrv_published_aggregate_results.csv"
OUT = ROOT / "outputs"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
GRAY = "#777777"
LIGHT_GRAY = "#E7E7E7"
RED = "#B2182B"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> pd.DataFrame:
    frame = pd.read_csv(DATA, dtype=str, keep_default_na=False)
    required = {
        "figure",
        "panel",
        "task",
        "scenario",
        "stage",
        "tier",
        "metric",
        "estimate",
        "ci_lower",
        "ci_upper",
        "category",
        "status",
        "note",
    }
    if set(frame.columns) != required:
        raise ValueError(f"unexpected public-data columns: {list(frame.columns)}")
    forbidden = {"patient_id", "record_id", "event_count", "n_patients", "n_records"}
    if forbidden.intersection(frame.columns):
        raise ValueError("patient- or cohort-level fields are forbidden")
    for column in ("estimate", "ci_lower", "ci_upper"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def subset(frame: pd.DataFrame, figure: str, metric: str) -> pd.DataFrame:
    return frame[(frame["figure"] == figure) & (frame["metric"] == metric)].copy()


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    metadata = {"Creator": "CMRV public reproduction", "CreationDate": None, "ModDate": None}
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def figure_1() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.25), gridspec_kw={"wspace": 0.28})
    states = ["I0\nDemographics", "I1\nBaseline health", "I2\nEncounter context", "I3\nProximal process"]
    for index, label in enumerate(states):
        axes[0].add_patch(
            FancyBboxPatch(
                (0.05 + index * 0.235, 0.42),
                0.19,
                0.22,
                boxstyle="round,pad=0.02",
                facecolor=mpl.colors.to_rgba(BLUE, 0.12 + index * 0.08),
                edgecolor=BLUE,
            )
        )
        axes[0].text(0.145 + index * 0.235, 0.53, label, ha="center", va="center", fontsize=7.6)
        if index < 3:
            axes[0].add_patch(FancyArrowPatch((0.24 + index * 0.235, 0.53), (0.275 + index * 0.235, 0.53), arrowstyle="->", color=GRAY))
    axes[0].text(0.5, 0.78, "Cumulative clinical-information states", ha="center", weight="bold")
    axes[0].set_title("A", loc="left", weight="bold")
    axes[0].axis("off")

    axes[1].add_patch(FancyBboxPatch((0.13, 0.63), 0.30, 0.15, boxstyle="round,pad=0.02", fc="#DCEAF4", ec=BLUE))
    axes[1].add_patch(FancyBboxPatch((0.57, 0.63), 0.30, 0.15, boxstyle="round,pad=0.02", fc="#F8E2D5", ec=VERMILLION))
    axes[1].text(0.28, 0.705, "Clinical model", ha="center", va="center")
    axes[1].text(0.72, 0.705, "Clinical + ECG", ha="center", va="center")
    axes[1].text(0.5, 0.44, "Same eligible observations\nPaired proper-loss contrast", ha="center", va="center", weight="bold")
    axes[1].add_patch(FancyArrowPatch((0.28, 0.61), (0.44, 0.49), arrowstyle="->", color=GRAY))
    axes[1].add_patch(FancyArrowPatch((0.72, 0.61), (0.56, 0.49), arrowstyle="->", color=GRAY))
    axes[1].set_title("B", loc="left", weight="bold")
    axes[1].axis("off")

    labels = ["Residual value", "Implemented capture", "Saturation", "Failure evidence"]
    colors = [BLUE, VERMILLION, GREEN, GOLD]
    for index, (label, color) in enumerate(zip(labels, colors, strict=True)):
        y = 0.78 - index * 0.18
        axes[2].add_patch(FancyBboxPatch((0.14, y - 0.055), 0.72, 0.11, boxstyle="round,pad=0.02", fc=mpl.colors.to_rgba(color, 0.13), ec=color))
        axes[2].text(0.5, y, label, ha="center", va="center", weight="bold")
    axes[2].text(0.5, 0.09, "Stage-specific claims only", ha="center", color=GRAY)
    axes[2].set_title("C", loc="left", weight="bold")
    axes[2].axis("off")
    fig.suptitle("CMRV: an information-state audit of residual modality value", weight="bold", y=0.99)
    save_figure(fig, "figure_01_cmrv_framework")


def figure_2(frame: pd.DataFrame) -> None:
    panels = {
        "A": (["absorption_balanced", "absorption_rare"], "Information absorption"),
        "B": (["persistent_unique"], "Persistent unique information"),
        "C": (["ecg_null"], "ECG null"),
        "D": (["implementation_failure"], "Implementation failure"),
    }
    values = frame[frame["figure"] == "2"]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True)
    metric_style = {
        "oracle_piv_nats": (BLUE, "o", "Oracle PIV"),
        "implemented_ifv_nats": (VERMILLION, "s", "Implemented IFV"),
    }
    for axis, (panel, (scenarios, title)) in zip(axes.flat, panels.items(), strict=True):
        block = values[(values["panel"] == panel) & (values["scenario"].isin(scenarios))]
        for scenario_index, scenario in enumerate(scenarios):
            for metric, (color, marker, label) in metric_style.items():
                rows = block[(block["scenario"] == scenario) & (block["metric"] == metric)].sort_values("tier")
                if rows.empty:
                    continue
                linestyle = "-" if scenario_index == 0 else "--"
                scenario_label = scenario.replace("_", " ").title()
                axis.plot(range(4), rows["estimate"], marker=marker, color=color, linestyle=linestyle, label=f"{scenario_label}: {label}")
        axis.axhline(0, color="#BBBBBB", linewidth=0.8)
        axis.set_xticks(range(4), TIER_ORDER)
        axis.set_ylabel("Value (nats)")
        axis.set_title(f"{panel}  {title}", loc="left", weight="bold")
        axis.legend(frameon=False, fontsize=6.8)
    gate = subset(frame, "2", "gate_pass_count")
    if len(gate) != 1 or int(gate.iloc[0]["estimate"]) != 9 or gate.iloc[0]["status"] != "PASS":
        raise ValueError("known-truth gate summary must be 9/9 PASS")
    fig.suptitle("Known-truth recovery separates saturation, null value, and failure", weight="bold")
    fig.text(0.5, 0.01, "All nine prespecified recovery gates passed", ha="center", color=GREEN, weight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "figure_02_known_truth")


def figure_3(frame: pd.DataFrame) -> None:
    cells = subset(frame, "3", "implemented_ifv_normalized")
    summaries = subset(frame, "3", "implemented_saturation_normalized")
    task_titles = {"perioperative_ecg": "Perioperative ECG task", "cardiac_surgery": "Cardiac-surgery task"}
    stage_style = {"train": (BLUE, "o", -0.08), "cal": (GREEN, "s", 0.0), "lte": (VERMILLION, "^", 0.08)}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharey=True)
    for panel, (axis, task) in enumerate(zip(axes, task_titles, strict=True)):
        task_rows = cells[cells["task"] == task]
        for stage, (color, marker, offset) in stage_style.items():
            rows = task_rows[task_rows["stage"] == stage].set_index("tier").loc[list(TIER_ORDER)]
            x = np.arange(4) + offset
            y = rows["estimate"].to_numpy(float)
            low = rows["ci_lower"].to_numpy(float)
            high = rows["ci_upper"].to_numpy(float)
            axis.errorbar(x, y, yerr=np.vstack((y - low, high - y)), color=color, marker=marker, capsize=2.5, label=stage.upper())
            computed = information_saturation(dict(zip(TIER_ORDER, y, strict=True)))
            reported = summaries[(summaries["task"] == task) & (summaries["stage"] == stage)]
            if len(reported) != 1 or not np.isclose(computed, reported.iloc[0]["estimate"], atol=2e-12):
                raise ValueError(f"saturation mismatch for {task}/{stage}")
        axis.axhline(0, color="#BBBBBB", linewidth=0.8)
        axis.axvspan(-0.25, 1.25, color=BLUE, alpha=0.05)
        axis.axvspan(1.75, 3.25, color=GRAY, alpha=0.05)
        axis.set_xticks(range(4), TIER_ORDER)
        axis.set_title(f"{'AB'[panel]}  {task_titles[task]}", loc="left", weight="bold")
        axis.set_xlabel("Clinical-information tier")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Implemented ECG value / outcome entropy")
    fig.suptitle("Implemented ECG value concentrates in lower-information states", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, "figure_03_cross_stage_saturation")


def figure_4(frame: pd.DataFrame) -> None:
    values = frame[frame["figure"] == "4"]
    task_titles = {"perioperative_ecg": "Perioperative ECG task", "cardiac_surgery": "Cardiac-surgery task"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for panel, (axis, task) in enumerate(zip(axes, task_titles, strict=True)):
        task_rows = values[values["task"] == task]
        implemented = task_rows[task_rows["metric"] == "implemented_ifv_nats"].set_index("tier").loc[list(TIER_ORDER)]
        attainable = task_rows[task_rows["metric"] == "model_class_attainable_value_nats"].set_index("tier").loc[list(TIER_ORDER)]
        y = np.arange(4)
        for index in range(4):
            axis.plot([implemented.iloc[index]["estimate"], attainable.iloc[index]["estimate"]], [y[index], y[index]], color="#BBBBBB", linewidth=2)
        axis.scatter(implemented["estimate"], y, marker="s", color=VERMILLION, label="Implemented IFV", zorder=3)
        axis.scatter(attainable["estimate"], y, marker="D", color=BLUE, label="Estimated model-class contrast", zorder=3)
        axis.axvline(0, color="#999999", linewidth=0.8)
        axis.set_yticks(y, TIER_ORDER)
        axis.invert_yaxis()
        axis.set_xlabel("Signed train-OOF contrast (nats)")
        axis.set_title(f"{'AB'[panel]}  {task_titles[task]}", loc="left", weight="bold")
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Model-class attainable and implemented values can diverge", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, "figure_04_value_frontier")


def figure_5(frame: pd.DataFrame) -> None:
    cells = subset(frame, "5", "configuration_ifv_nats")
    gates = subset(frame, "5", "operating_gate")
    categories = [
        "engineering_and_local_validation_priority",
        "selective_or_local_validation",
        "no_positive_implemented_value",
    ]
    expected_counts = [10, 8, 6]
    actual_counts = [int((cells["category"] == category).sum()) for category in categories]
    if actual_counts != expected_counts:
        raise ValueError(f"configuration counts changed: {actual_counts}")
    category_code = {category: index for index, category in enumerate(categories)}
    cmap = ListedColormap([GREEN, GOLD, LIGHT_GRAY])
    task_titles = {"perioperative_ecg": "Perioperative ECG task", "cardiac_surgery": "Cardiac-surgery task"}
    stages = ["train", "cal", "lte"]
    fig = plt.figure(figsize=(9.4, 5.8))
    grid = fig.add_gridspec(2, 2, height_ratios=(3.2, 1.35), hspace=0.34, wspace=0.25)
    for panel, task in enumerate(task_titles):
        axis = fig.add_subplot(grid[0, panel])
        block = cells[cells["task"] == task]
        matrix = np.zeros((len(stages), len(TIER_ORDER)))
        estimates = np.zeros_like(matrix)
        for row_index, stage in enumerate(stages):
            for column_index, tier in enumerate(TIER_ORDER):
                row = block[(block["stage"] == stage) & (block["tier"] == tier)]
                if len(row) != 1:
                    raise ValueError(f"missing configuration cell: {task}/{stage}/{tier}")
                matrix[row_index, column_index] = category_code[row.iloc[0]["category"]]
                estimates[row_index, column_index] = row.iloc[0]["estimate"]
        axis.imshow(matrix, cmap=cmap, vmin=-0.5, vmax=2.5, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                category = categories[int(matrix[row_index, column_index])]
                code = {
                    "engineering_and_local_validation_priority": "P",
                    "selective_or_local_validation": "S",
                    "no_positive_implemented_value": "C",
                }[category]
                text_color = "white" if code in {"P", "C"} else "#332A00"
                axis.text(column_index, row_index - 0.12, code, ha="center", va="center", fontsize=10, weight="bold", color=text_color)
                axis.text(column_index, row_index + 0.16, f"{estimates[row_index, column_index]:+.3f}", ha="center", va="center", fontsize=7.2, color=text_color)
        axis.set_xticks(range(4), TIER_ORDER)
        axis.set_yticks(range(3))
        if panel == 0:
            axis.set_yticklabels(["Train", "Calibration", "Locked evaluation"])
        else:
            axis.set_yticklabels([])
            axis.tick_params(axis="y", length=0)
        axis.set_title(f"{'AB'[panel]}  {task_titles[task]}", loc="left", weight="bold")
    gate_axis = fig.add_subplot(grid[1, :])
    expected_gates = {
        "Clean signal": "REPLICATED",
        "G3 transport": "PASS",
        "G2 stress": "FAIL",
        "P19 localization": "PARTIAL",
    }
    for index, (label, expected_status) in enumerate(expected_gates.items()):
        row = gates[gates["category"] == label]
        if len(row) != 1 or row.iloc[0]["status"] != expected_status:
            raise ValueError(f"operating gate changed: {label}")
        color = GREEN if expected_status in {"PASS", "REPLICATED"} else RED if expected_status == "FAIL" else GOLD
        x = 0.035 + index * 0.242
        gate_axis.add_patch(FancyBboxPatch((x, 0.28), 0.205, 0.48, boxstyle="round,pad=0.02", fc=mpl.colors.to_rgba(color, 0.12), ec=color))
        gate_axis.text(x + 0.1025, 0.58, label, ha="center", va="center", weight="bold")
        gate_axis.text(x + 0.1025, 0.40, expected_status, ha="center", va="center", color=color, weight="bold")
    gate_axis.text(0.5, 0.06, "Task-level evidence only; no patient-level acquisition, treatment, workflow, or deployment claim", ha="center", color=GRAY, fontsize=8)
    gate_axis.set_title("C  Frozen operating envelope", loc="left", weight="bold")
    gate_axis.axis("off")
    fig.legend(
        handles=[
            Patch(facecolor=GREEN, label="P  Positive implemented value"),
            Patch(facecolor=GOLD, label="S  Selective/local validation"),
            Patch(facecolor=LIGHT_GRAY, label="C  Clinical-only favored in this implementation"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=7.5,
    )
    fig.suptitle("Descriptive configurations must be read with the operating boundary", weight="bold")
    save_figure(fig, "figure_05_configuration_map")


def main() -> None:
    configure_style()
    frame = load_data()
    figure_1()
    figure_2(frame)
    figure_3(frame)
    figure_4(frame)
    figure_5(frame)
    print(f"WROTE 5 PDF and 5 PNG figures under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
