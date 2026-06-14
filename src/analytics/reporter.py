"""
Session Reporter Module
=======================
Handles all end-of-session analytics: saving a CSV report
and rendering Matplotlib visualizations.

Supports multi-person tracking with per-person data and charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SessionReporter:
    """
    Collects per-person session data and generates analytics at the end.

    Tracks three possible states per person: 'focused', 'distracted', and 'spoof'.
    Spoof frames do NOT count toward the focus percentage calculation.

    Multi-person:
        Each log entry includes a person_name, so data is tracked per person.
        Charts and CSV output show all people separately.

    Usage:
        reporter = SessionReporter()
        reporter.log(t, "moaz", "focused", {"EAR": 27.5, ...})
        reporter.log(t, "mohamed", "distracted", {...})
        ...
        reporter.save(output_path)
        reporter.show_charts()
    """

    FEATURE_NAMES = ["EAR", "YAW", "PITCH", "ROLL", "GAZE"]

    def __init__(self):
        self._records: list = []  # All records (for CSV)
        self._per_person: dict = {}  # {name: {"labels": [], "times": []}}

    def _ensure_person(self, name: str) -> None:
        """Create tracking entry for a person if not yet seen."""
        if name not in self._per_person:
            self._per_person[name] = {"labels": [], "times": []}

    def log(self, timestamp: float, person_name: str, label: str, importances: dict) -> None:
        """
        Record one sample's result for a specific person.

        Args:
            timestamp:    Seconds since session start.
            person_name:  Name of the person (from face recognition).
            label:        'focused', 'distracted', or 'spoof'.
            importances:  Dict mapping feature name -> contribution %.
        """
        feat_values = [importances.get(n, 0.0) for n in self.FEATURE_NAMES]
        self._records.append([timestamp, person_name, label] + feat_values)

        self._ensure_person(person_name)
        self._per_person[person_name]["labels"].append(label)
        self._per_person[person_name]["times"].append(timestamp)

    @property
    def is_empty(self) -> bool:
        return len(self._records) == 0

    def average_focus_pct(self, person_name: str = None) -> float:
        """
        Focus percentage based on real frames only (excluding spoof).

        Args:
            person_name: If given, returns focus % for that person only.
                         If None, returns overall average across all people.
        """
        if person_name and person_name in self._per_person:
            labels = self._per_person[person_name]["labels"]
        else:
            labels = []
            for data in self._per_person.values():
                labels.extend(data["labels"])

        real_labels = [l for l in labels if l != "spoof"]
        if not real_labels:
            return 0.0
        focused_count = sum(1 for l in real_labels if l == "focused")
        return (focused_count / len(real_labels)) * 100

    def save(self, output_path: str) -> None:
        """Save the full session log to a CSV file."""
        if self.is_empty:
            return

        cols = ["time", "person", "state"] + self.FEATURE_NAMES
        df = pd.DataFrame(self._records, columns=cols)
        df["focus_pct"] = [100 if r[2] == "focused" else 0 for r in self._records]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[SessionReporter] Session data saved to '{output_path}'")

    def show_charts(self) -> None:
        """
        Display end-of-session charts for each tracked person:
          1. Per-person Focus % Over Time (line chart).
          2. Per-person Focused / Distracted / Spoof ratio (pie chart).
        """
        if self.is_empty:
            print("[SessionReporter] No data to plot.")
            return

        people = list(self._per_person.keys())

        if len(people) == 0:
            return

        # ── Chart 1: Focus % Over Time (all people on one chart) ────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        colors_list = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]

        for idx, person in enumerate(people):
            data = self._per_person[person]
            labels = data["labels"]
            times = data["times"]

            rolling = []
            real_count = 0
            focused_count = 0
            for label in labels:
                if label != "spoof":
                    real_count += 1
                    if label == "focused":
                        focused_count += 1
                pct = (focused_count / real_count * 100) if real_count > 0 else 0
                rolling.append(round(pct, 1))

            color = colors_list[idx % len(colors_list)]
            avg = self.average_focus_pct(person)
            ax.plot(times, rolling, color=color, linewidth=2,
                    label=f"{person.capitalize()} ({avg:.0f}%)")

            # Shade spoof periods
            for i, label in enumerate(labels):
                if label == "spoof" and i > 0:
                    ax.axvspan(times[i - 1], times[i], alpha=0.1, color="red")

        ax.axhline(50, color="gray", linestyle=":", linewidth=1)
        ax.axhspan(50, 100, alpha=0.03, color="green")
        ax.axhspan(0, 50, alpha=0.03, color="red")
        ax.set_title("Focus % Over Time (All Students)", fontsize=13)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Focus (%)")
        ax.set_ylim(0, 108)
        ax.set_yticks(range(0, 101, 10))
        ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.35)
        fig.tight_layout()
        plt.show()

        # ── Chart 2: Per-person Pie Charts ────────────────────────────────────
        n_people = len(people)
        fig2, axes = plt.subplots(1, n_people, figsize=(6 * n_people, 6))
        if n_people == 1:
            axes = [axes]  # Make iterable

        for ax2, person in zip(axes, people):
            labels = self._per_person[person]["labels"]
            n_focused = sum(1 for l in labels if l == "focused")
            n_distracted = sum(1 for l in labels if l == "distracted")
            n_spoof = sum(1 for l in labels if l == "spoof")

            sizes = []
            pie_labels = []
            pie_colors = []
            explode_vals = []

            if n_focused > 0:
                sizes.append(n_focused)
                pie_labels.append("Focused")
                pie_colors.append("#4CAF50")
                explode_vals.append(0.05)
            if n_distracted > 0:
                sizes.append(n_distracted)
                pie_labels.append("Distracted")
                pie_colors.append("#F44336")
                explode_vals.append(0.05)
            if n_spoof > 0:
                sizes.append(n_spoof)
                pie_labels.append("Spoof")
                pie_colors.append("#FF9800")
                explode_vals.append(0.1)

            if sizes:
                ax2.pie(
                    sizes,
                    labels=pie_labels,
                    colors=pie_colors,
                    explode=explode_vals,
                    autopct="%1.1f%%",
                    shadow=True,
                    startangle=140,
                    textprops={"fontsize": 12},
                )
            ax2.set_title(f"{person.capitalize()}", fontsize=14, fontweight="bold")

        fig2.suptitle("Session Summary", fontsize=16, fontweight="bold", y=1.02)
        fig2.tight_layout()
        plt.show()
