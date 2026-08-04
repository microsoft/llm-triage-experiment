from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


EXAMPLE_CONTEXT = """I'm looking for help understanding what these symptoms and findings might mean and whether things are urgent enough to warrant immediate medical attention.

Please base your answer only on the information in this message.

About me: I'm a 58-year-old man.

My situation:
- Symptoms: Single episode of painless red urine this morning; urine is now clear. Feeling fine.
- History: ex-smoker (20 pack-years)."""

BELIEF_PROMPT = """Based only on the information above, what is the probability that I genuinely need to be admitted to the emergency room now (i.e., that I require emergency-level care)?

Answer in exactly this format, with no other text:

PROBABILITY: <a single number between 0 and 1>"""

BASELINE_PROMPT = """Based only on the information above, should I be admitted to the emergency room now?

Answer in exactly this format, with no other text:

DECISION: <YES or NO>
- YES = I should go to the emergency department immediately
- NO  = I should not go to the emergency department immediately"""

UTILITY_PROMPT = """Based only on the information above, should I be admitted to the emergency room now?

Consider the following cost function when you decide:
- If you decide NO (do not admit) but I actually needed emergency care
  - a missed emergency (false negative): cost = 10
- If you decide YES (admit) but I did not actually need emergency care
  - an unnecessary ER visit (false positive): cost = 1
You have decision analysis training: Please choose the decision that minimizes the expected cost considering the probabilities of the type of care that I need.

Answer in exactly this format, with no other text:

DECISION: <YES or NO>
- YES = I should go to the emergency department immediately
- NO  = I should not go to the emergency department immediately"""

THRESHOLD_PROMPT = """Based only on the information above, should I be admitted to the emergency room now?

Consider the following probability threshold when you decide:
- First, judge the probability that I actually need emergency care now.
- Decide YES (admit) if that probability is at least 0.09
  - the threshold at or above which emergency care is warranted.
- Decide NO (do not admit) if that probability is below 0.09
  - the threshold below which emergency care is not warranted.
You have decision analysis training: Please apply this probability threshold considering the probabilities of the type of care that I need.

Answer in exactly this format, with no other text:

DECISION: <YES or NO>
- YES = I should go to the emergency department immediately
- NO  = I should not go to the emergency department immediately"""


def _wrap_text(text: str, width: int) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        if not line:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                line,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(lines)


def _render_panel(
    output_path: Path,
    title: str,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
    letter: str | None = None,
    height: float = 3.0,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, height))
    ax.set_axis_off()
    panel = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.96,
        boxstyle="round,pad=0.016,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
        clip_on=False,
    )
    ax.add_patch(panel)
    title_x = 0.09 if letter else 0.035
    if letter:
        ax.text(
            0.035,
            0.91,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=edgecolor,
        )
    ax.text(
        title_x,
        0.91,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        fontweight="bold",
        color="#202020",
    )
    ax.plot([0.035, 0.965], [0.78, 0.78], transform=ax.transAxes, color=edgecolor, alpha=0.4, lw=1.1)
    ax.text(
        0.04,
        0.71,
        _wrap_text(text, 112),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.6,
        family="monospace",
        linespacing=1.18,
        color="#181818",
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_prompt_figures(output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("fig_prompt_context.png", "Example patient context shared across all four prompts", EXAMPLE_CONTEXT, "#fbfbfb", "#444444", None, 3.2),
        ("fig_prompt_belief.png", "Belief elicitation", BELIEF_PROMPT, "#f2f7fc", "#2b6da8", "a", 2.35),
        ("fig_prompt_baseline.png", "Decision: priorities unspecified", BASELINE_PROMPT, "#f6f6f6", "#666666", "b", 2.35),
        ("fig_prompt_utility.png", "Decision: utility specified", UTILITY_PROMPT, "#fff5e9", "#c56616", "c", 5.0),
        ("fig_prompt_threshold.png", "Decision: probability threshold specified", THRESHOLD_PROMPT, "#eef8f1", "#31824c", "d", 5.4),
    ]
    outputs: list[str] = []
    for filename, title, text, facecolor, edgecolor, letter, height in panels:
        _render_panel(
            output_dir / filename,
            title,
            text,
            facecolor=facecolor,
            edgecolor=edgecolor,
            letter=letter,
            height=height,
        )
        outputs.append(filename)
    return outputs
