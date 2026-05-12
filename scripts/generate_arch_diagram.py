#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle, FancyArrowPatch
import os

# Create output directory
os.makedirs("docs", exist_ok=True)

# Initialize figure
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
ax.set_xlim(0, 120)
ax.set_ylim(0, 60)
ax.axis("off")

# Define boxes (x, y, width, height, label, color)
boxes = [
    (5, 40, 15, 12, "User", "#f0fdf4"),
    (25, 40, 15, 12, "Dashboard", "#fef9c3"),
    (45, 40, 15, 12, "FastAPI", "#fce7f3"),
    (65, 50, 15, 8, "Qdrant", "#dbeafe"),
    (65, 37, 15, 8, "BM25 Index", "#f5f5f4"),
    (65, 24, 15, 8, "LLM API", "#e9d5ff"),
    (85, 37, 18, 12, "RAGAS Evaluator", "#fef3c7"),
]

# Draw boxes
for x, y, w, h, label, color in boxes:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.3",
        facecolor=color,
        edgecolor="#374151",
        linewidth=2,
        zorder=1
    )
    ax.add_patch(box)
    # Add text
    ax.text(
        x + w/2, y + h/2, label,
        ha="center", va="center",
        fontsize=11, fontweight="semibold",
        zorder=2
    )

# Define arrows (from box index to box index)
arrows = [
    (0, 1),  # User -> Dashboard
    (1, 2),  # Dashboard -> FastAPI
    (2, 3),  # FastAPI -> Qdrant
    (2, 4),  # FastAPI -> BM25
    (2, 5),  # FastAPI -> LLM
    (3, 6),  # Qdrant -> RAGAS
    (4, 6),  # BM25 -> RAGAS
    (5, 6),  # LLM -> RAGAS
]

# Draw arrows
arrowstyle = ArrowStyle("->", head_width=0.8, head_length=1.0)
for i, j in arrows:
    x1, y1, w1, h1, _, _ = boxes[i]
    x2, y2, w2, h2, _, _ = boxes[j]
    
    # Calculate start and end points
    start_x = x1 + w1
    start_y = y1 + h1/2
    end_x = x2
    end_y = y2 + h2/2
    
    # Handle vertical arrows
    if i == 2 and j in [3, 4, 5]:
        start_x = x1 + w1/2
        start_y = y1
        end_x = x2 + w2/2
        end_y = y2 + h2
    if i in [3, 4, 5] and j == 6:
        start_x = x1 + w1
        start_y = y1 + h1/2
        end_x = x2
        end_y = y2 + h2/2
        
    arrow = FancyArrowPatch(
        (start_x, start_y),
        (end_x, end_y),
        arrowstyle=arrowstyle,
        color="#1f2937",
        linewidth=2,
        mutation_scale=20,
        zorder=3
    )
    ax.add_patch(arrow)

plt.title("DocuMind Architecture", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig("docs/architecture.png", bbox_inches="tight")
print("Architecture diagram saved to docs/architecture.png")
