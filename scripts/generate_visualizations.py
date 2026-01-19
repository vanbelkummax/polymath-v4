#!/usr/bin/env python3
"""Generate publication-quality visualizations of Polymath knowledge structure."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (professional, colorblind-friendly)
COLORS = {
    'METHOD': '#2E86AB',      # Steel blue
    'PROBLEM': '#A23B72',     # Raspberry
    'DOMAIN': '#F18F01',      # Orange
    'ENTITY': '#C73E1D',      # Vermillion
    'DATASET': '#3B1F2B',     # Dark purple
    'primary': '#2E86AB',
    'secondary': '#F18F01',
    'accent': '#A23B72',
}

OUTPUT_DIR = Path('/home/user/polymath-v4/docs/images')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_concept_distribution():
    """Create a donut chart showing concept type distribution."""
    # Data from database query
    categories = ['DOMAIN', 'METHOD', 'ENTITY', 'PROBLEM', 'DATASET']
    counts = [3059519, 2020456, 1480370, 607368, 181736]
    colors = [COLORS[c] for c in categories]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create donut chart
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        colors=colors,
        autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
        pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
        startangle=90
    )

    # Style the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # Add center text
    ax.text(0, 0, '7.36M\nConcepts', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#333')

    # Create legend with counts
    legend_labels = [f'{cat}: {count:,}' for cat, count in zip(categories, counts)]
    ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5),
              frameon=True, fancybox=True, shadow=False)

    ax.set_title('Knowledge Base Concept Distribution', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'concept_distribution.png', facecolor='white')
    plt.close()
    print(f"✓ Saved concept_distribution.png")


def create_system_scale():
    """Create a horizontal bar chart showing system scale."""
    metrics = ['Code Chunks', 'Passages', 'Repositories', 'Documents', 'Paper-Code Links']
    values = [578830, 174321, 1881, 2193, 524]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Create gradient colors
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(metrics)))

    y_pos = np.arange(len(metrics))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel('Count')
    ax.set_title('Polymath v4 Knowledge Base Scale', fontsize=14, fontweight='bold')

    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        ax.text(width + max(values)*0.02, bar.get_y() + bar.get_height()/2,
                f'{val:,}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, max(values) * 1.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'system_scale.png', facecolor='white')
    plt.close()
    print(f"✓ Saved system_scale.png")


def create_knowledge_graph_schema():
    """Create a schema diagram of the knowledge graph structure."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Node positions
    nodes = {
        'Paper': (2, 6),
        'Passage': (6, 6),
        'Concept': (10, 6),
        'Repository': (2, 2),
        'CodeChunk': (6, 2),
    }

    # Draw edges first (behind nodes)
    edges = [
        ('Paper', 'Passage', 'CONTAINS'),
        ('Passage', 'Concept', 'MENTIONS'),
        ('Paper', 'Repository', 'LINKS_TO'),
        ('Repository', 'CodeChunk', 'CONTAINS'),
        ('CodeChunk', 'Concept', 'IMPLEMENTS'),
    ]

    for start, end, label in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]

        # Draw arrow
        ax.annotate('', xy=(x2-0.8, y2), xytext=(x1+0.8, y1),
                    arrowprops=dict(arrowstyle='->', color='#666', lw=2,
                                   connectionstyle='arc3,rad=0.1'))

        # Edge label
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 + 0.3
        ax.text(mid_x, mid_y, label, ha='center', va='center',
                fontsize=9, color='#666', style='italic',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none'))

    # Draw nodes
    node_colors = {
        'Paper': COLORS['primary'],
        'Passage': COLORS['secondary'],
        'Concept': COLORS['accent'],
        'Repository': '#28a745',
        'CodeChunk': '#17a2b8',
    }

    node_counts = {
        'Paper': '2,193',
        'Passage': '174,321',
        'Concept': '930K unique',
        'Repository': '1,881',
        'CodeChunk': '578,830',
    }

    for node, (x, y) in nodes.items():
        circle = plt.Circle((x, y), 0.7, color=node_colors[node], ec='white', lw=3, zorder=10)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=10,
                fontweight='bold', color='white', zorder=11)
        ax.text(x, y-1.1, node_counts[node], ha='center', va='center',
                fontsize=9, color='#333')

    # Title
    ax.text(6, 7.5, 'Knowledge Graph Schema', ha='center', va='center',
            fontsize=16, fontweight='bold')

    # Legend for edge types
    ax.text(0.5, 0.5, 'Edge Types:', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.1, '→ Structural (CONTAINS)\n→ Semantic (MENTIONS, IMPLEMENTS)\n→ Cross-modal (LINKS_TO)',
            fontsize=9, color='#666')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'knowledge_graph_schema.png', facecolor='white')
    plt.close()
    print(f"✓ Saved knowledge_graph_schema.png")


def create_concept_type_bars():
    """Create a bar chart of concept types with Neo4j counts."""
    # Neo4j counts (actual graph nodes)
    types = ['PROBLEM', 'ENTITY', 'METHOD', 'DATASET', 'DOMAIN', 'METRIC']
    counts = [342361, 284493, 231990, 36857, 34323, 149]
    colors = [COLORS.get(t, '#666') for t in types]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(types))
    bars = ax.bar(x, counts, color=colors, edgecolor='white', linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=0)
    ax.set_ylabel('Unique Concepts')
    ax.set_title('Concept Types in Knowledge Graph (Neo4j)', fontsize=14, fontweight='bold')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 5000,
                f'{height:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, max(counts) * 1.1)

    # Add total
    total = sum(counts)
    ax.text(0.98, 0.95, f'Total: {total:,} unique concepts', transform=ax.transAxes,
            ha='right', va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='none'))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'concept_types_neo4j.png', facecolor='white')
    plt.close()
    print(f"✓ Saved concept_types_neo4j.png")


if __name__ == '__main__':
    print("Generating publication-quality visualizations...")
    create_concept_distribution()
    create_system_scale()
    create_knowledge_graph_schema()
    create_concept_type_bars()
    print(f"\n✓ All visualizations saved to {OUTPUT_DIR}")
