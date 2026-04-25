# =============================================================================
#  visualization/plots.py — All plotting functions
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config.config import CONFIG

LABEL_COLORS = {
    'HEALTHY'    : '#2ecc71',
    'MILD_STRESS': '#f39c12',
    'STRESSED'   : '#e74c3c',
}


def plot_sample_locations(ts_store: dict,
                          feat_df: pd.DataFrame,
                          n_samples: int = 6,
                          save_path: str = None):
    """
    Plot raw vs smoothed time series for all 4 indices across
    sample locations, with rows color-coded by stress label.

    Args:
        ts_store   (dict)        : Output of smooth_all().
        feat_df    (pd.DataFrame): Must contain 'stress_label' column.
        n_samples  (int)         : Number of locations to plot.
        save_path  (str)         : Optional filepath to save the figure.
    """
    keys        = list(ts_store.keys())
    sample_keys = keys[:n_samples]

    fig, axes = plt.subplots(
        n_samples, 4,
        figsize=(20, 3.5 * n_samples),
        constrained_layout=True
    )
    if n_samples == 1:
        axes = [axes]

    fig.suptitle('Crop Stress Time Series — Sample Locations',
                 fontsize=16, fontweight='bold')

    for row, key in enumerate(sample_keys):
        ts    = ts_store[key]
        dates = ts['dates']

        row_feat = feat_df[
            (feat_df['lat'] == key[0]) & (feat_df['lon'] == key[1])
        ]
        label = row_feat['stress_label'].values[0] if len(row_feat) else 'UNKNOWN'
        color = LABEL_COLORS.get(label, 'gray')

        for col, idx_name in enumerate(CONFIG['indices']):
            ax  = axes[row][col]
            raw = ts[idx_name]
            smt = ts[f'{idx_name}_smooth']

            ax.plot(dates, raw, color='lightgray', lw=1, label='Raw')
            ax.plot(dates, smt, color=color,       lw=2, label='Smoothed')
            ax.axhline(0, color='black', lw=0.5, linestyle='--')
            ax.set_ylim(-1, 1)
            ax.grid(alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

            if col == 0:
                ax.set_ylabel(f'Label: {label}\n{idx_name}',
                              color=color, fontweight='bold')
            else:
                ax.set_title(idx_name, fontsize=9)

            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Time series plot saved → {save_path}")
    plt.show()


def plot_stress_distribution(feat_df: pd.DataFrame,
                             save_path: str = None):
    """
    Bar chart showing count and percentage of each stress label.

    Args:
        feat_df   (pd.DataFrame): Must contain 'stress_label' column.
        save_path (str)         : Optional filepath to save the figure.
    """
    order  = ['HEALTHY', 'MILD_STRESS', 'STRESSED']
    counts = feat_df['stress_label'].value_counts()
    counts = counts.reindex([o for o in order if o in counts.index])
    colors = [LABEL_COLORS[o] for o in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, counts.values):
        pct = val / len(feat_df) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_title('Crop Stress Label Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Locations')
    ax.set_xlabel('Stress Label')
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Distribution plot saved → {save_path}")
    plt.show()


def plot_feature_heatmap(feat_df: pd.DataFrame,
                         save_path: str = None):
    """
    Heatmap of mean feature values per stress label (normalized 0–1).
    Helps visually confirm that clusters are meaningfully separated.

    Args:
        feat_df   (pd.DataFrame): Must contain 'stress_label' column.
        save_path (str)         : Optional filepath to save the figure.
    """
    cols_to_plot = [
        'ndvi_mean', 'ndvi_std', 'ndvi_max_decline',
        'ndwi_mean', 'ndwi_stress_fraction',
        'evi_mean', 'savi_mean', 'dual_stress_fraction'
    ]
    agg    = feat_df.groupby('stress_label')[cols_to_plot].mean()
    scaled = (agg - agg.min()) / (agg.max() - agg.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(scaled.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(len(agg.columns)))
    ax.set_xticklabels(agg.columns, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(agg.index)))
    ax.set_yticklabels(agg.index, fontsize=11, fontweight='bold')
    ax.set_title('Feature Means per Stress Label (normalized 0–1)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Normalized Value')

    for i in range(len(agg.index)):
        for j in range(len(agg.columns)):
            ax.text(j, i, f'{agg.values[i, j]:.2f}',
                    ha='center', va='center', fontsize=8, color='black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Heatmap saved → {save_path}")
    plt.show()
