from typing import OrderedDict
from .simulator import Config, EmuWorld, Message
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

colors = {
    "Decode": '#FFBE7A',
    Message.DRAFT: '#AFC8C6',
    Message.TARGET: '#F98070',
    Message.TURNON_AGGREGATOR: '#A1C6E7'
}
auxiliary_vline_color = "blue"
bar_height = 0.1

def visualize(config: Config, world: EmuWorld):


    records = OrderedDict({
        title: [[], []] for title in ["Decode", Message.DRAFT, Message.TARGET, Message.TURNON_AGGREGATOR]
    })
    for record in world.records:
        src_id = record.meta.get('src_id', record.meta.get('rank'))
        records[record.title][src_id].append((record.beg_time, record.duration))
    total_records = [sum([len(items[src_id]) if title != "Decode" else 1 for title, items in records.items()]) for src_id in (0, 1)]
    
    figure, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(world.wall_time, sum(total_records) / 10), gridspec_kw={'height_ratios': total_records}, dpi=100)
    for src_id in (0, 1):
        cum_height = 0
        for title, items in records.items():
            for beg_time, duration in items[src_id]:
                if title != "Decode":
                    cum_height -= bar_height
                axes[src_id].add_patch(patches.Rectangle(
                    xy=(beg_time, cum_height), width=duration, height=bar_height, 
                    edgecolor='black', facecolor=colors[title], linewidth=0.5, zorder=2, label=title))
        axes[src_id].set_yticks([])
        axes[src_id].set_ylim(cum_height, bar_height)
        axes[src_id].set_xticks(np.arange(0, world.wall_time, int(10 * (world.wall_time / 10)) / 10))
        axes[src_id].set_xlim(0, world.wall_time)
        axes[src_id].legend()
        handles, labels = axes[src_id].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[src_id].legend(by_label.values(), by_label.keys())
        axes[src_id].grid(zorder=1, alpha=0.5)
    auxiliary_vlines = list(set([patch.get_x() for patch in axes[0].patches] + [patch.get_x() for patch in axes[1].patches]))
    for ax in axes:
        for x in auxiliary_vlines:
            ax.vlines(x, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], color=auxiliary_vline_color, linestyle='-', lw=0.5, zorder=1)
    axes[1].set_xlabel('Time (s)')
    plt.suptitle(config.method)
    figure.tight_layout()
    return figure
