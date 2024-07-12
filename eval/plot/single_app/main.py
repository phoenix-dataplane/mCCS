#!/usr/bin/env python3
import sys
import os

workdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(os.path.join(workdir, "../"))
sys.path.append(rootdir)

from single_app import style
from plt_show import plt_show

# %%
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import numpy as np
from typing import Tuple, Optional
import math
import scipy.stats as st


# %%
def init():
    paper_rc = {
        'lines.linewidth': 3,
        'lines.markersize': 15,
        'figure.figsize': (6, 4),
        #'figure.autolayout': True,
    }
    # paper_rc = {'lines.linewidth': 3, 'lines.markersize': 15}
    sns.set(
        context='paper',
        # font="Arial", # this is the default font for sans-serif on my system
        # font="Helvetica Neue",
        font_scale=2.8,
        style="ticks",
        palette="Paired",
        rc=paper_rc)
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42


# %%
def plot_bandwidth(df, figname: str, hue_order):
    xlabel = "Data Size"
    y1label = "Algo BW (GB/s)"
    df.rename(columns={'Size (Bytes)':xlabel}, inplace=True)
    df.rename(columns={'AlgBW (GB/s)':y1label}, inplace=True)

    aliases = {
        'NCCL BR': 'NCCL',
        'NCCL GR': 'NCCL(OR)',
        '4GPU_ECMP': 'MCCS(-FA)',
        '4GPU_FLOW': 'MCCS',
        '8GPU_ECMP': 'MCCS(-FA)',
        '8GPU_FLOW': 'MCCS',
    }
    df['Solution'] = df['Solution'].map(lambda x: aliases[x])
    hue_order = ['NCCL', 'NCCL(OR)', 'MCCS(-FA)', 'MCCS']
    x = sorted(df[xlabel].unique().tolist())
    y1 = {}
    y_err = {}
    for sol in hue_order:
        data = df[df["Solution"] == sol].groupby([xlabel])
        y1[sol] = np.asarray(data[y1label].mean().tolist())

        y_err[sol] = []
        data = data[y1label].apply(list).reset_index(name=y1label)
        for i in x:
            y_raw = data[data[xlabel] == i][y1label].tolist()[0]
            err = st.t.interval(confidence=0.95,
                                df=len(y_raw) - 1,
                                loc=np.mean(y_raw),
                                scale=st.sem(y_raw))
            y_err[sol].append(np.mean(y_raw) - err[0])
        y_err[sol] = np.asarray(y_err[sol])
    df = df.loc[df['Solution'].isin(hue_order)]

    print(y1)

    # g = sns.lineplot(
    #     data=df,
    #
    #     hue='Solution',
    #     hue_order=hue_order,
    #     markers=style.markers,
    #     style='Solution',
    #     err_style='band',
    #     ci=95,
    #     palette=style.palette,
    #     dashes=style.dashes,
    #     fc='none'
    #     # legend='full',
    # )

    # fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    fig, ax1 = plt.subplots(1, 1, gridspec_kw={'width_ratios': [1]})

    lines1 = []
    for sol in hue_order:
        line, = ax1.plot(x,
                         y1[sol],
                         linewidth=3,
                         linestyle=style.dashes[sol],
                         color=style.palette[sol],
                         marker=style.markers[sol],
                         markersize=13,
                         markerfacecolor='white',
                         markeredgewidth=3)
        ax1.fill_between(x,
                         y1[sol] - y_err[sol],
                         y1[sol] + y_err[sol],
                         alpha=0.3,
                         color=style.palette[sol])
        lines1.append(line)
    ax1.set_ylabel(y1label)

    for ax in [ax1]:
        ax.set_xlim(16*1024, 1024*1024*1024)
        ax.set_xscale('log', base=2)
        xticks = list(map(lambda x: 1 << x, range(15, 30, 2)))
        ax.set_xticks(xticks)
        # ax.set_xticklabels(list(map(str, xticks)), rotation=30)
        xticklables = ["32KB", "128KB", "512KB", "2MB", "8MB", "32MB", "128MB", "512MB"]
        ax.set_xticklabels(xticklables, rotation=30)
        ax.set_xlabel(xlabel)
        ax.grid(axis='both', linestyle=':', linewidth=0.5)

    if figname == 'allgather_4gpu.pdf':
        plt.figlegend(
            lines1,
            hue_order,
            frameon=False,
            loc='lower right',
            bbox_to_anchor=(1.0, 0.27),
            prop={"size": 23},
            ncol=1,
            columnspacing=0.3,
            labelspacing=0.1,
            handletextpad=0.3,
        )
        ax1.set_ylim(0, 8)
        ax1.set_yticks(np.arange(0, 9, 2))
    elif figname == 'allgather_8gpu.pdf':
        plt.figlegend(
            lines1,
            hue_order,
            frameon=False,
            loc='upper left',
            bbox_to_anchor=(0.11, 1.05),
            prop={"size": 23},
            ncol=1,
            columnspacing=0.3,
            labelspacing=0.1,
            handletextpad=0.3,
        )
        ax1.set_ylim(0, 12)
        ax1.set_yticks(np.arange(0, 13, 3))
    elif figname == 'allreduce_4gpu.pdf':
        ax1.set_ylim(0, 4)
        ax1.set_yticks(np.arange(0, 5, 1))
    elif figname == 'allreduce_8gpu.pdf':
        ax1.set_ylim(0, 8)
        ax1.set_yticks(np.arange(0, 9, 2))


    sns.despine(fig=None, top=True, right=True, left=False, bottom=False)
    # for i in plt.gca().spines.values():
    #     i.set_linewidth(0.2)
    #     i.set_color('gray')

    plt.tight_layout(pad=0.0)
    plt.savefig(f'{rootdir}/figures/{figname}')
    plt_show()
    plt.clf()


# %%
def main():
    init()
    df = pd.concat([pd.read_csv(os.path.join(workdir, "allgather_4gpu.csv"))],
                   ignore_index=True)
    plot_bandwidth(df,
                   'allgather_4gpu.pdf',
                   hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"])

    # # %%
    df = pd.concat([pd.read_csv(os.path.join(workdir, "allgather_8gpu.csv"))],
                   ignore_index=True)
    plot_bandwidth(df,
                   'allgather_8gpu.pdf',
                   hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"])

    # # %%
    df = pd.concat([pd.read_csv(os.path.join(workdir, "allreduce_4gpu.csv"))],
                   ignore_index=True)
    plot_bandwidth(df,
                   'allreduce_4gpu.pdf',
                   hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"])

    # # %%
    df = pd.concat([pd.read_csv(os.path.join(workdir, "allreduce_8gpu.csv"))],
                   ignore_index=True)
    plot_bandwidth(df,
                   'allreduce_8gpu.pdf',
                   hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"])


# %%
if __name__ == '__main__':
    main()
# %%