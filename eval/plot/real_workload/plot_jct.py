import os 
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

workdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(os.path.join(workdir, "../"))
sys.path.append(rootdir)

paper_rc = {
    'lines.linewidth': 3,
    'lines.markersize': 15,
    'figure.figsize': (10, 3),
    #'figure.autolayout': False
}
sns.set(
    context='paper',
    font="Arial",
    font_scale=2.8,
    style="ticks",
    palette="muted",
    rc=paper_rc
)
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot():
    columns = [
        "Solution",
        "Job",
        "JCT"
    ]
    method_maps = {
        #"nccl-ecmp": "NCCL",
        "ecmp-fair": "ECMP",
        "fair": "FFA",
        "qosv1": "PFA",
        "qosv2": "PFA+TS",
    }
    method_order = ["ECMP", "FFA", "PFA", "PFA+TS"]
    workload_maps = {
        "vgg": "VGG (A)",
        "gpt_1": "GPT (B)",
        "gpt_2": "GPT (C)",
    }
    df = pd.read_csv(
        f'/{rootdir}/real_workload/jct.csv',
        header=None,
        names=columns,
        usecols=columns,
    )
    df = df[df["Solution"].isin(method_maps)]
    df["Solution"].replace(to_replace=method_maps, inplace=True)
    df["Job"].replace(to_replace=workload_maps, inplace=True)
    ffa_df = df[df['Solution'] == 'FFA']
    mean_ffa_jct = ffa_df.groupby('Job')['JCT'].mean().reset_index()
    mean_ffa_jct.rename(columns={'JCT': 'FFA_mean_JCT'}, inplace=True)
    df = pd.merge(df, mean_ffa_jct, on='Job', how='left')
    df["Normalized JCT"] = df["JCT"] / df["FFA_mean_JCT"]
    df = df[["Solution", "Job", "Normalized JCT"]]

    _, ax = plt.subplots()
    g = sns.barplot(
        data=df,
        x="Solution",
        y="Normalized JCT",
        errorbar=('pi', 95),
        hue="Job",
        order=method_order,
        capsize=0.3,
        errwidth=1.6,
        errcolor="black",
        edgecolor="black",
        linewidth=1,
    )
    g.legend(
        frameon=False,
        loc='upper right',
        bbox_to_anchor=(1.0, 1.08),
        prop={"size": 23},
        labelspacing=0.3,
        handletextpad=0.3,
        ncol=3,
        columnspacing=0.5,
    )
    ax.grid(axis='both', linestyle=':', linewidth=0.5)
    ax.set_xlabel("Solution")
    ax.set_ylim(0.0, 2.0)
    ax.set_ylabel("Norm. JCT")
    #pos = ax.axes.yaxis.label.get_position()
    #ax.axes.yaxis.label.set_position((pos[0], pos[1] - 0.10))
    sns.despine(fig=None, top=True, right=True, left=False, bottom=False)
    plt.tight_layout(pad=0.0)
    plt.savefig(f'/{rootdir}/figures/real_workload_jct.pdf')
    plt.clf()


if __name__ == "__main__":
    plot()