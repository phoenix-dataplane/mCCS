import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


paper_rc = {
    'lines.linewidth': 3,
    'lines.markersize': 15,
    'figure.figsize': (8, 5),
    'figure.autolayout': False
}
sns.set(
    context='paper',
    font="Arial",
    font_scale=2.8,
    style="ticks",
    palette="muted",
    rc=paper_rc
)

def plot():
    columns = [
        "system",
        "message-size",
        "algorithm-bw"
    ]
    method_maps = {
        "nccl-shm": "NCCL (shm)",
        "nccl-shm-degrade": "NCCL (shm-degraded)",
        "nccl-net": "NCCL (rdma)",
        "mCCS": "mCCS (rdma+shm)",
    }
    df = pd.read_csv(
        'result.csv',
        header=None,
        names=columns,
        usecols=columns,
    )
    df = df[df["system"].isin(method_maps)]
    df["system"].replace(to_replace=method_maps, inplace=True)
    df.columns = [
        "System", "Message Size", "Algorithm Bandwidth", 
    ] 

    g = sns.lineplot(
        data=df,
        x="Message Size",
        y="Algorithm Bandwidth",
        hue="System",
        style="System",
        hue_order=["mCCS (rdma+shm)", "NCCL (shm)", "NCCL (shm-degraded)", "NCCL (rdma)"],
        markers=True,
        dashes=True,
    )
    g.set_xlabel("Message Size")
    g.set_ylabel("Algo BW (GB/s)")
    g.set_ylim(0, 15)
    labels = g.get_xticklabels()
    g.set_xticks(range(len(labels)))
    g.set_xlim(-0.3, len(labels) - 0.7)
    g.set_xticklabels(labels, rotation=45, ha='right')
    g.tick_params(axis='x', labelrotation=45, labelsize=21)

    g.legend(
        frameon=False,
        loc='lower right',
        #bbox_to_anchor=(0, 1.08),
        prop={"size": 16},
        ncol=1,
        columnspacing=0.5,
        labelspacing=0.3,
        handletextpad=0.3,
    )
    g.grid(axis='both', linestyle=':', linewidth=0.5)
    sns.despine(fig=None, top=True, right=True, left=False, bottom=False)
    plt.tight_layout(pad=0.0)
    plt.savefig("result.pdf")
    plt.clf()


if __name__ == "__main__":
    plot()