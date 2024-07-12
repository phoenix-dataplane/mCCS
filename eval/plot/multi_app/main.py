#!/usr/bin/env python3
import sys
import os

workdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.abspath(os.path.join(workdir, "../"))
sys.path.append(rootdir)

from single_app import style
from plt_show import plt_show

# %%
import pandas as pd
import seaborn as sns
from seaborn.utils import desaturate
import matplotlib.pyplot as plt

def plot_bar(df, figname: str, hue_order, nccl_or, max_bw=15):
    if nccl_or:
        solution_map = {
            "Multi-Allreduce-ECMP": "MCCS\n(-FFA)",
            "Multi-Allreduce-Flow": "MCCS",
            "NCCL GR": "NCCL\n(OR)",
            "NCCL BR": "NCCL",
        }
    else:
        df_br = df[df["Solution"] == "NCCL GR"].copy()
        df_br["Solution"].replace(to_replace="NCCL GR", value="NCCL BR", inplace=True)
        df = pd.concat([df, df_br], ignore_index=True)
        solution_map = {
            "Multi-Allreduce-ECMP": "MCCS\n(-FFA)",
            "Multi-Allreduce-Flow": "MCCS",
            "NCCL GR": "NCCL\n(OR)",
            "NCCL BR": "NCCL",
        } 
    method_order = ["NCCL", "NCCL\n(OR)", 'MCCS\n(-FFA)', 'MCCS']

    app_map = {
        "blue": "A",
        "red": "B",
        "green": "C",
    }
    df["App"].replace(to_replace=app_map, inplace=True)
    df = df[df["Solution"].isin(solution_map)]
    df["Solution"].replace(to_replace=solution_map, inplace=True)
    df = df.groupby(['Solution', 'Trial ID', "App"])['BusBW (GB/s)'].mean().reset_index()
    pivot_data = df.pivot_table(index='Solution', columns='App', values='BusBW (GB/s)', aggfunc='mean')
    pivot_data = pivot_data.loc[method_order]
    plt.figure(figsize=(6, 4))
    # pivot_data.plot(kind='bar', stacked=True, color=['blue', 'green', 'red'], ax=plt.gca())

    apps = len(df["App"].unique())
    palette = sns.color_palette("muted", n_colors=apps)
    colors = [desaturate(palette[i], 0.75) for i in range(apps)]
    pivot_data.plot(
        kind='bar', 
        stacked=True,
        color=colors,
        ax=plt.gca(),
        edgecolor="black",
        linewidth=1,
    )

    agg_bw = df.groupby(['Solution', "Trial ID"])['BusBW (GB/s)'].sum().reset_index()
    agg_p025 = agg_bw.groupby(['Solution'])['BusBW (GB/s)'].quantile(q=0.025)
    agg_p975 = agg_bw.groupby(['Solution'])['BusBW (GB/s)'].quantile(q=0.975)
    agg_bw_mean = agg_bw.groupby(['Solution'])['BusBW (GB/s)'].mean()
    print('')
    # Adding error bars at only the top
    for i, solution in enumerate(pivot_data.index):
        # Calculate the cumulative sum for the y position of the error bars
        mean = agg_bw_mean.loc[solution]
        print('mean:', mean, pivot_data.loc[solution].sum())
        p025 = agg_p025.loc[solution]
        p975 = agg_p975.loc[solution]
        err = [[mean - p025], [p975 - mean]]
        x = i  # x position is the index of the solution
        y = mean # y position is the top of the stack
        (_, caps, _) = plt.gca().errorbar(x, y, yerr=err, fmt='none', color='black', capsize=6.0, elinewidth=1.2)
        for cap in caps:
            cap.set_markeredgewidth(1.0)


    # Calculate the standard error for each app
    # std_error = df.groupby(['Solution', 'App'])['BusBW (GB/s)'].std().unstack()

    # Display the calculated standard errors
    # std_error.head()

    # # Adding error bars at the top of each stack
    # for i, solution in enumerate(pivot_data.index):
    #     # Calculate the cumulative sum for the y position of the error bars
    #     cumulative = pivot_data.loc[solution].cumsum()
    #     for j, app in enumerate(pivot_data.columns):
    #         err = std_error.loc[solution, app]
    #         x = i  # x position is the index of the solution
    #         y = cumulative[j]  # y position is the top of the stack
    #         plt.gca().errorbar(x, y, yerr=err, fmt='none', color='black', capsize=5)

    # plt.title('Stacked Average Bus Bandwidth (GB/s) per App')
    # plt.xlabel('Solution')
    plt.xticks(rotation=0)
    plt.xlabel('')
    plt.ylim(0, 15)
    plt.ylabel('Bus BW (GB/s)')
    ncols = 2 if apps == 3 else 1
    plt.legend(
        frameon=False,
        loc='upper left',
        bbox_to_anchor=(0.0, 1.08),
        prop={"size": 23},
        labelspacing=0.2,
        handletextpad=0.3,
        ncol=2,
        columnspacing=0.5,
    )
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.grid(axis='both', linestyle=':', linewidth=0.5)
    sns.despine(fig=None, top=True, right=True, left=False, bottom=False)
    plt.savefig(f'{rootdir}/figures/{figname}', bbox_inches='tight')
    plt_show()
    plt.clf()

# %%
def main():
    df = pd.concat([pd.read_csv(os.path.join(workdir, "setting1.csv"))],
                   ignore_index=True)
    plot_bar(df, 'setting1_allreduce.pdf', hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"], nccl_or=False)

    df = pd.concat([pd.read_csv(os.path.join(workdir, "setting2.csv"))],
                   ignore_index=True)
    plot_bar(df, 'setting2_allreduce.pdf', hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"], nccl_or=True)

    df = pd.concat([pd.read_csv(os.path.join(workdir, "setting3.csv"))],
                   ignore_index=True)
    plot_bar(df, 'setting3_allreduce.pdf', hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"], nccl_or=True)

    df = pd.concat([pd.read_csv(os.path.join(workdir, "setting4.csv"))],
                   ignore_index=True)
    plot_bar(df, 'setting4_allreduce.pdf', hue_order=["NCCL BR", "NCCL GR", "mCCS ECMP", "mCCS"], nccl_or=False)

# %%
if __name__ == '__main__':
    main()
# %%