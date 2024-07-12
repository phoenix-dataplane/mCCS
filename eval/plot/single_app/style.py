try:
    from IPython.display import set_matplotlib_formats
    set_matplotlib_formats('png', quality=220)
except ImportError:
    pass

from typing import Tuple, Dict, Any, Sequence

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['figure.dpi'] = 300

# sns.plotting_context('paper')
paper_rc = {
    'lines.linewidth': 3,
    'lines.markersize': 15,
    'figure.figsize': (6, 2.4)
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
print(matplotlib.rcParams['figure.figsize'])


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    r, g, b = rgb
    h = ''.join([hex(int(r * 256)), hex(int(g * 256)), hex(int(b * 256))])
    h = ''.join(h.split('0x')[1:])
    return '#' + h


my_color = {
    'blue': sns.color_palette().as_hex()[1],
    'orange': sns.color_palette("Set1").as_hex()[4],
    'green': sns.color_palette().as_hex()[3],
    'red': sns.color_palette().as_hex()[5],
}
# palette = ["#D4D443", "#2e86de", "#99FFCC", "#CC99FF", "#ee5253", "#660033"]
text_colors = [
    "blue", "orange", "red", "purple", "pink", "yellow", "cyan", "grey"
]

solutions = [
    "NCCL Bad Ring", "NCCL Good Ring", "mCCS ECMP", "mCCS"
]
palettes = [
    dict(zip(solutions, text_colors)),
    dict(
        zip(solutions,
            map(rgb_to_hex, sns.color_palette(n_colors=len(solutions))))),
    dict(
        zip(solutions, [
            '#e66101', '#fdb863', '#b2abd2', '#5e3c99', '#e66101', '#fdb863',
            '#b2abd2', '#5e3c99'
        ])),
    dict(
        zip(solutions, [
            '#66c2a5', '#e78ac3', '#fc8d62', '#8da0cb', '#e41a1c', '#984ea3',
            '#4daf4a', '#377eb8'
        ])),
    dict(
        zip(solutions, [
            '#d95f02', '#1b9e77', '#7570b3', '#e7298a', '#e66101', '#b2abd2',
            '#fdb863', '#5e3c99'
        ])),
    dict(
        zip(solutions, [
            '#e66101', '#b2abd2', my_color['green'], '#5e3c99', '#e66101',
            '#7fc97f', my_color['green'], '#5e3c99'
        ])),
]

# dashes = dict(zip(solutions, sns._core.unique_dashes(10)))
markers = dict(zip(solutions, ['o', 'o', 's', 'D', 's', 'd', 'o', 'o']))
dashes = dict(zip(solutions, ['-', '-', ':', ':', '-.', '-.', '-', '-']))


def create_alias(d: Dict[str, Any], aliases: Sequence[Tuple[str, str]]):
    for (a, b) in aliases:
        d[a] = d[b]


aliases = [
    ('4GPU_ECMP', 'mCCS ECMP'),
    ('4GPU_FLOW', 'mCCS'),
    ('8GPU_ECMP', 'mCCS ECMP'),
    ('8GPU_FLOW', 'mCCS'),
    ('Multi-Allreduce-ECMP', 'mCCS ECMP'),
    ('Multi-Allreduce-Flow', 'mCCS'),
    ('NCCL BR', 'NCCL Bad Ring'),
    ('NCCL GR', 'NCCL Good Ring'),
    ('NCCL', 'NCCL Bad Ring'),
    ('NCCL(OR)', 'NCCL Good Ring'),
    ('MCCS(-FA)', 'mCCS ECMP'),
    ('MCCS', 'mCCS'),
]
for p in palettes:
    create_alias(p, aliases)
create_alias(dashes, aliases)
create_alias(markers, aliases)

COLORS = sns.color_palette("muted")
palette = {
    "NCCL": COLORS[2],
    'NCCL(OR)': COLORS[0],
    'MCCS(-FA)': COLORS[-1],
    "MCCS": COLORS[3],
    "eRPC+Proxy": COLORS[4],
}
