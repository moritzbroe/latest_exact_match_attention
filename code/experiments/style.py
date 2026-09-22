"""One look for every figure: font sizes for the paper's 5.5-inch text width, one colour
per architecture and one shade per LEMA head size.

    from experiments.style import COLOR, HEAD_COLOR, NAME, paper
    paper()                       # sets matplotlib's rcParams; call before plotting
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR = {"lema": "#2b6cb0", "rope": "#d95f02", "gdn": "#1b9e77"}
NAME = {"lema": "LEMA", "rope": "softmax", "gdn": "GDN", "sb": "stick-breaking"}
HEAD_COLOR = {8: "#9ecae1", 16: "#6baed6", 32: "#3182bd", 64: "#08519c"}   # LEMA d_h


def paper():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
