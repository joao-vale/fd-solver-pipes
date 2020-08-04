import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def style_cmyk():
    # https://matplotlib.org/tutorials/introductory/customizing.html
    plt.rcParams["font.family"] = "Arial"
    # plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.autolimit_mode'] = 'data'  # 'round_numbers' to eliminate axis margin
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['figure.frameon'] = False
    plt.rcParams['figure.figsize'] = 18 / 2.54, 12 / 2.54
    mpl.rcParams['axes.prop_cycle'] = cycler(color=['#00A8E8', '#E62A88', '#FFBC1F', 'k', '#AD343E'])


def style_savefig():
    plt.rcParams['savefig.format'] = 'svg'  # also eps for vector
    # plt.rcParams['savefig.dpi'] = 300


def example_plot():
    style_cmyk()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    # ax1.set_title("This is my title")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.plot([0, 1], [0, 1], label='test1')
    ax1.plot([1, 2], [3, 4], label='test1')
    ax1.plot([0, 1], [1, 3], label='test1')
    ax1.plot([1, 2], [1, 4], label='test1')
    txt = 'Figure 1. I need the caption to be present a little below X-axis'
    plt.figtext(0.5, 0.03, txt, wrap=True, horizontalalignment='center', fontsize=10)
    plt.subplots_adjust(top=0.95, bottom=0.15)
    ax1.legend()
    plt.show()


def example_subplots():
    fig = plt.figure()
    for i, label in enumerate(('A', 'B', 'C', 'D')):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top')
    plt.show()
