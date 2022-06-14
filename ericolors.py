import seaborn as sns

def get_ericsson_colors():

    return ["#0068bf", "#cc2828", "#ff8c0a",
            "#fad22d", "#0c9b5b", "#8c60a8",
            "#b2d8f9", "#ffc1c1", "#fcf2bf",
            "#b7edd6", "#e8d6f2"]

def set_ericsson_style(classic=False, fontsize=16):

    sns.set_palette(get_ericsson_colors())

    sns.set_style("ticks")


    style = {"font.size":fontsize,"axes.titlesize":fontsize,"axes.labelsize":fontsize}
    style['font.family'] = 'Hilda 10'

    if classic:
        style['axes.linewidth'] = 1.2
    else:
        style['axes.linewidth'] = 0
        style['axes.facecolor'] = '#F2F2F2'
        style['figure.facecolor'] = '#F2F2F2'
        style['axes.grid'] = True

    style['text.color'] = "#4E4E4E"

    #print(style)
    sns.set_style(style)
   # sns.set_context("talk")

def set_ericsson_axis_properties(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis='both', pad=10)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10