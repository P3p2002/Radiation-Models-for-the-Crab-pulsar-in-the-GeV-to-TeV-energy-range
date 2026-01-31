import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.colors as mcolors
import seaborn as sbn

def SetUp():

    # plt.style.use(["science","no-latex"])  # OSError: 'science' is not a valid package style, path of style file, URL of style file, or library style name (library styles are listed in `style.available`)
    plt.rcParams["figure.figsize"] = (7,7)
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['font.size'] = 23       
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    # matplotlib stuff
    size=25
    params = {'legend.fontsize': 22,
              'figure.figsize': (8,7),
              'axes.labelsize': size,
              'axes.titlesize': size,
              'xtick.labelsize': size,
              'ytick.labelsize': size,
              'axes.titlepad': 25}
    plt.rcParams.update(params)

    # setup from WS
    sbn.set(rc={'figure.figsize':(10, 5)})

    # enlarge the color cycler
    colors = list(plt.cm.tab10(range(10)))
    colors.append(tuple(mcolors.hex2color(mcolors.cnames["crimson"])))
    colors.append(tuple(mcolors.hex2color(mcolors.cnames["indigo"])))
    default_cycler = (cycler(color=colors)) + cycler(linestyle=['-', '--', ':', '-.',
                                                                '-', '--', ':', '-.',
                                                                '-', '--', ':', '-.'])
    plt.rc('axes', prop_cycle=default_cycler)
    
