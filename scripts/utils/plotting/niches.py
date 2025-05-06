import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import logging
import sys
import io
from pysankey2 import Sankey  # see https://github.com/SZJShuffle/pySankey2/blob/master/example/pySankey2_demo.ipynb

def get_colors_phenotype() -> dict:
    colors = {
        'cancer cell': '#e41a1c',
        'other': '#377eb8',
        'immune cell': '#4daf4a',
        'stromal cell': '#984ea3',
        'myofibroblast': '#ff7f00',
        'endothelial cell': '#ffff33',
        'plasma cell': '#a65628',
        'epithelial cell': '#f781bf',
        'apoptotic cell': '#999999'
    }
    return colors

def get_colors_niche(k: int) -> dict:
    if k == 3:
        colors = {
            0: '#e41a1c',
            1: '#ff7f00',
            2: '#377eb8',
        }
    elif k == 4:
        colors = {
            0: '#e41a1c',
            1: '#ff7f00',
            2: '#377eb8',
            3: '#4daf4a'
        }
    elif k == 5:
        colors = {
            0: '#e41a1c',
            1: '#ff7f00',
            2: '#377eb8',
            3: '#4daf4a',
            4: '#984ea3',
        }
    else:
        raise NotImplementedError(f"Color scheme for {k} niches not implemented.")
    return colors



def _suppress_matplotlib_font_warning():
        """Suppress 'findfont' warnings from matplotlib."""
        logger = logging.getLogger("matplotlib")
        old_level = logger.level
        logger.setLevel(logging.ERROR)  # Suppress logs below ERROR level
        return old_level  # Return previous log level to restore later

def sankey_plot(
        df: pd.DataFrame,
        mapping_cols: list | np.ndarray = ['niche_2D_mapped', 'niche_3D_mapped'],
        colors_niche: dict | None = None,
        title: str = '',
        kwargs: dict = {}, # passed to Sankey.plot()
        savefig: str | None = None # path to save figure
) -> None:
    """
    Plot Sankey diagrams for niche mapping between 2D and 3D niches.
    """
    k = df[ mapping_cols[0] ].nunique()
    
    # Update plotting params
    kwargs_default = { 'figSize':(2.5,3.5), 'fontSize':12, 'fontPos':(.7,.5), 'text_kws':{'horizontalalignment':'left'} }
    [ kwargs.update({key:value}) for key,value in kwargs_default.items() if key not in kwargs.keys() ]
    if colors_niche is None:
        colors_niche = get_colors_niche(k)

    # Suppress 'findfont' warning from matplotlib
    # due to calling plt.rc('font', family='Arial') during sky.plot()
    old_log_level = _suppress_matplotlib_font_warning()
    stderr_backup = sys.stderr  # Save current stderr
    sys.stderr = io.StringIO()  # Redirect stderr to a dummy buffer

    # Create plot
    sky = Sankey(
        df[ mapping_cols ], 
        stripColor='left',
        colorDict =  colors_niche)
    # Sort niches numerically
    sky.__dict__['_layerLabels'] = OrderedDict([ (f'layer{i+1}', list(np.arange(k))) for i in range(2) ])

    # Plot
    try:
        fig, ax = sky.plot(
            **kwargs,
            )
        plt.title( title )
        
        if savefig is not None:
            plt.savefig(savefig, bbox_inches='tight')
        
        plt.show()
    
    # Reset global parameters
    finally:
        sys.stderr = stderr_backup  # Restore stderr
        logging.getLogger("matplotlib").setLevel(old_log_level)  # Restore log level
    plt.style.use('default')



def niche_pt_composition_subplot(
    id_to_niche: pd.Series, # with cell id as index
    id_to_phenotype: pd.Series, # with cell id as index
    ax: plt.Axes,
    sort_niches_by: str | list = 'niche_abundance',
    sort_phenotypes_by: str | list = 'phenotype_abundance',
    colors_phenotype: dict | None = None,
    show_ncells: bool = True,
    ncells_fraction: int = 1,
) -> None:
    """
    Plot the abundance of each niche per cell type.
    """
    k = id_to_niche.nunique()

    # Compute cluster composition
    id_to_niche.index.rename('id', inplace=True)
    id_to_phenotype.index.rename('id', inplace=True)
    data = pd.merge(id_to_niche, id_to_phenotype, on='id', how='inner')
    data.columns = ['niche', 'phenotype']
    comp = data.groupby(['niche', 'phenotype'], observed=False).size().unstack().fillna(0).T # n_cells per type and niche
    comp = comp / comp.sum() # abundance of type per niche w.r.t. total n_cells per niche
    
    # Niche order
    if isinstance(sort_niches_by, str):
        if sort_niches_by == 'niche_abundance': # by n_cells per niche, descreasing
            sort_niches_by = id_to_niche.value_counts().sort_values(ascending=False).index
            comp = comp[ sort_niches_by ]
        else:
            raise NotImplementedError(f"sort_niches_by '{sort_niches_by}' not implemented.")
    else:
        sort_niches_by = np.array(sort_niches_by)
        sort_niches_by = sort_niches_by[ np.isin( sort_niches_by, comp.columns ) ] # remove niches not present in data
        comp = comp[ sort_niches_by ]

    # Phenotype order
    if isinstance(sort_phenotypes_by, str):
        if sort_phenotypes_by == 'phenotype_abundance': # by n_cells per type (total), decreasing
            comp = comp.loc[ id_to_phenotype.value_counts().sort_values(ascending=False).index ]
        else:
            raise NotImplementedError(f"sort_phenotypes_by '{sort_phenotypes_by}' not implemented.")
    else:
        sort_phenotypes_by = np.array(sort_phenotypes_by)
        sort_phenotypes_by = sort_phenotypes_by[ np.isin( sort_phenotypes_by, comp.index ) ]
        comp = comp.loc[ sort_phenotypes_by ]

    # Plot
    if colors_phenotype is None:
        colors_phenotype = get_colors_phenotype()
    bottom = np.zeros(comp.columns.size)
    for pt, values in comp.iterrows():
        ax.bar(comp.columns, values, bottom=bottom, label=pt, 
               facecolor=colors_phenotype[pt], edgecolor=colors_phenotype[pt])
        bottom += values
    ax.set_ylim(0,1)
    ax.set_xticks(np.arange(k), sort_niches_by) # tick labels match niche order

    if show_ncells: # Add n_cells per niche
        n_cells_per_niche = id_to_niche.value_counts()
        n_cells_per_niche = n_cells_per_niche[ sort_niches_by ] # same order as above

        ax.set_xticks(np.arange(k), np.arange(k), size='large')
        ax.set_xlabel('Niche', size='large')
        for j, niche in enumerate(n_cells_per_niche.index):
            if ncells_fraction > 1:
                ax.text(j, 1.01, f'{n_cells_per_niche[niche] / ncells_fraction:.1f}', ha='center', va='bottom')
            else:
                ax.text(j, 1.01, f'{n_cells_per_niche[niche]}', ha='center', va='bottom')
    
    # Default plotting parameters
    sns.despine(ax=ax, top=True, right=True)
    ax.set_xticks(np.arange(k), sort_niches_by, size='large')
    ax.tick_params(axis='both', labelsize='large')
    ax.set_xlabel('Niche', size='large')