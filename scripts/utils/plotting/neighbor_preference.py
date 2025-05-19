import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
from pathlib import Path

def spatial_interaction_plot_mean(
        df: pd.DataFrame,
        method: str = 'scimap', # 'scimap' or 'COZI'
        figsize: tuple = (3.5, 3.5),
        title: str = None,
        kwargs: dict | None = None, # keywords to pass to sns.heatmap
        cbar_kwargs: dict | None = None, # keywords to pass to cbar_kws within sns.heatmap
        savefig: str | None = None # if not None, save figure to this path
):
    df = df.copy() # avoid modifying original data
    df.set_index(['phenotype', 'neighbour_phenotype'], inplace=True)
    
    # Filter for columns with score of interest
    if method == 'scimap':
        # Filter by column position, since no prefix is used
        counts = df.iloc[:, np.arange(0,df.shape[1],2)].copy()
    if method == 'COZI':
        counts = df.filter(like='zscore', axis=1).copy()
    
    # Take mean of scores per cell type pair
    counts.replace([np.inf, -np.inf], np.nan, inplace=True)
    counts = pd.DataFrame({ 'mean': np.nanmean(counts, 1) }, index=counts.index).unstack()
    counts.columns = counts.columns.droplevel(0)
    
    # Update plotting parameters, if required
    all_kwargs = {'cmap':'vlag', 'center':0, 'square':True, 'annot':True, 'fmt':".1f"}
    if method == 'scimap':
        all_kwargs.update({'vmin':-1, 'vmax':1})
    if kwargs is not None:
        all_kwargs.update(kwargs)    
    all_cbar_kwargs = {'shrink':.2, 'aspect':5, 'location':'left', 'ticklocation':'right'}
    if cbar_kwargs is not None:
        all_cbar_kwargs.update(cbar_kwargs)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(counts, ax=ax, 
                cbar_kws=all_cbar_kwargs, **all_kwargs)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis=u'both', which=u'both',length=0) # remove ticks
    ax.set_xlabel('neighbour phenotype')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    if title is not None:
        plt.title(title)
    
    if savefig is not None:
        # Check that path under savefig is valid
        assert Path( os.path.dirname(savefig) ).exists(), f'Path {savefig} does not exist'
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()


def plot_alongZ(
        score_3f: float,
        scores_3m: pd.DataFrame,
        scores_2: pd.DataFrame,
        interaction: tuple,
        method: str, # 'scimap' or 'COZI'
        sections_3D_min: np.ndarray = np.arange(10,150,10), # central 2D sections used to create 3D min data
        radius = 20, # NBH radius in um
        savefig: str | None = None # if not None, save figure to this filename
):
    # Get scores for each adata object
    if method=='COZI': # scores of interest have prefix 'zscore_'
        score_3f = score_3f.loc[interaction, 'zscore_imageid']
        scores_3m = scores_3m.filter(like='zscore', axis=1).loc[interaction, :]
        scores_3m.index = scores_3m.index.str.split('_').str[1].astype(int) # convert to int, without prefix
        scores_2 = scores_2.filter(like='zscore', axis=1).loc[interaction, :]
        scores_2.index = scores_2.index.str.split('_').str[1].astype(int) # convert to int, without prefix
    else: # scores of interest are in every other column
        score_3f = score_3f.loc[interaction, 'imageid']
        try:
            scores_3m = scores_3m.loc[interaction, sections_3D_min]
        except KeyError:
            scores_3m = scores_3m.loc[interaction, sections_3D_min.astype(str)]
            scores_3m.columns = sections_3D_min # convert to int
        try:
            scores_2 = scores_2.loc[interaction, np.arange(0,152)]
        except KeyError:
            scores_2 = scores_2.loc[interaction, np.arange(0,152).astype(str)]
            scores_2.columns = np.arange(0,152) # convert to int
    
    for a in [scores_3m, scores_2]:
        a[ np.isinf(a) ] = np.nan

    # Extend 3D min scores to region covered respective 2D sections
    radius_sections = radius // 2 # with section height of 2
    section_spacing = 10
    min_section = max(0, sections_3D_min.min()-radius_sections)
    max_section = min(152, sections_3D_min.max()+radius_sections)
    if radius_sections > (section_spacing/2): # in between central 2D sections, values of 3D min scores must be truncated
        scores_3m = np.concatenate([np.repeat(np.nan, min_section), # not included in 3D min data
                                    np.repeat(scores_3m.values[0], section_spacing - (radius_sections//2)), # left overhang of first section
                                    np.repeat(scores_3m.values, section_spacing), # values of 3D min scores
                                    np.repeat(scores_3m.values[-1], section_spacing - (radius_sections//2)), # right overhang of last section
                                    np.repeat(np.nan, 152-max_section) ]) # not included in 3D min data
    else:
        scores_3m = np.concatenate([np.repeat(np.nan, min_section), # not included in 3D min data
                                    np.repeat(scores_3m.values, 10), # values of 3D min scores
                                    np.repeat(np.nan, 152-max_section)]) # not included in 3D min data

    # Plotting params
    cmap = plt.get_cmap('vlag')
    cmap.set_bad('grey')
    vmax = max(score_3f, np.nanmax(scores_3m), np.nanmax(scores_2))
    vmin = min(score_3f, np.nanmin(scores_3m), np.nanmin(scores_2))
    kwargs = {'cmap':cmap, 'cbar':False, 'annot':False, 'vmax':vmax, 'vmin':vmin, 'center':0}
    cbar_kws = {'ticklocation':'left'}

    # Plot scores
    fig, ax = plt.subplots(3,1, figsize=(12,1), sharex=True)
    print(f'{method}, {interaction[0]} <> {interaction[1]}') # FIXME suptitle overlaps with first row subplot
    sns.heatmap(pd.DataFrame(np.repeat(score_3f, 152)[:, np.newaxis]).T, ax=ax[0],
                cbar_kws=cbar_kws, **kwargs )
    sns.heatmap(pd.DataFrame(scores_3m).T, ax=ax[1],
                mask=np.isnan(scores_3m)[np.newaxis, :],
                cbar_kws=cbar_kws, **kwargs )
    sns.heatmap(pd.DataFrame(scores_2).T, ax=ax[2], xticklabels=10,
                cbar_kws=cbar_kws, **kwargs )

    # Configure colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=(max(vmax, abs(vmin)))))
    sm.set_array([])
    cbar_ax, _ = mpl.colorbar.make_axes(ax, location='left', shrink=.7, aspect=5)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.outline.set_visible(False)

    # Configure axes
    ax[0].text(1.01, .5, '3D full', transform=ax[0].transAxes, rotation=0, va='center')
    ax[1].text(1.01, .5, '3D min.', transform=ax[1].transAxes, rotation=0, va='center')
    ax[2].text(1.01, .5, '2D', transform=ax[2].transAxes, rotation=0, va='center')
    ax[2].xaxis.tick_bottom() # visible ticks
    ax[2].set_xlabel('Section')
    for i in np.arange(3):    
        ax[i].set_yticks([],[])
        ax[i].set_ylabel(None)

    # Save
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()