import numpy as np
import pandas as pd
import anndata as ad
from itertools import permutations
from scipy.optimize import linear_sum_assignment

def match_labels(
        df: pd.DataFrame,
        k: int,
        from_col: str = 'niche_2D',
        to_col: str = 'niche_3D',
) -> dict:
    """
    Function to match labels between two clustering runs on the same data.

    Matches labels of DataFrame columns `from_col` to `to_col`
    by minimizing the number of mis-assigned cells.
    
    Returns dict with keys = to_col, values = from_col.
    Note: Dict values are of type int.
    """
    df[from_col] = df[from_col].astype(int)
    df[to_col] = df[to_col].astype(int)
    
    # If i is assigned to j, how many cells are mis-assigned?
    comb = np.zeros((k,k), dtype=int)
    for i, j  in permutations(np.arange(k), 2): # fill non-diagonal elements
        comb[i,j] = (df[from_col]==i).sum() - (df[ df[from_col]==i ][to_col]==j).sum()
    for i in range(k): # fill diagonal elements
        comb[i,i] = (df[from_col]==i).sum() - (df[ df[from_col]==i ][to_col]==i).sum()

    # Solve assignment problem: minimize the number of mis-assigned cells
    row_ind, col_ind = linear_sum_assignment(comb)
    
    return dict(zip(col_ind, row_ind))

def match_niches(
        subsets: list | np.ndarray,
        adata0: ad.AnnData,
        adata1: ad.AnnData,
        niche_col: str, # e.g., 'spatial_kmeans'
        k: int, # number of clusters
        subset_cols: list | np.ndarray = ['z','section'], # column names in obs of adata0, adata1 to subset by
        niche_names: list | np.ndarray = ['niche_2D', 'niche_3D'], # col names in return dict, with suffix '_mapped'
        id_col: str = 'id',
        phenotype_col: str = 'ct_broad',
        coord_loc: str = 'obs', # obs (with ['x','y']) or obsm (where columns 0:'x', 1:'y')
        sort: bool = True, # sort mapping by total number of cells per niche0, decreasing
) -> dict:
    """
    Maps clustering results between two datasets.
    """
    niche0 = niche_names[0]
    niche1 = niche_names[1]
    niche0_mapped = f'{niche0}_mapped'
    niche1_mapped = f'{niche1}_mapped'

    niche_mapping = {}
    for subset in subsets:
        # Subset data
        clusters0 = adata0[ adata0.obs[ subset_cols[0] ]==subset ].obs.set_index(id_col)[niche_col].copy()
        if coord_loc == 'obs':
            df = pd.DataFrame({
                niche0: clusters0.values.astype(int),
                niche1: adata1[ adata1.obs[ subset_cols[1] ]==subset ].obs.set_index(id_col).loc[ clusters0.index, niche_col ].values.astype(int),
                'phenotype': adata0[ adata0.obs[subset_cols[0]]==subset ].obs[ phenotype_col ].values,
                'x': adata0[ adata0.obs[subset_cols[0]]==subset ].obs['x'].values,
                'y': adata0[ adata0.obs[subset_cols[0]]==subset ].obs['y'].values })
        elif coord_loc == 'obsm':
            df = pd.DataFrame({
                niche0: clusters0.values.astype(int),
                niche1: adata1[ adata1.obs[ subset_cols[1] ]==subset ].obs.set_index(id_col).loc[ clusters0.index, niche_col ].values.astype(int),
                'phenotype': adata0[ adata0.obs[subset_cols[0]]==subset ].obs[ phenotype_col ].values,
                'x': adata0[ adata0.obs[subset_cols[0]]==subset ].obsm['spatial'][:,0],
                'y': adata0[ adata0.obs[subset_cols[0]]==subset ].obsm['spatial'][:,1] })
        else:
            raise NotImplementedError(f"coord_loc '{coord_loc}' not implemented.")
        df.index = clusters0.index # id_col

        # Map 2D niches to 3D niches
        mapping = match_labels(df, k=k, from_col=niche1, to_col=niche0)
        df[ niche0_mapped ] = df[ niche0 ].map( mapping )
        
        if sort: # Sort mapping by total number of cells per niche0, decreasing
            niches_by_ncells = dict(zip( df[ niche0_mapped ].value_counts().index.values, np.arange(k) ))
            df[ niche0_mapped ] = df[ niche0_mapped ].map( niches_by_ncells )
            df[ niche1_mapped ] = df[ niche1        ].map( niches_by_ncells )
        
        df['mismatch'] = df[ niche0_mapped ] != df[ niche1_mapped ]

        niche_mapping[ subset ] = df
    return niche_mapping