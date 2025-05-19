import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
import glob

def load_data_2D(
        path_centroids: str = "../../data/Kuett_2022/MainHer2BreastCancerModel/measured_mask_centroids_2D/measured_mask_centroids_2D_*.csv",
        path_phenotype: str = '../../data/phenotyping_final_labels.csv',
        centroids_to: str = 'obs'
) -> ad.AnnData:
    """
    Load AnnData object with 2D data.

    Returns
    -------
    ad.AnnData
        contained in obs: cell id ('id'), phenotype ('ct_broad' or 'phenograph'), imageID ('z'),
        centroid position in μm ('x','y') are stored in obs or obsm['spatial], depending on 'centroids_to'
    """
    # Phenotypes
    assert Path(path_phenotype).exists(), 'Path to phenotypes does not exist'
    phenotypes = pd.read_csv(path_phenotype, dtype={'id':int, 'phenograph':'category', 'ct_broad':'category'})

    # Centroids
    files = np.array(glob.glob(path_centroids)) # unsorted list of files (full path)
    assert files.size > 0, 'No files with centroids found'
    dtype_dict = {'id':int, 'x':float, 'y':float, 'z':int}
    centroids = pd.concat([pd.read_csv(file, dtype=dtype_dict) for file in files])
    centroids.sort_values(by='z', inplace=True)
    centroids['z'] = centroids['z'].astype('category')
    
    data = pd.merge(centroids, phenotypes, on='id') # id = cell id, z = imageid
    # TODO convert index to str or suppress warning
    if centroids_to == 'obs':
        adata = ad.AnnData(obs=data)
    elif centroids_to == 'obsm':
        adata = ad.AnnData(
            obs=data.drop(columns=['x','y']),
            obsm={'spatial': data[['x','y']].values})
    else:
        raise NotImplementedError('centroids_to must be "obs" or "obsm"')

    return adata

def load_data_3D_full(
        path_centroids: str = "../../data/Kuett_2022/MainHer2BreastCancerModel/measured_mask_centroids_3D.csv",
        path_phenotype: str = '../../data/phenotyping_final_labels.csv',
        centroids_to: str = 'obs'
) -> ad.AnnData:
    """
    Load AnnData object with 3D data
    
    Returns
    -------
    ad.AnnData
        contained in obs: cell id ('id'), phenotype ('ct_broad' or 'phenograph'), imageid placeholder ('imageid'),
        centroid position in μm ('x','y','z') are stored in obs or obsm['spatial], depending on 'centroids_to'
    """
    # Phenotypes
    assert Path(path_phenotype).exists(), 'Path to phenotypes does not exist'
    phenotypes = pd.read_csv(path_phenotype, dtype={'id':int, 'phenograph':'category', 'ct_broad':'category'})

    # Centroids
    assert Path(path_centroids).exists(), 'Path to centroids does not exist'
    centroids = pd.read_csv(path_centroids)

    data = pd.merge(centroids, phenotypes, on='id') # id = cell id
    data['imageid'] = np.repeat('imageid', data.shape[0]) # placeholder
    # TODO convert index to str or suppress warning
    if centroids_to == 'obs':
        adata = ad.AnnData(obs=data)
    elif centroids_to == 'obsm':
        adata = ad.AnnData(
            obs=data.drop(columns=['x','y','z']),
            obsm={'spatial': data[['x','y','z']].values})
    else:
        raise NotImplementedError('centroids_to must be "obs" or "obsm"')

    return adata

def load_data_3D_min(
        path_centroids: str | None = None,
        path_phenotype: str = '../../data/phenotyping_final_labels.csv',
        radius: int = 20, # in um; ignored if path_centroids is given
        sections: np.ndarray = np.arange(10,152,10),
) -> ad.AnnData:
    """
    Load AnnData object with minimal 3D data: centroids and phenotypes.
    Use param 'sections' to specify which sections to load:
    e.g., sections = [20] will load the minimal 3D image that contains all
    3D neighbors of cells in the 2D section with section ID ('z') = 20.

    Returns
    -------
    ad.AnnData
        contained in obs: cell id ('id'), phenotype ('ct_broad' or 'phenograph'), imageid ('section'),
        centroid position in μm ('x','y','z') are stored in obs or obsm['spatial], depending on 'centroids_to'
    """
    assert Path(path_phenotype).exists(), 'Path to phenotypes does not exist'
    if path_centroids is None:
        path_centroids = f"../../data/temp/measured_mask_centroids/stacks/radius{radius}um/centroids_stack_*.csv"
    assert np.array(glob.glob(path_centroids)).size > 0, 'No files with centroids found'

    # Phenotypes
    phenotypes = pd.read_csv(path_phenotype, dtype={'id':int, 'phenograph':'category', 'ct_broad':'category'})

    # Centroids
    fn = path_centroids.replace('*', str(sections[0]))
    centroids = pd.read_csv(fn).rename(columns={'# z':'z'})
    centroids['section'] = np.repeat(sections[0], centroids.shape[0])
    for section in sections[1:]:
        fn = path_centroids.replace('*', str(section))
        centroids_section = pd.read_csv(fn).rename(columns={'# z':'z'})
        centroids_section['section'] = np.repeat(section, centroids_section.shape[0])
        centroids = pd.concat([centroids, centroids_section], ignore_index=True)
    centroids['section'] = centroids['section'].astype('category')
    centroids['id'] = centroids['id'].astype(int)  
    
    data = pd.merge(centroids, phenotypes, on='id') # id = cell id, section = imageid

    return ad.AnnData(obs=data) # TODO convert index to str or suppress warning