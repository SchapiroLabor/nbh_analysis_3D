import argparse
import anndata as ad
import cellcharter as cc
import time

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input",
                    action="store", type=str, required=True,
                    help="Path to .h5ad file.")
parser.add_argument("-o", "--output", dest="output",
                    action="store", type=str, required=True,
                    help="Output directory (will create folder of that name)")
parser.add_argument("--lowest_k", dest="lowest_k",
                    action="store", type=int, required=False, default=3,
                    help="Lowest number of clusters k")
parser.add_argument("--highest_k", dest="highest_k",
                    action="store", type=int, required=False, default=15,
                    help="Highest number of clusters k")
args = parser.parse_args()

## Load data
adata = ad.read_h5ad(args.input)

## Cluster Stability Analysis
# Initialize models
model_params = {
        'random_state': 42,
        'trainer_params': {
            'accelerator':'cpu',
            'enable_progress_bar': True
        },
    }
model = cc.tl.ClusterAutoK(n_clusters=(args.lowest_k, args.highest_k),
                              model_class=cc.tl.GaussianMixture,
                              model_params=model_params)

# Run stability analysis
t0 = time.process_time()
model.fit(adata)

## Print processing time
print( time.strftime('Process time: %m days, %H:%M:%S', time.gmtime(int(time.process_time()-t0))) )

## Save models to pickle (temporary)
model.save(args.output)