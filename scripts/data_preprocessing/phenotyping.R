## Imports
#renv::install("bioc::FlowSOM", lock=TRUE)
library(dplyr)
library(Biobase)
library(flowCore)
library(FlowSOM)

## Load data
# Required to convert csv to flowframe first
# Inspired from https://github.com/sydneycytometry/CSV-to-FCS/blob/master/CSV-to-FCS%20v2.0.R

# Read in csv file
infile <- "data/temp/Kuett_2022_reclustering_macrophages.csv"
infile <- "data/temp/Kuett_2022_reclustering_plasmacells.csv"
dat <- read.csv(infile)
head(dat)
# Separate intensity values from annotation data
ann <- dat[ , c('id','phenograph')]
expr <- dat[ , 3:dim(dat)[2]]
rownames(expr) <- ann$id
# Convert to flowframe
metadata <- data.frame(name=dimnames(expr)[[2]],
                       desc=paste('column',dimnames(expr)[[2]],'from dataset'))
metadata$minRange <- apply(expr,2,min)
metadata$maxRange <- apply(expr,2,max)
dat.ff <- new('flowFrame',
              exprs=as.matrix(expr),
              parameters=AnnotatedDataFrame(metadata))
# Convert to FlowSOM object
fsom <- ReadInput(dat.ff,
                  compensate=FALSE,
                  transform=FALSE,
                  scale=FALSE)
str(fsom) ## Inspect


## Analysis

set.seed(42)
# Build self-organizing map
fsom <- BuildSOM(fsom,
                 xdim=7,
                 ydim=7)
str(fsom$map) ## Inspect
# Build minimal spanning tree
fsom <- BuildMST(fsom)
str(fsom$MST) ## Inspect

# Visualize
#PlotFlowSOM(fsom, view="grid")
#PlotFlowSOM(fsom, view='MST') ## tree without annotation
#PlotStars(fsom) ## tree with annotation
PlotMarker(fsom, "CD68") ## Macrophages. Also: "cPARP.cCasp3", "CD44", "HER2..bis.", "panCK"
PlotMarker(fsom, "CD138") ## Plasma Cells
PlotMarker(fsom, "CD45") ## Plasma Cells
PlotNumbers(fsom) ## node indices

# Meta-clustering to help inform group assignments
metaClustering <- as.character(metaClustering_consensus(fsom$map$codes, k=8))
PlotPies(fsom,
         cellTypes=ann$phenograph, ## pie fill
         backgroundValues=metaClustering) ## pie background shade
PlotLabels(fsom, labels=metaClustering) ## cluster labels as nodes

## New cell assignments
# for macrophages
group1 <- c(2,10,9,1,18,17)
group2 <- c(64,56,63,55,47,48,39,40)
group3 <- c(32,31,23,22,24,15,16,7,8,14,6,13,5,12,4,11,3,19,21,20)
# for plasma cells
group1 <- c(47,48,49,41,42,35,34)

# Recode node labels (for plotting)
n_nodes <- max(fsom$map$mapping[,1])
new_node_labels <- rep('unassigned', n_nodes) ## default value: not assigned
new_node_labels[ 1:n_nodes %in% group1 ] <- "group_1"
new_node_labels[ 1:n_nodes %in% group2 ] <- "group_2"
new_node_labels[ 1:n_nodes %in% group3 ] <- "group_3"
# Recode cell labels
cell2node_orig <- fsom$map$mapping[,1]
cell2node_new <- rep('unassigned', length(cell2node_orig))
cell2node_new[ cell2node_orig %in% group1 ] <- "group_1"
cell2node_new[ cell2node_orig %in% group2 ] <- "group_2"
cell2node_new[ cell2node_orig %in% group3 ] <- "group_3"
table(cell2node_new) ## n cells per group

# Visualize new group assignments
PlotPies(fsom,
         cellTypes=ann$phenograph, ## pie fill
         backgroundValues=new_node_labels) ## pie background shade
PlotLabels(fsom, labels=new_node_labels) ## cluster labels as nodes

## Save new group assignments
ann$flowsom <- cell2node_new
outfile <- "data/temp/Kuett_2022_reclustering_macrophages_flowsom.csv"
outfile <- "data/temp/Kuett_2022_reclustering_plasmacells_flowsom.csv"
write.csv(ann, file=outfile, row.names=FALSE)
