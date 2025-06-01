library(SoupX)
library(Seurat)
library(scater)
library(scDblFinder)
library(BiocParallel)

run_soupX <- function(data,data_tod,genes,cells,soupx_groups){
    
    # specify row and column names of data
    rownames(data) = genes
    colnames(data) = cells
    # ensure correct sparse format for table of counts and table of droplets
    data <- as(data, "sparseMatrix")
    data_tod <- as(data_tod, "sparseMatrix")

    # Generate SoupChannel Object for SoupX 
    sc = SoupChannel(data_tod, data, calcSoupProfile = FALSE)

    # Add extra meta data to the SoupChannel object
    soupProf = data.frame(row.names = rownames(data), est = rowSums(data)/sum(data), counts = rowSums(data))
    sc = setSoupProfile(sc, soupProf)
    # Set cluster information in SoupChannel
    sc = setClusters(sc, soupx_groups)

    # Estimate contamination fraction
    sc  = autoEstCont(sc, doPlot=FALSE, forceAccept=TRUE)
    # Infer corrected table of counts and rount to integer
    out = adjustCounts(sc, roundToInt = TRUE)
    out
}

run_scdblfinder <- function(data_mat){
    sce = scDblFinder(
        SingleCellExperiment(
            list(counts=data_mat),
        ) 
    )
    list(
        doublet_score = sce$scDblFinder.score,
        doublet_class = as.vector(sce$scDblFinder.class)
        )
}