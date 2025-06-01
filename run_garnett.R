library(garnett)
library(org.Hs.eg.db)

garnett_setting_up <- function(mat,pdata,fdata,estimate_sf=T){
    pd <- new("AnnotatedDataFrame", data = pdata)
    fd <- new("AnnotatedDataFrame", data = fdata)
    cds <- newCellDataSet(mat,phenoData = pd,featureData = fd)
    if (estimate_sf){
        print('Estimate size factor...')
        cds <- estimateSizeFactors(cds)
    } else {
        print('assigning size factor using precompuated...')
        sizeFactors(cds) <- pdata$Size_Factor
    }
    cds
}

garnett_checking_marker <- function(
    cds, marker_file_path, saving_path){
    marker_check <- check_markers(
        cds, marker_file_path,
        db=org.Hs.eg.db,
        cds_gene_id_type = "SYMBOL",
        marker_file_gene_id_type = "SYMBOL"
    )
    f = plot_markers(marker_check)
    ggplot2::ggsave(
        paste0(saving_path,'/garnett_marker_check.pdf'), 
        plot = f,
        scale = 1, limitsize = TRUE
    )
    write.csv(
        file = paste0(saving_path,'/marker_check.csv'),
        marker_check
    )
    marker_check
}


run_garnett <- function(cds, marker_file_path, saving_path, num_unknown = 5000){
    classifier <- train_cell_classifier(
        cds = cds,
        marker_file = marker_file_path,
        db=org.Hs.eg.db,
        cds_gene_id_type = "SYMBOL",
        num_unknown = num_unknown,
        marker_file_gene_id_type = "SYMBOL"
    )

    save(
        file=paste0(saving_path,'/garnett_clf.Rds'),
        classifier
    )
    cds <- classify_cells(
        cds, classifier,
        db = org.Hs.eg.db,
        cluster_extend = TRUE,
        cds_gene_id_type = "SYMBOL"
    )
    labeling <- pData(cds)
    write.csv(
        file = paste0(saving_path,'/labeling.csv'),
        labeling
    )
    labeling
}