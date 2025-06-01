run_monocle <- function(
    expression_matrix,cell_metadata,gene_annotation,
    X_umap=NULL,clusters=NULL,
    color_label,
    save_dir, num_clusters=NULL
){
    library(monocle)
    cds <- newCellDataSet(
        as(expression_matrix, "sparseMatrix"),
        phenoData = new("AnnotatedDataFrame", data = cell_metadata),
        featureData  = new("AnnotatedDataFrame", data = gene_annotation),
        expressionFamily=negbinomial.size()
    )
    if ('Size_Factor' %in% colnames(cell_metadata)){
        sizeFactors(cds) <- cell_metadata$Size_Factor
    }else{
        cds <- estimateSizeFactors(cds)
    }
    cds <- estimateDispersions(cds)
    
    if ( is.null(X_umap) | is.null(clusters) ){
        disp_table <- dispersionTable(cds)
        unsup_clustering_genes <- subset(disp_table, mean_expression >= 0.1)
        cds <- setOrderingFilter(cds, unsup_clustering_genes$gene_id)
        cds <- reduceDimension(cds, max_components = 2, num_dim = 6, reduction_method = 'tSNE', verbose = T)
        cds <- clusterCells(cds, num_clusters = num_clusters)
        pData(cds)$Cluster <- factor(pData(cds)$Cluster)
    } else{
        reducedDimA(cds) <- t(X_umap)
        #SingleCellExperiment::reducedDims(cds)[['UMAP']] <- X_umap
        # cds@clusters[['UMAP']]$partitions <- factor(x = rep(1, length(rownames(colData(cds)))), levels = 1)
        # names(cds@clusters$UMAP$partitions) <- rownames(colData(cds))
        # cds@clusters[['UMAP']]$clusters <- clusters
        pData(cds)$Cluster <- factor(clusters)
    }
    
    URMM_clustering_DEG_genes <- differentialGeneTest(cds, fullModelFormulaStr = '~Cluster', cores = 16)
    ordering_genes <- row.names(URMM_clustering_DEG_genes)[order(URMM_clustering_DEG_genes$qval)][1:1000]
    
    cds <- setOrderingFilter(cds, ordering_genes)
    cds <- reduceDimension(cds, max_components = 2,method = 'DDRTree')
    cds <- orderCells(cds)
    
    f = plot_cell_trajectory(
        cds, 
        color_by = color_label,
    )

    ggplot2::ggsave(
        paste0(save_dir,'monocle_cluster.pdf'),
        plot = f,
        scale = 1, 
        limitsize = TRUE)
    
#     cds <- orderCells(cds, root_state=get_earliest_principal_node(cds))

    f = plot_cell_trajectory(
        cds, 
        color_by = "Pseudotime",
    )

    ggplot2::ggsave(
        paste0(save_dir,'monocle_time.pdf'),
        plot = f,
        scale = 1, 
        limitsize = TRUE)
    write.csv(pData(cds)[c('Cluster','Pseudotime','State')], paste0(save_dir,'monocle_res.csv'))
    cds
}

run_monocle3 <- function(
    expression_matrix,cell_metadata,gene_annotation,
    X_umap=NULL,clusters=NULL,
    color_label,
    save_dir
){
    library(monocle3)
    cds <- new_cell_data_set(
        as(expression_matrix, "sparseMatrix"),
        cell_metadata = cell_metadata,
        gene_metadata = gene_annotation)
    
    if ( is.null(X_umap) | is.null(clusters) ){
        cds <- preprocess_cds(cds, num_dim = 50)
        cds <- align_cds(
            cds, 
            alignment_group = "batch", 
            # residual_model_formula_str = "~ bg.300.loading + bg.400.loading + bg.500.1.loading + bg.500.2.loading + bg.r17.loading + bg.b01.loading + bg.b02.loading"
        )

        cds <- reduce_dimension(cds)
        cds <- cluster_cells(cds)
    } else{
        SingleCellExperiment::reducedDims(cds)[['UMAP']] <- X_umap
        cds@clusters[['UMAP']]$partitions <- factor(x = rep(1, length(rownames(colData(cds)))), levels = 1)
        names(cds@clusters$UMAP$partitions) <- rownames(colData(cds))
        cds@clusters[['UMAP']]$clusters <- clusters
    }
    
    cds <- learn_graph(cds)

    f = plot_cells(
        cds, 
        label_groups_by_cluster=FALSE,  
        color_cells_by = color_label,
        label_leaves=FALSE,
        label_branch_points=FALSE,
    )

    ggplot2::ggsave(
        paste0(save_dir,'monocle_cluster.pdf'),
        plot = f,
        scale = 1, 
        limitsize = TRUE)
    
    get_earliest_principal_node <- function(cds){
      cell_ids <- colnames(cds)
      closest_vertex <- cds@principal_graph_aux[["UMAP"]]$pr_graph_cell_proj_closest_vertex
      closest_vertex <- as.matrix(closest_vertex[colnames(cds), ])
      root_pr_nodes <- igraph::V(principal_graph(cds)[["UMAP"]])$name[as.numeric(names(which.min(table(closest_vertex[cell_ids,]))))]
      root_pr_nodes
    }
    
    cds <- order_cells(cds, root_pr_nodes=get_earliest_principal_node(cds))

    f = plot_cells(
        cds, 
        label_groups_by_cluster=FALSE,  
        color_cells_by = "pseudotime",
        label_leaves=FALSE,
        label_branch_points=FALSE,
    )

    ggplot2::ggsave(
        paste0(save_dir,'monocle_time.pdf'),
        plot = f,
        scale = 1, 
        limitsize = TRUE)
    cds
}