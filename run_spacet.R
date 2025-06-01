library(SpaCET)

run_spacet <- function(
    visiumPath,cancerType, savedir='./',
    coreNo=8
){
    timestamp()
    
    SpaCET_obj <- create.SpaCET.object.10X(visiumPath = visiumPath)
    SpaCET_obj <- SpaCET.quality.control(SpaCET_obj)
    SpaCET_obj <- SpaCET.deconvolution(SpaCET_obj, cancerType=cancerType, coreNo=coreNo)
    
    SpaCET_obj <- SpaCET.CCI.colocalization(SpaCET_obj)
    SpaCET_obj <- SpaCET.CCI.LRNetworkScore(SpaCET_obj,coreNo=coreNo)
    SpaCET_obj <- SpaCET.identify.interface(SpaCET_obj)
    SpaCET_obj <- SpaCET.deconvolution.malignant(SpaCET_obj, coreNo = coreNo)
    
    write.csv(SpaCET_obj@results$deconvolution$propMat, file=paste0(savedir, 'probMat.csv'))
    write.csv(SpaCET_obj@results$CCI$colocalization, file=paste0(savedir, 'colocalization.csv'))
    write.csv(SpaCET_obj@results$CCI$LRNetworkScore, file=paste0(savedir, 'LRNetworkScore.csv'))
    saveRDS(SpaCET_obj, file=paste0(savedir, 'SpaCET.obj.rds'))
    
    timestamp()
}