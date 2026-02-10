from .pca_methods import PCA, PCAfull, PCAtest, SpatialPCA, SpatialTempPCA, ScikitPCA, CumulantPCA, PCAnorm, PCA_time_norm
from .temporal_pca_methods import TempPCA
from .ivac_methods import TICA, IVAC, TILDA, CumulantIVAC
from .LDAMethod import LDA
from .distinct_methods import DistinctPCA
from .Spatial_IVAC import SpatialIVAC, SpatialIVACnorm
__all__ = [
    "PCA", 
    "TempPCA", 
    "IVAC", 
    "PCAfull", 
    "PCAtest", 
    "LDA", 
    "TICA", 
    "SpatialPCA", 
    "TILDA", 
    "ScikitPCA",
    "CumulantPCA",
    "CumulantIVAC",
    "DistinctPCA",
    "PCAnorm",
    "SpatialIVAC",
    "SpatialIVACnorm",
    "PCA_time_norm",
]