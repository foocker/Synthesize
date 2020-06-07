DEBUG = True

def GET_FAISS_RESOURCES():
    return None


def GET_FAISS_INDEX():
    raise NotImplementedError


def GET_FAISS_ID_TO_VECTOR():
    raise NotImplementedError


UPDATE_FAISS_AFTER_SECONDS = None

IMAGESEARCH_TMP = "/tmp/search/"

# --------------------- Feature Detect
# resize size
NOR_X = 120
NOR_Y = 200

# phash size
PHASH_X = 8
PHASH_Y = 8

MODE = 'float'  # 'float' for L2, cosine feature, like sift, surft, 'binary' for orb etc...
# feature's count extracted from each image
NUM_FEATURES = 100   # this only for orb, we should do filte in brisk, akaze  and so on ways. 
FEATURE_CLIP = 100   # for surf
# 100:79/13, 40:32, 30:23, 20:15(detected feature vec), if every registry img has two with same class, can choose 20:15
isAddPhash = False

# get feature vector param
WAYS = ['AKAZE', 'BRISK', 'ORB', 'SURF']

AKAZE_D = 61
BRISK_D = 64
ORB_D = 32
SURF_D = 64
DIMENSIONS = [AKAZE_D, BRISK_D, ORB_D, SURF_D]

# https://docs.opencv.org/2.4/modules/features2d/doc/feature_detection_and_description.html#brisk
param_akaze = dict(nOctaves=1, nOctaveLayers=1)
param_brisk = dict(thresh=30, octaves=3, patternScale=1.2)
param_orb = dict(nfeatures=NUM_FEATURES, scaleFactor=1.5, nlevels=3,edgeThreshold=31,
                firstLevel=0, WTA_K=2, patchSize=31, fastThreshold=20)
param_surf = dict(hessianThreshold=500, nOctaves=3, nOctaveLayers=2)   # extended, 128
PARAMETERS = [param_akaze, param_brisk, param_orb, param_surf]

WAY_INDEX = 3    # correspond to ways, dimensions, parameters

# --------------------- Train
INDEX_KEY = "IDMap,Flat"
# INDEX_KEY = "IDMap,IMI2x10,Flat"
# INDEX_KEY = "IDMap,OPQ16_64,IMI2x12,PQ8+16"
USE_GPU = False    # slower?

train_image_dir = "/aidata/dataset/cigarette/etmoc_en_2/train"
# test_img_dir = "/aidata/dataset/cigarette/etmoc_en_2/test"
test_img_dir = "/aidata/dataset/cigarette/etmoc_en_2/image"
index_path = "/aidata/dataset/cigarette/fassi_lab/cig_index_surf_adddata"    # cig_index is the index binary file
# ids_vectors_path = '/faiss_service/resources/ids_paths_vectors'
ids_path = '/aidata/dataset/cigarette/cig_ids_paths.pkl'

# ---------------------  Search
TOP_SEARCH = 5
TOP_N = 5
# SIMILARITY = (NUM_FEATURES - 7) * 0.6 if WAY_INDEX != 3 else int((40.8)*0.6)   # std * 0.6
SIMILARITY = 3
