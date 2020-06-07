from collections import defaultdict
import numpy as np
import logging
import faiss
import pickle
import os
from random import sample


from .config import *
from .utils import parser_name, iterate_files
from .feature_detect import way_feature, get_way


def registry_index(way_index):
    # assert way_index in range(len(DIMENSIONS))
    # prepare index
    dimensions = DIMENSIONS[way_index]
    if isAddPhash:
        dimensions += PHASH_X * PHASH_Y
    # https://github.com/facebookresearch/faiss/wiki/Binary-indexes
    # https://github.com/facebookresearch/faiss/blob/22b7876ef5540b85feee173aa3182a2f37dc98f6/tests/test_index_binary.py#L213
    if way_index != 3:
        # nbits/8 https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#relationship-with-lsh
        index = faiss.IndexBinaryHash(dimensions*8, 1)   
    else:
        index = faiss.index_factory(dimensions, INDEX_KEY)
    if USE_GPU:
        print("Use GPU...")
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # start training
    images_list = iterate_files(train_image_dir)    # may change
    # prepare ids
    ids_count = 0
    index_defaultdict = defaultdict(list)
    # ids = None
    # features = np.matrix([])
    features = []
    ids = []
    cla_name_temp = parser_name(images_list[0])
    way = get_way(w_index=way_index)    # ORB , surf, and so on
    for file_name in images_list:
        cla_name = parser_name(file_name)
        ret, feature = way_feature(way, file_name)
        numf = feature.shape[0]
        if way_index == 3 and FEATURE_CLIP:
            numf = FEATURE_CLIP if feature.shape[0] > FEATURE_CLIP else feature.shape[0]
            # feature = feature[:FEATURE_CLIP, :]
            choosed_fea = sample(range(feature.shape[0]), numf)
            feature = feature[choosed_fea, :]

        if ret == 0 and feature.any():
            if cla_name != cla_name_temp:
                ids_count += 1    # change when same img not only one 
                cla_name_temp = cla_name
            # record id and path
            # image_dict = {ids_count: (file_name, feature)}
            # image_dict = {ids_count: file_name}   # smaller than above
            index_defaultdict[ids_count].append(file_name)   # here in registry, on_id may have more than one img(obj)
            # print(way_feature.shape[0])
            # ids_list = np.linspace(ids_count, ids_count, num=feature.shape[0], dtype="int64")
            ids_list = np.linspace(ids_count, ids_count, num=numf, dtype="int64")
            print(feature.shape, ids_count, len(ids_list), ids_list.shape)
            features.append(feature)
            ids.append(ids_list)
    
            # if features.any():
            #     # print(feature[0].dtype)    # uint8
            #     features = np.vstack((features, feature))    # <class 'numpy.matrix'>
            #     # print(feature.shape)
            #     ids = np.hstack((ids, ids_list))    # None --> empty matrix
            #     print(ids.dtype, ids)
            # else:  # all feature is 0
            #     features = feature
            #     ids = ids_list

            # print(ids, ids.dtype)  # int64
            # print(index.is_trained)
            # print(features.shape, ids.shape)
            # if ids_count % 500 == 499:    # optim
            #     if not index.is_trained:
                    
            #         index.train(features)
                    
            #     index.add_with_ids(features, ids)    # https://github.com/facebookresearch/faiss/issues/856
            #     ids = None
            #     features = np.matrix([])
    # print(len(features), len(ids))
    features = np.vstack(features)
    ids = np.hstack(ids)
    print(features.shape, ids.shape)

    if features.any():
        if not index.is_trained:
            index.train(features)
        index.add_with_ids(features, ids)    # change

    # save index
    if WAY_INDEX == 3:
        faiss.write_index(index, index_path)
    else:
        faiss.write_index_binary(index, index_path)
    

    # save ids
    if not os.path.exists(ids_path):
        with open(ids_path, 'wb+') as f:
            try:
                pickle.dump(index_defaultdict, f, True)
            except EnvironmentError as e:
                logging.error('Failed to save index file error:[{}]'.format(e))
            except RuntimeError as v:
                logging.error('Failed to save index file error:[{}]'.format(v))
        
    print('Registry completed')

    # return index
