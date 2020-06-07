from .registry_findex import registry_index
from .faiss_search import FaissIndex
from .config import index_path, TOP_N, test_img_dir, WAY_INDEX, train_image_dir
from .utils import show_results

import faiss

import os
from os.path import join

import time


def test():
    start = time.time()
    if not os.path.exists(index_path):
        print('begin registry')
        registry_index(WAY_INDEX)
        print('registry complete')
    # if WAY_INDEX == 3:
    #     index = faiss.read_index(index_path)
    # else:
    #     index = faiss.read_index_binary(index_path)
    # print('get index')
    # FI = FaissIndex(index, WAY_INDEX, False)

    # print('begin serach')
    # pred_bools = []
    # i = 0
    # for d, _, fs in os.walk(test_img_dir):
    #     for f in fs:
    #         results = FI.search_by_image(join(d, f), TOP_N)
    #         pred_bool = show_results(f, results)
    #         pred_bools.append(pred_bool)
    #     i += 1
    #     if i == 20000:
    #         break
    # print('acc is: ', sum(pred_bools) / len(pred_bools))
    # end = time.time()
    # avg_img = (end - start)/ i
    # print(avg_img, i)


def double_test():
    way_index1 = 2
    way_index2 = 3
    start = time.time()
    if not os.path.exists(index_path):
        print('begin registry')
        registry_index(way_index1)
        registry_index(way_index2)
        print('registry complete')
    index2 = faiss.read_index('/aidata/dataset/cigarette/fassi_lab/cig_index_surf_adddata')

    index1 = faiss.read_index_binary('/aidata/dataset/cigarette/fassi_lab/cig_index_orb_b1_num100_adddata')
    print('get index')
    FI1 = FaissIndex(index1, way_index1, False)
    FI2 = FaissIndex(index2, way_index2, False)

    print('begin serach')
    pred_bools = []
    i = 0
    rotated_gamma1 = 0
    rotated_gamma2 = 0
    rotated_gamma = 0
    wrong_results = ''
    for d, _, fs in os.walk(test_img_dir):
        
        for f in fs:
            if f.endswith('csv'):
                continue
            results1 = FI1.search_by_image(join(d, f), TOP_N)
            results2 = FI2.search_by_image(join(d, f), TOP_N)
            print(f)
            print(results2[0]['neighbors'],  results1[0]['neighbors'])
            pred_bool1 = show_results(f, results1, save_wrong=False)
            pred_bool2 = show_results(f, results2, save_wrong=False)
            if 'rotate_gamma' in f and not pred_bool1:
                rotated_gamma1 += 1
            if 'rotate_gamma' in f and not pred_bool2:
                rotated_gamma2 += 1
            
            pred_bool = pred_bool1 or pred_bool2
            pred_bools.append(pred_bool)
            if 'rotate_gamma' in f and not pred_bool:
                rotated_gamma += 1
            i += 1
            # if not pred_bool and 'rotate_gamma' in f:
            #     wrong_results += '{}\n'.format(f)
            if not pred_bool and 'rotate_gamma' not in f:
                wrong_results += '{}\n'.format(f)
        # i += 1    # dir 
        # if i == 20000:
        #     break
    
    # with open('./wrong_results_merge_adddata.json', 'a+') as f:
    #     f.writelines(wrong_results)
    
    print('acc is: ', sum(pred_bools) / len(pred_bools), len(pred_bools), rotated_gamma1, rotated_gamma2, rotated_gamma)
    end = time.time()
    avg_img = (end - start)/ i
    print(avg_img, i)
