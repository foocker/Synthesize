import uuid
import pickle
from os import walk, rename
import numpy as np
from os.path import dirname, basename, join
from .feature_detect import get_way, way_feature
from .config import ids_path, IMAGESEARCH_TMP, WAY_INDEX, train_image_dir, test_img_dir


def iterate_files(_dir):
    result = []
    for root, _, files in walk(_dir, topdown=True):
        for fl in files:
            if fl.endswith("jpg") or fl.endswith("png"):    # JPG, PNG, JPEG, BMP
                result.append(join(root, fl))
    return result


def parser_name(fp):
    cla_name = basename(dirname(fp))
    # cla_index = cla_name.split('_')[-1]
    return cla_name


def read_array(input_, dimensions):
    array = np.fromfile(input_, dtype='>f4')
    return reshape_array(array, dimensions)


def reshape_array(array, dimensions):
    size = array.shape[0]
    cols = dimensions
    rows = size / dimensions
    array = array.reshape((rows, cols))
    return np.matrix(array)

def filter_feature(feature):
    # use locate, similariy to filter bad vec
    pass


def get_features(dir_p):
    features = []
    images_list = iterate_files(dir_p)
    way = get_way(w_index=WAY_INDEX)
    for file_name in images_list:
        ret, feature = way_feature(way, file_name)
        if ret == 0:
            features.append(feature)

    return np.vstack(features)

def static_data_vec(dir_p):
    blod_num = []
    obj_count = 0
    images_list = iterate_files(dir_p)
    way = get_way(w_index=WAY_INDEX)
    for file_name in images_list:
        ret, feature = way_feature(way, file_name)
        if ret == 0:
            blod_num.append(feature.shape[0])
            obj_count += 1
    mean, std = np.mean(blod_num), np.std(blod_num)
    median, mode = np.median(blod_num), np.argmax(np.bincount(blod_num))

    return mean, std, median, mode

def cluster_high_diemension(data, n_clusters=10):
    # for high dimensions
    from sklearn.cluster import KMeans
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data)
    label_pred = estimator.labels_
    centroids = estimator.cluster_centers_
    inertia = estimator.inertia_
    return label_pred, centroids, inertia

def cluster_one_dimension(vector):
    # https://stackoverflow.com/questions/11513484/1d-number-array-clustering
    pass



def strategy_config_param(static_vec):
    pass


def save_tmp_image(base64_str):
    img_data = base64_str.decode('base64')
    file_postfix = str(uuid.uuid1()) + ".jpg"
    filename = join(IMAGESEARCH_TMP, file_postfix)
    fw = open(filename, 'wb')
    fw.write(img_data)
    fw.close()
    return filename


def show_results(fn, results, save_wrong=True):
    # wrong_results = ''
    with open(ids_path, 'rb') as f:
        cig_ids_paths = pickle.load(f)
    print_pred = {}
    # print(results)
    for result in results:    # img mode just one id_ is [None]
        neighbors_scores = result['neighbors']
        for ns in neighbors_scores:
            # print(ns)
            pred_id = ns['id'] 
            pred_score = ns['score']
            first_registryfile = cig_ids_paths[pred_id][0]
            # print(first_registryfile, 'xx')
            pred_cla_name = parser_name(first_registryfile)
            print(pred_cla_name, 'xx2')
            print_pred[pred_cla_name] = pred_score

    # print(fn, 'pred result:', print_pred)
    real_cla = fn
    pred_bool = False
    for n in print_pred.keys():
        if n == fn[:len(n)]:
            real_cla = fn[:len(n)]
            pred_bool = True
            break

    pred_score = print_pred.get(real_cla) if pred_bool else 0
    
    # if not pred_bool:
    #     print(pred_bool, real_cla, 'pred score:', pred_score)
    #     wrong_results += '{}\n'.format(real_cla)
    # if pred_score != '100.0':
    #     print(pred_score)
    # if save_wrong:
    #     with open('./temp_wrong_results.json', 'a+') as f:
    #         f.writelines(wrong_results)
    return pred_bool


def recongize_img(json_f):
    from random import sample
    # only using on my own dir name format
    # move some pred wrong img from test to train for registry, improve performance(may) 
    with open(json_f, 'r+') as f:
        imgs_n = f.readlines()
    cla_flag = '2217'
    cla_count = 0
    cla_imgs = []
    for img_f in imgs_n:
        img = img_f.strip()
        num = img.split('_')[2]
        cla_imgs.append(img)
        cla_count += 1
        if num != cla_flag:
            
            if cla_count > 1:
                choosed_img = sample(cla_imgs, 1)[0]
                num_ = choosed_img.split('_')[2]
                inx = choosed_img.find(num_)
                cla_name = choosed_img[:inx] + num_ 
                # print(join(test_img_dir, cla_name, choosed_img), '\n', join(train_image_dir, cla_name, choosed_img))
                try:
                    rename(join(test_img_dir, cla_name, choosed_img), join(train_image_dir, cla_name, choosed_img))
                except:
                    # when run second should try
                    continue
            cla_flag = num
            cla_count = 1
            cla_imgs = [img]
        # print(cla_flag, cla_count, cla_imgs)
    print('Done')