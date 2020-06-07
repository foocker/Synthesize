from faiss_demo.feature_detect import way_feature, get_way
from faiss_demo.config import train_image_dir, test_img_dir
from faiss_demo.utils import parser_name, iterate_files, static_data_vec, recongize_img

from faiss_demo.demo import test, double_test



def test_data_static():
    images_list = iterate_files(train_image_dir)
    way = get_way() 
    co = 0
    c = 0
    for file_name in images_list:
        c += 1
        cla_name = parser_name(file_name)
        ret, feature = way_feature(way, file_name)
        if not ret:
            co += 1
            print(cla_name, feature.shape)
        else:
            print(file_name, 'some thing wrong')
    print(co / c)




if __name__ == "__main__":
    # test_data_static()
    # test()
    double_test()
    # recongize_img('./wrong_results_merge.json')
    # mean, std, median, mode = static_data_vec(train_image_dir)
    # print(mean, std, median, mode)    # 89.0, 40.8,  85.0, 100
