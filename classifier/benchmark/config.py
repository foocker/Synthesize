save_path = '/*/Synthesize/weights_classify'
datainfo = 'xx'
datalabelmode = 'onelabel'    # 'onelabel', 'multilabel'
multi_threshold = 0.4
flagsfile = 'flags.txt'
split_data_ration = [0.2]
input_size = (640, 640)    # 448, 640 ? 640, 448


classes = ["*" ]

num_class = len(classes)
num_class_primary = num_class    # when start 0, equal num_class, when fintune_pri, substract **, 
device = 'cuda:0'


data_dir = '*/train'
data_dir_test = '*/train'
batch_size = 1
lr = 5e-4    # 1e-4
lr_momentum = 0.9
lr_gamma = 0.5
epochs = 16
epoch_plus = 8   # when start new is big, when reuse is small, ada, epoch can be small
lr_scheduler = False
milestones = [7, 8]

# for cnn_visual
# heatmap_path = '/*/Synthesize/temp_data/0_10'
heatmap_path = '*/pred_wrong/8_3'
# heatmap_path = '*/train/'
heatmap_class_name = '*'
heatmap_sample_num = 90



weight_path = ''

conf_matrix_path_json = '*/Synthesize/conf_matrix_path_{}.json'.format(datainfo)
temp_data_path = '/*/Synthesize/temp_data'
preds_results_json = '/*/Synthesize/preds_results_{}.json'.format(datainfo)
