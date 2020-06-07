import os
import cv2
import torch
from PIL import Image
from .model import create_model
from matplotlib import pyplot as plt
from .config import num_class, weight_path, classes
from data.classifier_data import data_train_val_trainsform


def get_img_tensor(file_path):
    img = Image.open(file_path).convert('RGB')
    img_tensor = data_train_val_trainsform(img)
    img_tensor = img_tensor.unsqueeze(0)
    # print(img_tensor.shape)
    return img_tensor


def ternsor_to_array(img_tensor):
    img_tensor = img_tensor.cpu().squeeze(0).permute(1, 2, 0)
    img_array = img_tensor.numpy() * 255
    return img_array


def get_model():
    model = create_model(num_class)
    state_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in state_dict.keys():
        print("get_model:load from model_state_dict!!!")
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        print("not load from model_state_dict!!!")
        model.load_state_dict(state_dict)
    model.eval()
    return model


model = get_model()
model = model.to('cuda')
# if not isinstance(model, torch.cuda.FloatTensor):
#     model = model.to('cuda')

# dataset_class_to_idx = {'Cover_no':0 ,'Expose_garbage': 1, 'Overhead_lines': 2, 'Pile_stock': 3, 'Small_Ad': 4, 'You_Shang':5}
# dataset_class_to_idx = {"Cigrarette_butts": 0, "Fruit_flesh": 1, "Fruit_peel": 2, "Napkin": 3}
# dataset_idx_to_class = {v: k for (k, v) in dataset_class_to_idx.items()}
dataset_class_to_idx = {v: k for k, v in enumerate(classes)}
dataset_idx_to_class = {v: k for (k, v) in dataset_class_to_idx.items()}


def inference(file_path, threshold=0.65, show=False):
    img_tensor = get_img_tensor(file_path)
    img_tensor = img_tensor.to('cuda')
    img_array = ternsor_to_array(img_tensor)
    h, w, _ = img_array.shape

    with torch.no_grad():
        logits = model(img_tensor)
        probality = torch.nn.functional.softmax(logits, dim=1)
        print("probality:", probality.max())
        # probality[torch.where(probality<threshold)] = 0
        if probality.max() < threshold:
            data_class = 'others'
            print(probality)
        else:
            pred = probality.argmax(1).cpu().item()
            # print(pred)
            data_class = dataset_idx_to_class[pred]
    if show:
        probality_txt = '{:.3f}'.format(probality.cpu().max().item())
        img_array = cv2.putText(img_array, data_class + '|:' + probality_txt,
        (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), thickness=3)
        print(h, w)
        cv2.imwrite('./' + os.path.basename(file_path), img_array)
    return data_class

