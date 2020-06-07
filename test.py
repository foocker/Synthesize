from classifier.benchmark.evaluate import evaluate
from classifier.benchmark.utils import DynamicAdjustData
from cnn_visual.gradcam import test_gradcam
from detection.test_widerface import test
from detection.test_retinageneral import test_general
# from detection.inference_two_detection import temp_test
from data.coco import test_coco_aug
from detection.anchors.prior_box import test_anchor
from classifier.benchmark.inference import inference
# from recognition.cv2.match import


if __name__ == "__main__":
    # test()
    # test_general()
    # inference('*.jpg', show=True)
    # test_anchor()
    # test_coco_aug()
    evaluate()
    # classes = ['' ]
    # dad = DynamicAdjustData()
    # dad.copy_conf_matrix_path()

    # dad.rename_imgs('')
    # dad.resize_imgs(')
    # test_gradcam('layer4', save_dir='')
    # test_gradcam(7, save_dir='cnn_visual/heatmap/')