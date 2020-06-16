# from Synthesize.classifier.benchmark.inference import inference
from detection.inference_face import inference
import os


if __name__ == "__main__":
    # dirs = '*'
    # for file_img in os.listdir(dirs):
    #     if file_img.endswith('jpg'):
    #         img_class = inference(os.path.join(dirs, file_img), threshold=0.45)
    #         print(file_img, img_class, "\n")
    # inference()
