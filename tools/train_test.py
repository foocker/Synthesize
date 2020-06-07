# from Synthesize.classifier.train_classifier import train
# from Synthesize.classifier.train_eff import main
# from Synthesize.detection.train_face import train
# from Synthesize.detection.train_detection import train
from detection.train_two_detection import train, train_simple

if __name__ == "__main__":
    # import sys
    # print(sys.executable)
    # main()
    # train()
    train_simple()