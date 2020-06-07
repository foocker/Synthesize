from classifier.train_classifier import train
# from detection.train_retinageneral import train
# from detection.train_face import train
# from classifier.train_eff import main

if __name__ == "__main__":
    # import sys
    # print(sys.executable)
    # progressive， finetune, train_primary
    train(mode='pro_finetune')
    # train(mode='progressive')
    # train(mode='train_primary')
    # train()
    # main()


