import cProfile


if __name__ == '__main__':
    cProfile.run('python main.py --model resnet50 --dataset CBIS-DDSM --num_epochs $EPOCHS --no-data_augment')   