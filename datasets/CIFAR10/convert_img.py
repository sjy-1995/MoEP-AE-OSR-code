import cv2
import numpy as np
import os


def unpickle(file):
    # import cPickle
    import _pickle as cPickle
    with open(file, 'rb') as f:
        dict = cPickle.load(f,encoding='iso-8859-1')
    return dict


def main(cifar10_data_dir):
    for i in range(1, 6):
        train_data_file = os.path.join(cifar10_data_dir, 'data_batch_' + str(i))
        print(train_data_file)
        data = unpickle(train_data_file)
        print('unpickle done')
        print(data.keys())
        for j in range(10000):
            img = np.reshape(data['data'][j], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img_name = 'train/' + str(data['labels'][j]) + '_' + str(j + (i - 1) * 10000) + '.jpg'
            cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)

    test_data_file = os.path.join(cifar10_data_dir, 'test_batch')
    data = unpickle(test_data_file)
    for i in range(10000):
        img = np.reshape(data['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        img_name = 'test/' + str(data['labels'][i]) + '_' + str(i) + '.jpg'
        cv2.imwrite(os.path.join(cifar10_data_dir, img_name), img)


if __name__ == "__main__":
    main('/media/robot/study/sjy_openset/cac-openset-master/datasets/data/cifar-10-batches-py')