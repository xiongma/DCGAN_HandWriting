import os
import numpy as np

data_dir='/Users/maxiong/Workpace/Code/Python/GANS/MNIST_data/'

def read_data():

    # 打开训练集
    fd=open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded=np.fromfile(file=fd,dtype=np.uint8)
    # 返回的是28*28的shape，不要被这里的1给迷惑了，这里要注意60000，有60000个大的list，里面每一个list代表一张图片

    trX=loaded[16:].reshape(60000,28,28,1).astype(np.float)
    print(trX[2][14][14][0])
    fd=open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded=np.fromfile(file=fd,dtype=np.uint8)
    trY=loaded[8:].reshape(60000).astype(np.float)

    # 打开测试集
    fd=open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded=np.fromfile(file=fd,dtype=np.uint8)
    teX=loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd=open((os.path.join(data_dir,'t10k-labels-idx1-ubyte')))
    loaded=np.fromfile(file=fd,dtype=np.uint8)
    teY=loaded[8:].reshape(10000).astype(np.float)

    # 由于生成网络由服从某一分布的噪声生成图片，不需要测试集，
    # 所以把训练和测试两部分数据合并
    X=np.concatenate((trX,teX),axis=0) # X [70000,28,28,1]
    Y=np.concatenate((trY,teY),axis=0) # Y [70000]

    # 打乱排序
    seed=547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(Y)

    # 独热编码，使数据有序,这里相当于生成了一个[70000,10]大小的tensor,里面的数据全部为0
    y_vec = np.zeros((len(Y), 10), dtype=np.float)
    for i, label in enumerate(Y):
        y_vec[i, int(Y[i])] = 1.0 # Y[i]是基于0-9之间的，将相当于随机在[70000,10]这个tensor中随机置1

    return X / 255., y_vec