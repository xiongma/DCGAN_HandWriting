import scipy.misc
import numpy as np

# 在这里说明一个问题，Tensorflow读取图片时，会把图片的像素值0-255转换为0-1之间的小数
def save_images(images,size,path):
    # 图片归一化，主要用于生成器输出是tanh形式的归一化
    img=(images+1.0)/2.0
    #print(np.shape(images),np.shape(img))

    # 获得图片的长和宽
    h,w=img.shape[1],img.shape[2]

    # 产生一个大画布，用来保存生成的batch_size个图像,np.zeros(返回一个shape，大小为输入的shape，里面的数据全部为0)
    merge_img=np.zeros((h*size[0],w*size[1],3)) # 这里的3暂时理解为RGB，下面不在第2维和第3维加内容，所有还是黑白

    # 循环使得画布特定地方值为某一幅图像的值
    for idx,image in enumerate(images):
        #print(idx,np.shape(image))
        i=idx%size[1]
        # //表示整除
        j=idx//size[1]
        # 保存图片，这里image [28,28,1] 表示黑白的图片，如果为[28,28,3]的话就为rgb图片 image矩阵中的数字为小数
        merge_img[j*h:j*h+h,i*w:i*w+w,:]=image

        # print(scipy.misc.imsave(path,merge_img))
        # 保存画布
    return scipy.misc.imsave(path,merge_img) # 这里保存的时候，会把小数值转换为0-255区间内的数字