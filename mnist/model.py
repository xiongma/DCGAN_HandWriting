from mnist.ops import *
import tensorflow as tf

BATCH_SIZE=64

# 生成器，这里的约束条件也可以叫做特征
def generator(z,y,train=True):

    # y 是一个[BATCH_SIZE,10]维的向量，把y转换成四维张量
    yb=tf.reshape(y,[BATCH_SIZE,1,1,10],name='yb')

    # 把y作为约束条件和z拼接起来,因为y是真实的图片，将其作为约束条件可以保证生成图片的真实性
    # 加y不加yb的原因是，因为y和z具有同样的维数,这里在二维的地方进行连接，相当于把[64,10]+[64,100]=[64,110]
    z=tf.concat([z,y],1,name='z_concat_y')

    # 经过一个全连接层，归一化层和激活层
    h1=tf.nn.relu(batch_norm_layer(fully_connected(z,1024,'g_fully_connected1'),is_train=train,name='g_bn1'))
    # 再次把约束条件和上一层拼接起来
    h1=tf.concat([h1,y],1,name='active1_concat_y')

    # 这里的128暂时理解为神经元的个数
    h2=tf.nn.relu(batch_norm_layer(fully_connected(h1,128*49,'g_fully_connected2'),
                                   is_train=train,name='g_bn2'))
    h2=tf.reshape(h2,[64,7,7,128],name='h2_reshape')

    # 把约束条件和上一层拼接起来，yb为经过升维的向量
    h2=conv_cond_concat(h2,yb,name='active2_concat_y')
    # print(np.shape(h2))
    h3=tf.nn.relu(batch_norm_layer(deconv2d(h2,[64,14,14,128],
                                            name='g_deconv2d3'),is_train=train,name='g_bn3'))

    h3=conv_cond_concat(h3,yb,name='active3_concat_y')

    h4=tf.nn.sigmoid(deconv2d(h3,[64,28,28,1],
                              name='g_deconv2d4'),name='generate_image')

    return h4

# 判别器
def discriminator(image,y,reuse=False):

    # 因为真实数据和生成数据都要经过判别器，所以需要指定reuse是否可用
    # tf.get_variable_scope().reuse_variables() 允许共享当前节点下所有变量，当前节点是指当前的判别器下的所有变量
    if reuse:
        tf.get_variable_scope().reuse_variables()

    # 跟生成器一样，判别器也需要把约束条件串联起来,特征条件
    yb=tf.reshape(y,[BATCH_SIZE,1,1,10],name='yb')
    x=conv_cond_concat(image,yb,name='image_concat_y') # yb[64,1,1,10] image[64,28,28,1]
    # 卷积，激活，串联条件
    h1=lrelu(conv2d(x,11,name='d_conv2d1'),name='lrelu1')
    h1=conv_cond_concat(h1,yb,name='h1_concat_yb')
    h2=lrelu(batch_norm_layer(conv2d(h1,74,name='d_conv2d2'),
                              name='d_bn2'),name='lrelu2')
    # 这里[BATCH_SIZE,-1]表示先适应前面的大小，然后后面的自己再自适应
    h2=tf.reshape(h2,[BATCH_SIZE,-1],name='reshape_lrelu2_to_2d')
    tf.concat([h2,y],1,name='lrelu2_concat_y')

    h3=lrelu(batch_norm_layer(fully_connected(h2,1024,name='d_fully_connected3'),
                              name='d_bn3'),name='lrelu3')
    h3=tf.concat([h3,y],1,name='lrelu3_concat_y')

    h4=fully_connected(h3,1,name='d_result_withouts_sigmoid')
    #print(tf.nn.sigmoid(h4),h4)
    return tf.nn.sigmoid(h4,name='discriminator_result_with_sigmoid'),h4

# 定义训练过程的采样函数,函数的作用是在训练过程中对生成器生成的图片进行采样，所以这个函数必须指定reuse可用
def sampler(z,y,train=True):
    # 共享当前scope下所有的变量，这个scope代表生成器模型下面的变量
    tf.get_variable_scope().reuse_variables()
    return generator(z,y,train=train)