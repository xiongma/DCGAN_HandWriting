import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

# 偏执
def bias(name,shape,bias_start=0.0,trainable=True):
    dtype=tf.float32
    # 定义偏执
    war =tf.get_variable(name,shape,tf.float32,trainable=trainable,
                         initializer=tf.constant_initializer(bias_start,dtype=dtype))
    return war

# 权重
def weight(name,shape,stddv=0.02,trainable=True):
    dtype=tf.float32
    var = tf.get_variable(name,shape,tf.float32,trainable=trainable,
                          initializer=tf.random_uniform_initializer(stddv,dtype=dtype))

    return var

# 全连接层
def fully_connected(value,output_shape,name='fully_connected',with_w=False):
    shape=value.get_shape().as_list()

    with tf.variable_scope(name): # 在这里加入variable_scope 方便以后使用权重和偏执
        # 定义的列由输出的shape决定
        # 生成网络开始输入的value是[64,110],可以理解为有110个输入神经元，1024个输出神经元,这里的64理解为图片的张数
        # 这里实现了权值共享，每张图片共享同一组权值
        weights=weight('weights',[shape[1],output_shape],0.02)
        biases=bias('bias',[output_shape],0.0)

    if with_w:
        return tf.matmul(value,weights)+biases,weights,biases
    else:
        return tf.matmul(value,weights)+biases

# Leaky_Relu层
def lrelu(x,leak=0.2,name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x,leak*x,name=name)

# relu层
def relu(value,name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)

# 反卷积层，这里我的理解是生成图片
def deconv2d(value,output_shape,k_h=5,k_w=5,strides=[1,2,2,1],name='deconv2d',with_w=False):
    with tf.variable_scope(name):
        # 这里是定义一个[5,5,输出shape形状，输入shape形状],实际就像当于定义了5*5的卷积核
        # 将卷积神经网络反向看 output_shape [64, 14, 14, 128] value (64, 7, 7, 138) output_shape[-1],value.get_shape()[-1]] 128 138
        # 相当于卷积神经网络时128个输入神经元，138个输出神经元，现在只是相反
        weights=weight('weights',[k_h,k_w,output_shape[-1],value.get_shape()[-1]])

        deconv=tf.nn.conv2d_transpose(value,weights,
                                      output_shape,strides=strides)
        # output_shape[-1] 获取最里面那一维的纬度，如果得出的shape为[1,2,3]，得出的是3
        biases=bias('biases',[output_shape[-1]])

        deconv=tf.reshape(tf.nn.bias_add(deconv,biases),deconv.get_shape())
        if with_w:
            return deconv,weights,biases
        else:
            return deconv

# 卷积层
def conv2d(value,output_dim,k_h=5,k_w=5,strides=[1,2,2,1],name='conv2d'):
    with tf.variable_scope(name):
        weights=weight('weights',[k_h,k_w,value.get_shape()[-1],output_dim])
        conv=tf.nn.conv2d(value,weights,strides=strides,padding='SAME')
        biases=bias('biases',[output_dim])
        conv=tf.reshape(tf.nn.bias_add(conv,biases),conv.get_shape())

        return conv

# 把约束条件加入到feature map
def conv_cond_concat(value,cond,name='concat'):

    # 把张量的维度转换为python的list
    value_shapes=value.get_shape().as_list() # value[64,28,28,1] cond[64,1,1,10]
    cond_shapes=cond.get_shape().as_list()

    # 在第三个维度上（feature map维度上）把条件和输入串联起来
    # 条件会被预先设为四维张量的形式，假设输入为[64,32,32,32]维的张量
    # 条件为[64,32,32,10]维的张量，那么输出就是一个[64,32,32,42]维张量，相当于在第3维向量后，进行最后一维维度相加
    with tf.variable_scope(name):
        # tf.ones将shape中的数据全部转换为1
        # value_shapes[0:3]+cond_shapes[3:] a:[1,2,3,4] b:[5,6,7,8,9] 根据全面的式子得出的结果是[1,2,3,8,9],只进行一维的连接
        # value_shapes[0:3]+cond_shapes[3:] =（4,）cond*tf.ones(value_shapes[0:3]+cond_shapes[3:])]=(64, 7, 7, 10)
        return tf.concat([value,cond*tf.ones(value_shapes[0:3]+cond_shapes[3:])],3)

# 数据归一化
def batch_norm_layer(value,is_train=True,name='batch_norm'):
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value,decay=0.9,epsilon=1e-5,scale=True,
                              is_training=is_train,
                              updates_collections=None,scope=scope)

        else:
            return batch_norm(value,decay=0.9,epsilon=1e-5,scale=True,
                              is_training=is_train,reuse=True,
                              updates_collections=None,scope=scope)