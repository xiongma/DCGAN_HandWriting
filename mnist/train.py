import os
from mnist.read_data import *
from mnist.utils import *
from mnist.ops import *
from mnist.model import *
from mnist.model import BATCH_SIZE

def train():
    # 设置global_step,用来记录训练过程中的step
    global_step=tf.Variable(0,name='global_step',trainable=False)

    # 训练过程中的日志保存文件
    train_dir='/Users/maxiong/Workpace/Code/Python/GANS/model/'

    # 放置三个placeholder,y表示约束条件，images表示送入判别器的图片
    # z表示随机噪声
    y=tf.placeholder(tf.float32,[BATCH_SIZE,10],name='y')
    images=tf.placeholder(tf.float32,[64,28,28,1],name='real_images')
    z=tf.placeholder(tf.float32,[None,100],name='z')

    with tf.variable_scope("for_reuse_scope"):
        G=generator(z,y)
        # 这里进行判别的是 只是判断是否为真的图片，加入了约束条件，只是为了让权值更加趋近这个模型想要的权值
        D,D_logits=discriminator(images,y)
        samples=sampler(z,y)
        D_,D_logits_=discriminator(G,y,reuse=True)

    # 损失度计算
    # 计算当输入的图片为真正的图片时，损失度是多少，tf.ones_like(D)表示将判别器输出的概率强制转换成1
    # 理解：求出64张照片的sigmoid输出，然后再与64张照片为1进行比较
    d_loss_real=tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D),logits=D_logits))

    # 计算当输入的图片为生成器生成的图片时，损失度为多少
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_),logits=D_logits_))

    d_loss=d_loss_real+d_loss_fake

    # 看输出的概率跟1差多少
    g_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_),logits=D_logits_))

    # 记录数据
    z_sum=tf.summary.histogram('z',z)
    d_sum=tf.summary.histogram('d',D)
    d__sum=tf.summary.histogram('d_',D_)
    G_sum=tf.summary.image('G',G)

    d_loss_real_sum=tf.summary.scalar('d_real_loss',d_loss_real)
    d_loss_fake_sum=tf.summary.scalar('d_loss_fake',d_loss_fake)
    d_loss_sum=tf.summary.scalar('d_loss',d_loss)

    g_loss_sum=tf.summary.scalar('g_loss',g_loss)

    # 合并节点
    g_sum=tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum=tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    # 生成器和判别器需要更新的变量，用于tf.train.Optimizer的var_list
    t_vars=tf.trainable_variables()
    # print(t_vars)
    d_vars= [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver=tf.train.Saver()

    # 采用优化算法 beta1 表示第一次权值衰减时的指数 beta2 表示权值第二次衰减时的指数
    d_optim=tf.train.AdamOptimizer(0.0002,beta1=0.5) \
        .minimize(loss=d_loss,var_list=d_vars,global_step=global_step)

    g_optim = tf.train.AdamOptimizer(0.0002,beta1=0.5) \
        .minimize(loss=g_loss, var_list=g_vars, global_step=global_step)

    # 开始进行保存
    sess=tf.Session()
    init=tf.initialize_all_variables()
    summary_writer=tf.summary.FileWriter(train_dir,sess.graph)

    # 开始加入数据
    # 这里加入约束条件的意义是在进行生成时，生成的是增加了约束条件的图片。在进行判别的时候，也是加入了相同的相同的约束条件
    # 我们这里也可以加入其它的约束条件，但是生成的约束条件，必须和判别的约束条件一样，都是为了拟合输入生成和真实图片的分布趋近一致
    data_x,data_y=read_data()

    # 开始加入噪音 np.random.uniform返回的是高斯分布的点,这些点在-1和1区间上
    sample_z=np.random.uniform(-1,1,size=(BATCH_SIZE,100))

    # 取出64个示例标签
    sample_labels=data_y[0:64]

    # 进行开始前的初始化
    sess.run(init)

    # 开始训练
    for epoch in range(1):
        batch_idxs=1093
        for idx in range(batch_idxs):
            # 一次取出64张照片，这里返回的是一张图片的shape [64,28,28,1]
            batch_image=data_x[idx*64:(idx+1)*64]
            # 这里返回的是64个标签
            batch_labels=data_y[idx*64:(idx+1)*64] # [64,10]
            batch_z=np.random.uniform(-1,1,size=(BATCH_SIZE,100))

            # 更新D的参数，这里指定了d的梯度和d的loss和，那么就会去寻找需要喂食的节点
            _,summary_str=sess.run([d_optim,d_sum],
                                   feed_dict={images:batch_image,
                                              z:batch_z,
                                              y:batch_labels})
            summary_writer.add_summary(summary_str,idx+1)

            _, summary_str = sess.run([g_optim, g_sum],
                                      feed_dict={z: batch_z,
                                                 y: batch_labels})
            summary_writer.add_summary(summary_str, idx + 1)

            # 更新两次G的参数确保网络稳定
            _,summary_str=sess.run([g_optim,g_sum],
                                   feed_dict={z:batch_z,
                                              y:batch_labels})
            summary_writer.add_summary(summary_str,idx+1)

            # 计算训练过程中的损失，打印出来
            errD_fake=d_loss_fake.eval({z:batch_z,y:batch_labels},sess)
            errD_real=d_loss_real.eval({images:batch_image,y:batch_labels},sess)

            errG=g_loss.eval({z: batch_z, y: batch_labels},sess)

            if idx%20==0:
                print("Epoch: [%2d] [%4d/%4d] d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs, errD_fake + errD_real, errG))

                # 训练过程中，用采样器进行采样，并且保存采样图片
            if idx%100==0:
                sample=sess.run(samples,feed_dict={z:sample_z,y:sample_labels})

                samples_path='/Users/maxiong/Workpace/Code/Python/GANS/image/'

                save_images(sample, [8, 8],
                                samples_path + 'test_%d_epoch_%d.png' % (epoch, idx))

                print('save down')

            if idx%500==0:
                checkpoint_path=os.path.join(train_dir, 'DCGAN_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=idx + 1)

    sess.close()

if __name__ == '__main__':
    train()