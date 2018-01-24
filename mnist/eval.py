from mnist.model import*
from mnist.ops import *
from mnist.read_data import *
from mnist.utils import *

CURRENT_DIR=os.getcwd()
TEST_IMAGE_DIR=CURRENT_DIR+'/test_image/'
MODEL_DIR=CURRENT_DIR+'/model/'

def eval():
    #test_dir='/Users/maxiong/Workpace/Code/Python/GANS/test_image/'
    checkpoint_dir=MODEL_DIR

    y=tf.placeholder(tf.float32,[BATCH_SIZE,10],name='y')
    z=tf.placeholder(tf.float32,[None,100],name='z')

    with tf.variable_scope("for_reuse_scope"):
        G=generator(z,y)
    data_x,data_y=read_data()
    sample_z=np.random.uniform(-1,1,size=(BATCH_SIZE,100))
    sample_labels=data_y[120:184]

    # 读取ckpt 需要sess,saver
    print("Reading checkpoints...")
    ckpt=tf.train.get_checkpoint_state(checkpoint_dir)

    # saver
    saver=tf.train.Saver(tf.all_variables())

    # sess
    sess=tf.Session()
    #sess.run(tf.initialize_all_variables())
    # 从保存的模型中恢复变量，这个if是判断是否存在文件，判断ckpt是判断是否存在checkpoint_dir文件夹，下面一个文件是判断是否存在checkpoint文件
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt)
        # 获得model文件
        ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
        # 恢复变量
        saver.restore(sess,os.path.join(checkpoint_dir,ckpt_name))

    # 用恢复的变量进行生成器的测试
    test_sess=sess.run(G,feed_dict={z:sample_z,y:sample_labels})

    # 保存测试图片
    save_images(test_sess,[8,8],TEST_IMAGE_DIR+'test_%d.png'% 500)

    sess.close()

if __name__=='__main__':
    eval()