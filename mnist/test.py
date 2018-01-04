import tensorflow as tf
import numpy as np
import os

fd=open(os.path.join('/Users/maxiong/Workpace/Code/Python/GANS/image/','test_0_epoch_0.png'))
loaded=np.fromfile(fd,dtype=np.uint8)
print()

