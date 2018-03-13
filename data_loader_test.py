import numpy as np
import os,sys,unittest
from data_loader import *
from numpy.testing import assert_array_equal
from PIL import Image

class data_loader_test(unittest.TestCase):

    def test_train_data(self):
        imgs,lengths,digits=load_svhn_tfrecords('c:/dataset/SVHN/test.tfrecords')
        print(lengths.shape,digits.shape)
        self.assertEqual(lengths[0],1)
        assert_array_equal(digits[:,0],[5,10,10,10,10])
        img=Image.fromarray(imgs[0])
        img.show()

if __name__ == '__main__':
    unittest.main()