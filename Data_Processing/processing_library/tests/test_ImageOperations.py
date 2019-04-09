import os
import shutil
import unittest
import tempfile
from scipy import misc
from classes.ImageOperations import ImageOperations
 
class TestImageOperations(unittest.TestCase):
 
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        f = misc.face()
        impath = os.path.join(self.test_dir,'face.png')
        misc.imsave(impath, f)
        self.imageInst = ImageOperations(impath)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
 
    def test_invalid_bbox(self):
        bbox = [0, 0]
        crop, _ = self.imageInst.crop(bbox=bbox)
        self.assertIsNone(crop)

    def test_failed_cropping(self):
        bbox = [1000, 1000, 1000, 1000]
        crop, _ = self.imageInst.crop(bbox=bbox)
        self.assertIsNone(crop)
        
    def test_correct_cropping(self):
        bbox = [0, 0, 10, 10]
        crop, _ = self.imageInst.crop(bbox=bbox)
        self.assertTupleEqual((30,30,3), crop.shape)

 
if __name__ == '__main__':
    unittest.main()