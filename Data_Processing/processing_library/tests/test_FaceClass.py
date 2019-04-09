import os
import shutil
import unittest
import tempfile
from scipy import misc
from classes.FaceImage import FaceImage
 
 
class TestFaceImage(unittest.TestCase):
 
    def setUp(self):
        # Create a temporary directory
        dirname = os.path.dirname(__file__)
        self.test_dir = tempfile.mkdtemp()
        temp_image = shutil.copyfile(os.path.join(dirname, "test_drivers/abba.jpg"), os.path.join(self.test_dir, "abba.jpg"))
        self.imageInst = FaceImage(temp_image)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)
 
    def test_cropping_4_faces(self):
        crops = self.imageInst.cropFace()
        self.assertEqual(len(crops), 4)

    def test_gender_predicted(self):
        crops = self.imageInst.cropFace()
        labels = self.imageInst.predictGender(crops)
        self.assertListEqual(list(labels.values()), ['Female', 'Female', 'Male', 'Male'])

        
if __name__ == '__main__':
    unittest.main()