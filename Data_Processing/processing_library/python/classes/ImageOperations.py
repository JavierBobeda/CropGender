import os
import cv2 as cv


class ImageOperations:
    
    def __init__(self, ImageIn = ""):
        if not os.path.exists(ImageIn):
            raise Exception("No image found")
        self.imagePath = os.path.dirname(ImageIn)
        self.imageName = os.path.basename(ImageIn)
        self.image = cv.imread(ImageIn)
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

    def crop(self, bbox, padding=20, save=False, savepath=""):
    
        """
        Crops an image.

        Inputs
            conf_threshold
            padding
            save
            savepath

        Returns
            crop: numpy array with the cropped image
            savepath: path of the saved cropped image

        """

        if len(bbox) != 4:
            print ("Invalid bounding box dimensions")
            return None, None

        crop = self.image[max(0,bbox[1]-padding):min(bbox[3]+padding,self.height-1),max(0,bbox[0]-padding):min(bbox[2]+padding, self.width-1)]
        
        if not crop.shape[0] or not crop.shape[1]:
            print("Failed cropping!")
            return None, None
        
        if save or savepath:
            if not savepath:
                savepath = os.path.join(self.imagePath, self.imageName.split(".")[0] + "_crop" + ".jpg")

            cv.imwrite(savepath, crop)
            
        return crop, savepath
