import os
import cv2 as cv
from classes.ImageOperations import ImageOperations

class FaceImage(ImageOperations):
    
    def cropFace(self, conf_threshold=0.7):
        """
        Crops as many faces as the faceModel finds, stores them in a directory created where the data is located
        and returns a list with the location of the cropped images.

        Inputs
            conf_threshold

        Returns
            faces: list of the cropped images' path

        """
        faceProto = "/notebooks/Crisalix/data_engineer_test/data_engineer_test/Data_Processing/models/opencv_face_detector.pbtxt"
        faceModel = "/notebooks/Crisalix/data_engineer_test/data_engineer_test/Data_Processing/models/opencv_face_detector_uint8.pb"
        net = cv.dnn.readNet(faceModel, faceProto)
        frameOpencvDnn = self.image.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
        
        if not bboxes:
            print("No faces detected in {}".format(os.path.basename(self.imageName)))
            return None
        else:
            print("{} faces detected".format(len(bboxes)))
        
        # Creates cropping directory
        crop_dir = os.path.join(self.imagePath, "Crops")
        try:
            os.stat(crop_dir)
        except:
            os.mkdir(crop_dir)
        
        # Crops images and stores them
        faces = []
        for i, bbox in enumerate(bboxes):
            cropName = os.path.join(crop_dir, self.imageName.split(".")[0] + "_crop{}".format(str(i+1)) + ".jpg")
            face, _ = self.crop(bbox=bbox)
            cv.imwrite(cropName, face)
            faces.append(cropName)
            
        return faces
    
    def predictGender(self, faceList):
        """
        Predicts the gender of a series of faces.

        Inputs
            faceList: list of images to be analyzed.

        Returns
            genders: dict of the labels for each image.

        """
        
        genderProto = "/notebooks/Crisalix/data_engineer_test/data_engineer_test/Data_Processing/models/gender_deploy.prototxt"
        genderModel = "/notebooks/Crisalix/data_engineer_test/data_engineer_test/Data_Processing/models/gender_net.caffemodel"

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        genderList = ['Male', 'Female']
        genderNet = cv.dnn.readNet(genderModel, genderProto)
        
        genders = {}
        for faceIn in faceList:
            face = cv.imread(faceIn)
            try:
                blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                # print("Gender Output : {}".format(genderPreds))
                # print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
                genders[os.path.basename(faceIn)] =  gender
            except cv.error as e:
                genders[os.path.basename(faceIn)] = "Gender not recognized"
            
        return genders


