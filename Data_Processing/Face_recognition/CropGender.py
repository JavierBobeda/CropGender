import os
import json
import pprint
from classes.FaceImage import FaceImage
    

dirname = os.path.dirname(__file__)
project_fold = os.path.abspath(os.path.join(dirname, os.pardir))
datafolder = os.path.join(project_fold, "data")
images = [os.path.join(datafolder, i) for i in os.listdir(datafolder) if ".jpg" in i]
print("Images selected: ", len(images))
all_labels = {}
for n, im in enumerate(images):
    print(os.path.basename(im), n)
    imFace = FaceImage(im)
    crops = imFace.cropFace()
    if crops:
        labels = imFace.predictGender(crops)
        print(labels)
    else:
        labels = {os.path.basename(im): "Face not detected"}
    all_labels.update(labels)        

pprint.pprint(all_labels)

# Creates a json with the predicted labels per each cropped face.
json = json.dumps(all_labels)
f = open(os.path.join(datafolder, "Crops", "genders.json"),"w")
f.write(json)
f.close()
