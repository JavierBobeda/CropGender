import os
import json
import pprint
from classes.FaceClass import FaceImage
    

datafolder = "/notebooks/Crisalix/data_engineer_test/data_engineer_test/data"
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
