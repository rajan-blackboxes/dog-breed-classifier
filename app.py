import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision
from glob import glob
from models import model_transfer, use_cuda
from PIL import Image
import torchvision.transforms as transforms,ToPILImage

# list of class names by index, i.e. a name can be accessed like class_names[0]
class_names = [item[4:].replace("_", " ") for item in data_transfer['train'].classes]
path = os.path.abspath('haarcascades/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(path)

#models
model_transfer.load_state_dict(torch.load('weights/model_scratch.pt'))



#get the arguments from command line
def get_args():
    parser = argparse.ArgumentParser("Get the Dog breeds")
    
    #helpers
    i_desc = "Path location of given image"
    s_desc = "Choose the path where file to be saved"
    m_desc = "Morph dog images to human images, if human detected"

    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-i", help=i_desc, required=True)
    optional.add_argument("-s", help=s_desc)
    optional.add_argument("-d", help=d_desc)
    args = parser.parse_args()

    return args



##### Checking dogs and humans ####


def face_detector(img_path):
    """
    Check if human is in the face.
    Args: image path
    Returns: Tuple consisting (boolean human face, detected faces, faces cropped)
    
    """
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)  
    faces = []
    for (x,y,w,h) in face_rects:
        detected_faces = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
        faces.append(cv2.cvtColor(img[y: y + h, x: x + w], cv2.COLOR_BGR2RGB))
    if len(face_rects) > 0:
        return (True, detected_faces, faces)
    else:
        return (False, 0, 0)

def dog_detector(img_path):
	# check if dog face exists
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def predict_breed_transfer(img_path):
	"""load the image and return the predicted breed"""
	
	#Transform as per model_transfer
	transformed = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
	])
	image = transformed(plt.imread(img_path))
	# Add dimension to tensor for number of images
    image = image.unsqueeze(0)

    prediction = model_transfer(image)
    if use_cuda:
    	prediction = model_transfer(image).cuda()
    