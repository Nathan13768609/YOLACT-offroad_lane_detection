import os
import json
import cv2
from dotenv import load_dotenv
load_dotenv()

TRAIN_IMAGES_PATH = os.environ.get('TRAIN_IMAGES')
TRAIN_MASKS_PATH = os.environ.get('TRAIN_MASKS')
VALID_IMAGES_PATH = os.environ.get('VALID_IMAGES')
VALID_MASKS_PATH = os.environ.get('VALID_MASKS')

TRAIN_IMAGE_FILES = os.listdir(TRAIN_IMAGES_PATH)
TRAIN_MASK_FILES = os.listdir(TRAIN_MASKS_PATH)
VALID_IMAGE_FILES = os.listdir(VALID_IMAGES_PATH)
VALID_MASK_FILES = os.listdir(VALID_MASKS_PATH)

TRAIN_IMAGES_FILE_NAME = "train_images.json"
TRAIN_ANNOTATIONS_FILE_NAME = "train_annotations.json"

VALID_IMAGES_FILE_NAME = "valid_images.json"
VALID_ANNOTATIONS_FILE_NAME = "valid_annotations.json"

def getJson(object):
        json_dump = json.dumps(object)
        json_object = json.loads(json_dump)

        return json_object

class Image:
    def __init__(self, id, width, height, file_name):
        self.image = {
            "id": id, 
            "width": width,
            "height": height,
            "file_name": file_name,
        }
    def imageJson(self):
        image_json = getJson(self.image)
        return image_json
    
class Annotation:
    def __init__(self, id, image_id, segmentation, area, category_id=1, bbox=[0,0,0,0], iscrowd=0):
        self.annotation = {
            "id": id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": iscrowd,
        }
    def annotationJson(self):
        annotation_json = getJson(self.annotation)
        return  annotation_json

# PREPARE IMAGES OBJECT

def build_images_object(imageFiles, imagesPath):
    images = []
    for imgFile in imageFiles:
        print(imgFile)
        image_filepath = imagesPath + imgFile
        img = cv2.imread(image_filepath)
        height, width, _ = img.shape
        image_id = image_id = imageFiles.index(imgFile)
        img = Image(image_id, width, height, imgFile).imageJson()
        images.append(img)
        
    return images

# PREPARE ANNOTATIONS OBJECT
def getMaskPolygons(mask_img):
    contours, _ = cv2.findContours(cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
    
        for point in object:
            coords.append(int(point[0][0]))
            coords.append(int(point[0][1]))

        polygons.append(coords)
    
    return polygons

def getMaskArea(mask_img):
    gray = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    mask_pixels = cv2.countNonZero(thresh)

    return mask_pixels

def build_annoations_object(imageFiles, maskFiles, masksPath):
    annotations = []
    for mskFile in maskFiles:
        mask_filepath = masksPath + mskFile
        mask_img = cv2.imread(mask_filepath)
        area = getMaskArea(mask_img)
        segmentation = getMaskPolygons(mask_img)
        annotation_id = maskFiles.index(mskFile)
        image_id = imageFiles.index(mskFile)
        annotation = Annotation(annotation_id, image_id, segmentation, area).annotationJson()
        annotations.append(annotation)

    return annotations

def build_JSON_file(jsonFilename):
    # build json object 
    if jsonFilename == TRAIN_IMAGES_FILE_NAME:
        json_object = build_images_object(TRAIN_IMAGE_FILES, TRAIN_IMAGES_PATH)

    elif jsonFilename == TRAIN_ANNOTATIONS_FILE_NAME:
        json_object = build_annoations_object(TRAIN_IMAGE_FILES, TRAIN_MASK_FILES, TRAIN_MASKS_PATH)
    
    elif jsonFilename == VALID_IMAGES_FILE_NAME:
        json_object = build_images_object(VALID_IMAGE_FILES, VALID_IMAGES_PATH)

    elif jsonFilename == VALID_ANNOTATIONS_FILE_NAME:
        json_object = build_annoations_object(VALID_IMAGE_FILES, VALID_MASK_FILES, VALID_MASKS_PATH)
        
    # write to a JSON file 
    print(f"Building {jsonFilename[:-5]} JSON file for")
    newFilepath = os.path.abspath(os.path.join('data', jsonFilename))
    with open(newFilepath, "w") as file:
        json.dump(json_object, file)
        
def getDatasetObjects():
    json_train_images_path = os.path.abspath(os.path.join('data', TRAIN_IMAGES_FILE_NAME))
    json_train_annotations_path = os.path.abspath(os.path.join('data', TRAIN_ANNOTATIONS_FILE_NAME))
    json_valid_images_path = os.path.abspath(os.path.join('data', VALID_IMAGES_FILE_NAME))
    json_valid_annotations_path = os.path.abspath(os.path.join('data', VALID_ANNOTATIONS_FILE_NAME))

    # build train split JSON
    if not os.path.exists(json_train_images_path):
        build_JSON_file(TRAIN_IMAGES_FILE_NAME)

    if not os.path.exists(json_train_annotations_path):
        build_JSON_file(TRAIN_ANNOTATIONS_FILE_NAME)

    # build validation split JSON
    if not os.path.exists(json_valid_images_path):
        build_JSON_file(VALID_IMAGES_FILE_NAME)

    if not os.path.exists(json_valid_annotations_path):
        build_JSON_file(VALID_ANNOTATIONS_FILE_NAME)
    
    # return path to the created json files
    return json_train_images_path, json_train_annotations_path, json_valid_images_path, json_valid_annotations_path