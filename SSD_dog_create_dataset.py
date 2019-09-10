import os
import glob
import cv2
import numpy as np
import pandas as pd
import requests, tarfile, io
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import random
import csv

annotation_dir = "Annotation"
image_dir = "Images"


def download_dataset():
    ds_url = [r"http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar", 
              r"http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"]
    for url in ds_url:
        os.system("wget {}".format(url))
    os.system("tar -xvf annotation.tar")
    os.system("tar -xvf images.tar")

def get_image_path(annotation_path):
    path_items = annotation_path.split(os.path.sep)
    return os.path.join(image_dir, path_items[-2], path_items[-1] + '.jpg')

def get_labels(annotation_path):
    list_bndboxes = []
    xml = ET.parse(annotation_path).getroot()
    filename = os.path.basename(get_image_path(annotation_path))
    for obj in xml.findall("object"):
        bndbox = obj.find("bndbox")
        xmin, ymin, xmax, ymax = bndbox.find("xmin"), bndbox.find("ymin"), bndbox.find("xmax"), bndbox.find("ymax")
        xmin, ymin, xmax, ymax = int(xmin.text), int(ymin.text), int(xmax.text), int(ymax.text)
        list_bndboxes.append([filename, xmin, xmax, ymin, ymax, 1])
    return list_bndboxes


def process():
    ds_images_dir = "data"
    all_annotation_files = glob.glob('Annotation/**/*', recursive=True)
    all_annotation_files = [f for f in all_annotation_files if os.path.isfile(f)]
    dataset = random.sample(all_annotation_files, 3000)
    
    if os.path.exists(ds_images_dir):
        os.remove(ds_images_dir)
    os.mkdir(ds_images_dir)
    #Ensure all image are in jpeg format
    for ant in dataset:
        img_path = get_image_path(ant)
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(ds_images_dir, os.path.basename(img_path)), img)
    
    train_ds = dataset[: 2000]
    eval_ds = dataset[2000: 2500]
    test_ds = dataset[2500: ]
    
    label_files = ['labels_train.csv', 'labels_val.csv', 'labels_test.csv']
    ds = [train_ds, eval_ds, test_ds]
    
    for (label_file, d) in zip(label_files, ds):
        labels = []
        for ant in d:
            boxes = get_labels(ant)
            labels.extend(boxes)
        with open(label_file, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile, delimiter=',')
            writer.writerow(['image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id'])
            writer.writerows(labels)


if __name__ == '__main__':
    download_dataset()
    process()


