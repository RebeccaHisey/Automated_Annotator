import numpy
import os
import cv2
import yaml
import pandas
from YOLOv8 import YOLOv8
import numpy
import os
import cv2
import torch
from torch import nn, optim
from torchvision import models
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import yaml
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, ViTMAEForPreTraining, ViTForImageClassification
from PyQt6.QtWidgets import QApplication,QLabel,QProgressBar,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout

import copy
import gc
from sklearn.decomposition import PCA
from sklearn.cluster import AffinityPropagation

class Cluster():
    def getVITFeatures(self,data,vid_folder,prog_bar = None, prog_message = None):
        ViTModel = ViTForImageClassification.from_pretrained("facebook/vit-mae-base")
        ViTModel.classifier = Identity()
        try:
            ViTModel.cuda("cuda")
            use_cuda = True
        except AssertionError:
            use_cuda = False
        image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        allPredictions = numpy.array([])
        images = []
        ViTModel.eval()
        with torch.no_grad():
            for i in range(len(data)):
                imgFilePath = os.path.join(vid_folder, data[i])
                img_tensor = Image.open(imgFilePath)
                images.append(img_tensor)
                if i % 10 == 0 or i == len(data)-1:
                    images = image_processor(images)
                    imgs = numpy.array(images["pixel_values"])
                    if use_cuda:
                        images["pixel_values"] = torch.from_numpy(imgs).cuda("cuda")
                    else:
                        images["pixel_values"] = torch.from_numpy(imgs)
                    preds = ViTModel(**images)
                    if use_cuda:
                        outputs = preds.logits.cpu().detach().numpy()
                    else:
                        outputs = preds.logits.detach().numpy()
                    if allPredictions.shape[0] == 0:
                        allPredictions = outputs
                    else:
                        allPredictions = numpy.concatenate((allPredictions, outputs), axis=0)
                    print("ViT predictions: {}/{} complete".format(i , len(data)))
                    del imgs
                    del images
                    del preds
                    del outputs
                    gc.collect()
                    if use_cuda:
                        torch.cuda.empty_cache()
                    images = []
        del ViTModel
        del image_processor
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        return numpy.array(allPredictions)

    def performPCA(self,feature_vectors):
        print("Input shape: {}".format(feature_vectors.shape))
        pca = PCA(n_components=0.9,svd_solver="full")
        pca.fit(feature_vectors)
        print("Number of components: {}".format(pca.n_components_))
        print("Explained variance ratios:")
        print(pca.explained_variance_ratio_)
        components = pca.transform(feature_vectors)
        print("Reduced shape: {}".format(components.shape))
        return components

    def performClustering(self,feature_vectors):
        clustering = AffinityPropagation().fit(feature_vectors)
        center_indexes = clustering.cluster_centers_indices_
        print("Number of clusters: {}".format(center_indexes.shape[0]))
        return center_indexes

    def getBestImages(self,image_files,image_dir):
        vit_preds = self.getVITFeatures(image_files, image_dir)
        principal_components = self.performPCA(vit_preds)
        cluster_centers = self.performClustering(principal_components)
        best_images = [image_files[i] for i in cluster_centers]
        return best_images

    def main(self):
        results_dir = "D:\\Automated_Annotation"
        video_folder = "d:\\CentralLine\\Dataset\\Clean_Central_Line_Std\\AN01-20210104-154854\\RGB_right"
        dataset_path = os.path.dirname(os.path.dirname(video_folder))
        video_id = os.path.basename(os.path.dirname(video_folder))
        subtype = os.path.basename(video_folder)
        label_file = pandas.read_csv(os.path.join(video_folder, "{}_{}_Labels.csv".format(video_id, subtype)))

        # Get image features using ViT
        if not os.path.exists(os.path.join(results_dir, "vit_preds.npy")):
            filenames = [label_file["FileName"][i] for i in label_file.index]
            vit_preds = self.getVITFeatures(filenames,video_folder)
            numpy.save(os.path.join(results_dir, "vit_preds.npy"), vit_preds)
        else:
            vit_preds = numpy.load(os.path.join(results_dir, "vit_preds.npy"))

        # Reduce features using PCA
        principal_components = self.performPCA(vit_preds)

        # Perform clustering on reduced features
        cluster_centers = self.performClustering(principal_components)

        # Get label entries corresponding to cluster centers
        entries = label_file.iloc[cluster_centers]
        for i in entries.index:
            image = cv2.imread(os.path.join(video_folder,entries["FileName"][i]))
            try:
                bboxes = eval(entries["Tool bounding box"][i])
            except SyntaxError:
                bboxes = entries["Tool bounding box"][i].replace("[, ","[")
                bboxes = eval(bboxes)
                entries["Tool bounding box"][i] = bboxes
            for bbox in bboxes:
                image = cv2.rectangle(image, (bbox["xmin"], bbox["ymin"]),
                                      (bbox["xmax"], bbox["ymax"]),
                                      (255, 0, 0), 2)
                image = cv2.putText(image, bbox["class"],
                                    (bbox["xmin"], bbox["ymin"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow("Selected images",image)
            cv2.waitKey(0)

        #Create training csv
        fold = [0 for i in range(2*cluster_centers.shape[0])]
        set_names = ["Train" for i in range(cluster_centers.shape[0])] + ["Validation" for i in range(cluster_centers.shape[0])]
        folder = [video_folder for i in range(2*cluster_centers.shape[0])]
        training_csv = pandas.concat([entries,entries])
        training_csv["Fold"] = fold
        training_csv["Set"] = set_names
        training_csv["Folder"] = folder
        label_file["Fold"] = [0 for i in label_file.index]
        label_file["Set"] = ["Test" for i in label_file.index]
        label_file["Folder"] = [video_folder for i in label_file.index]
        training_csv = pandas.concat([training_csv,label_file])
        training_csv.index = [i for i in range(len(training_csv.index))]
        training_csv.to_csv(os.path.join(dataset_path,"auto_selected_images.csv"),index=False)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
if __name__ == "__main__":
    Cluster().main()