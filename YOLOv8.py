import numpy
import os
import cv2
import yaml
from ultralytics import YOLO

class YOLOv8:
    def __init__(self,mode):
        self.model = None
        self.mode = mode
        self.class_mapping = None

    def loadModel(self,modelFolder,modelName=None):
        self.model = YOLO(os.path.join(modelFolder,"train/weights/best.pt"))
        with open(os.path.join(modelFolder,"config.yaml"),"r") as f:
            config = yaml.safe_load(f)
            self.class_mapping = config['class_mapping']
            self.mode = config["mode"]

    def predict(self,image):
        '''image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image,0)'''
        results = self.model.predict(image)
        if self.mode == 'detect':
            bboxes = results[0].boxes.xyxy.cpu().numpy()
            class_nums = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            resultList = []
            print(bboxes.shape)
            for i in range(bboxes.shape[0]):
                class_value = class_nums[i]
                class_name = self.class_mapping[class_value]
                xmin,ymin,xmax,ymax = bboxes[i]
                confidence = confs[i] #[class_value]
                bbox = {"class":class_name,
                        "xmin":round(xmin),
                        "ymin":round(ymin),
                        "xmax":round(xmax),
                        "ymax":round(ymax),
                        "conf":confidence}
                resultList.append(bbox)
            return str(resultList)
        elif mode == 'segment':
            masks = results.masks.cpu().numpy()
            #overall_mask = numpy.zeros((image.shape[1],img.shape[0], 1),numpy.uint8)
            return masks

    def createModel(self):
        if self.mode == 'detect':
            self.model = YOLO("yolov8s.pt")
        elif self.mode == 'segment':
            self.model = YOLO("yolov8s-seg.pt")
        return self.model