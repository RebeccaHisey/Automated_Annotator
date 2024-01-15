import os
import shutil
import sys
import numpy
import cv2
import pandas
import math

import yaml
from PyQt6 import QtCore
from PyQt6.QtCore import Qt,QPoint
from PyQt6.QtWidgets import QProgressBar,QApplication,QDoubleSpinBox,QLabel,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QPushButton,QSpacerItem,QFileDialog,QTabWidget,QComboBox,QCheckBox,QSlider,QMainWindow,QLineEdit
from PyQt6.QtGui import QImage,QPixmap,QShortcut,QKeySequence
from superqt import QRangeSlider
from cluster import Cluster
import Train_YOLOv8
from YOLOv8 import YOLOv8

class ImageLabel(QLabel):
    def __init__(self,parent=None):
        super(QLabel,self).__init__(parent)
        self.setMouseTracking(False)


class Automated_Annotator(QWidget):
    def __init__(self):
        super().__init__()
        self.imgShape = (480, 640, 3)
        self.originalImageShape = None
        self.setMouseTracking(False)
        self.modifyBBoxStarted = False
        self.yolo = None
        self.modelDir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"Models")
        self.datacsv = None
        self.imageDirectory = None
        self.model = None
        self.setWindowTitle("Automated Annotator")
        self.setupWidget()
        self.setupWidgetConnections()
        self.show()

    def setupWidget(self):
        titleMsg = QLabel("<h1>Automated Annotator<h1>")
        titleMsg.move(20, 20)
        mainLayout = QHBoxLayout()

        #############################
        # Create Image manipulations layout
        #############################
        layout = QVBoxLayout()
        layout.addWidget(titleMsg)
        hbox = QHBoxLayout()
        self.selectModelLabel = QLabel("Select Model:")
        self.selectModelComboBox = QComboBox()
        models = [x for x in os.listdir(self.modelDir) if os.path.isdir(os.path.join(self.modelDir,x))]
        self.selectModelComboBox.addItems(["Select model", "Create new model"] + models)
        self.updateModelButton = QPushButton("Update model")
        self.updateModelButton.setEnabled(False)
        hbox.addWidget(self.selectModelLabel)
        hbox.addWidget(self.selectModelComboBox)
        hbox.addWidget(self.updateModelButton)
        layout.addLayout(hbox)
        self.selectImageDirButton = QPushButton("Select Image Directory")
        self.selectImageDirButton.setEnabled(False)
        layout.addWidget(self.selectImageDirButton)
        self.imageDirectoryLabel = QLabel("Image Directory: ")
        layout.addWidget(self.imageDirectoryLabel)
        layout.addItem(QSpacerItem(100, 20))

        self.deleteBoxButton = QPushButton("Delete current box")
        layout.addWidget(self.deleteBoxButton)

        self.detectionWidget = QWidget()
        self.detectionLayout = QGridLayout()
        self.currentBoxSelectorLabel = QLabel("Current box")
        self.detectionLayout.addWidget(self.currentBoxSelectorLabel, 1, 0)
        self.currentBoxSelector = QComboBox()
        self.detectionLayout.addWidget(self.currentBoxSelector, 1, 1,1,2)

        self.classSelectorLabel = QLabel("Class name")
        self.detectionLayout.addWidget(self.classSelectorLabel, 2, 0)
        self.classSelector = QComboBox()
        self.classSelector.addItems(["Select class name","Add new class"])
        self.detectionLayout.addWidget(self.classSelector, 2, 1, 1, 2)

        self.xCoordinateLabel = QLabel("X Coordinates")
        self.detectionLayout.addWidget(self.xCoordinateLabel, 3, 0)
        self.xminSelector = QDoubleSpinBox()
        self.xminSelector.setMaximum(1.0)
        self.xminSelector.setMinimum(0.0)
        self.xminSelector.setSingleStep(0.05)
        self.xmaxSelector = QDoubleSpinBox()
        self.xmaxSelector.setMaximum(1.0)
        self.xmaxSelector.setMinimum(0.0)
        self.xmaxSelector.setSingleStep(0.05)
        self.detectionLayout.addWidget(self.xminSelector, 3, 1)
        self.detectionLayout.addWidget(self.xmaxSelector, 3, 2)

        self.yCoordinateLabel = QLabel("Y Coordinates")
        self.detectionLayout.addWidget(self.yCoordinateLabel, 4, 0)
        self.yminSelector = QDoubleSpinBox()
        self.yminSelector.setMaximum(1.0)
        self.yminSelector.setMinimum(0.0)
        self.yminSelector.setSingleStep(0.05)
        self.ymaxSelector = QDoubleSpinBox()
        self.ymaxSelector.setMaximum(1.0)
        self.ymaxSelector.setMinimum(0.0)
        self.ymaxSelector.setSingleStep(0.05)
        self.detectionLayout.addWidget(self.yminSelector, 4, 1)
        self.detectionLayout.addWidget(self.ymaxSelector, 4, 2)
        self.detectionWidget.setLayout(self.detectionLayout)
        layout.addWidget(self.detectionWidget)

        self.messageLabel = QLabel("")
        layout.addItem(QSpacerItem(100, 20))
        layout.addWidget(self.messageLabel)
        mainLayout.addLayout(layout)
        mainLayout.addItem(QSpacerItem(20, 100))

        #############################
        # Create Image layout
        #############################
        imageLayout = QVBoxLayout()
        self.imageLabel = ImageLabel()
        image = numpy.zeros(self.imgShape)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImage = QImage(image.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixelmap = QPixmap.fromImage(qImage)
        self.imageLabel.setPixmap(pixelmap)
        imageLayout.addWidget(self.imageLabel)
        mainLayout.addLayout(imageLayout)

        self.createModelWidget()
        self.createClassWidget()

        self.setLayout(mainLayout)
        self.setupHotkeys()

    def setupWidgetConnections(self):
        self.selectModelComboBox.currentIndexChanged.connect(self.onModelSelected)
        self.selectImageDirButton.clicked.connect(self.onSelectImageDirectory)
        self.currentBoxSelector.currentIndexChanged.connect(self.onCurrentBoxChanged)
        self.classSelector.currentIndexChanged.connect(self.onClassSelected)
        self.xminSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.xmaxSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.yminSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.ymaxSelector.valueChanged.connect(self.updateBBoxCoordinates)
        self.deleteBoxButton.clicked.connect(self.onDeleteBox)
        self.updateModelButton.clicked.connect(self.updateModel)

    def mouseMoveEvent(self,event):
        if self.currentBoxSelector.currentIndex() >=2 and self.modifyBBoxStarted:
            cursorPosition = event.pos()
            cursorPosition = (cursorPosition.x(),cursorPosition.y())
            imageWidgetPosition = (self.imageLabel.x(),self.imageLabel.y())
            imageXCoordinate = max(0,min(self.imgShape[1],cursorPosition[0]-imageWidgetPosition[0]))
            imageYCoordinate = max(0,min(self.imgShape[0],cursorPosition[1]-imageWidgetPosition[1]))
            box_index = self.currentBoxSelector.currentIndex() - 2
            self.currentBBoxes = eval(str(self.currentBBoxes))
            self.currentBBoxes[box_index]["xmin"] = min(imageXCoordinate, self.startingPoint[0])/self.imgShape[1]
            self.currentBBoxes[box_index]["ymin"] = min(imageYCoordinate, self.startingPoint[1])/self.imgShape[0]
            self.currentBBoxes[box_index]["xmax"] = max(imageXCoordinate, self.startingPoint[0])/self.imgShape[1]
            self.currentBBoxes[box_index]["ymax"] = max(imageYCoordinate, self.startingPoint[1])/self.imgShape[0]
            #print(self.currentBBoxes)
            self.setBBoxCoordinates(self.currentBBoxes[box_index])
            self.setImage(self.currentImage)

    def mousePressEvent(self,event):
        if self.currentBoxSelector.currentIndex() >=2:
            cursorPosition = event.pos()
            cursorPosition = (cursorPosition.x(), cursorPosition.y())
            imageWidgetPosition = (self.imageLabel.x(), self.imageLabel.y())
            imageWidgetShape = (self.imageLabel.width(),self.imageLabel.height())
            if imageWidgetPosition[0]<=cursorPosition[0]<=(imageWidgetPosition[0]+imageWidgetShape[0]) and imageWidgetPosition[1]<=cursorPosition[1]<=(imageWidgetPosition[1]+imageWidgetShape[1]):
                self.modifyBBoxStarted = True

                imageXCoordinate = max(0, min(self.imgShape[1], cursorPosition[0] - imageWidgetPosition[0]))
                imageYCoordinate = max(0, min(self.imgShape[0], cursorPosition[1] - imageWidgetPosition[1]))
                self.startingPoint = (imageXCoordinate,imageYCoordinate)
                box_index = self.currentBoxSelector.currentIndex() - 2
                self.currentBBoxes = eval(str(self.currentBBoxes))
                self.currentBBoxes[box_index]["xmin"] = imageXCoordinate/self.imgShape[1]
                self.currentBBoxes[box_index]["ymin"] = imageYCoordinate/self.imgShape[0]
                self.currentBBoxes[box_index]["xmax"] = imageXCoordinate/self.imgShape[1]
                self.currentBBoxes[box_index]["ymax"] = imageYCoordinate/self.imgShape[0]
                self.setBBoxCoordinates(self.currentBBoxes[box_index])
                self.setImage(self.currentImage)
        else:
            self.modifyBBoxStarted = False

    def mouseReleaseEvent(self,event):
        self.modifyBBoxStarted = False
        if self.currentBoxSelector.currentIndex() >= 2 and self.modifyBBoxStarted:
            self.updateLabelFile()
            self.setImage(self.currentImage)

    def setupHotkeys(self):
        allVerticalFlipShortcut = QShortcut(self)
        allVerticalFlipShortcut.setKey("v")
        allVerticalFlipShortcut.activated.connect(self.onFlipAllImageVClicked)

        allhorizontalFlipShortcut = QShortcut(self)
        allhorizontalFlipShortcut.setKey("h")
        allhorizontalFlipShortcut.activated.connect(self.onFlipAllImageHClicked)

        nextShortcut = QShortcut(self)
        nextShortcut.setKey("n")
        nextShortcut.activated.connect(self.showNextImage)

        previousShortcut = QShortcut(self)
        previousShortcut.setKey("p")
        previousShortcut.activated.connect(self.showPreviousImage)

        removeShortcut = QShortcut(self)
        removeShortcut.setKey("d")
        removeShortcut.activated.connect(self.removeImage)

        exportShortcut = QShortcut(self)
        exportShortcut.setKey("Ctrl+e")
        exportShortcut.activated.connect(self.ExportToLabelFile)

    def removeImage(self):
        completed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"]==self.imageDirectory) & (self.imageLabelFile["Status"]=="Complete")]
        if len(completed_imgs.index) == len(self.imageFiles):
            file_name_to_remove = self.currentImage
            idx = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==file_name_to_remove].index[0]
            self.showNextImage()
            self.imageLabelFile = self.imageLabelFile.drop(idx,axis="index")
            self.imageLabelFile.index = [i for i in range(len(self.imageLabelFile.index))]
            self.imageFiles.remove(file_name_to_remove)
            self.imageLabelFile.to_csv(os.path.join(self.modelDir, self.selectModelComboBox.currentText(), "image_labels.csv"), index=False)

    def ExportToLabelFile(self):
        completed_imgs = self.imageLabelFile.loc[
            (self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Complete")]
        img_files = self.imageLabelFile.loc[self.imageLabelFile["Folder"] == self.imageDirectory]
        if len(completed_imgs.index) == len(img_files.index):
            videoId = os.path.basename(os.path.dirname(self.imageDirectory))
            subtype = os.path.basename(self.imageDirectory)


            if os.path.exists(os.path.join(self.imageDirectory, "{}_{}_Labels.csv".format(videoId, subtype))):
                self.videoID = videoId
                self.subtype = subtype
                label_file_path = os.path.join(self.imageDirectory, "{}_{}_Labels.csv".format(videoId, subtype))
                self.labelFile = pandas.read_csv(label_file_path)

            elif os.path.exists(os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype))):
                self.videoID = subtype
                self.subtype = None
                label_file_path = os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype))
                self.labelFile = pandas.read_csv(label_file_path)

            else:
                self.videoID = subtype
                self.subtype = None
                label_file_path = os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype))
                self.labelFile = pandas.DataFrame({"FileName":[img_files["FileName"][i] for i in img_files.index]})

            idx_to_drop = []
            for i in self.labelFile.index:
                fileName = self.labelFile["FileName"][i]
                entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==fileName]
                if entry.empty:
                    idx_to_drop.append(i)

            self.labelFile = self.labelFile.drop(idx_to_drop,axis="index")
            self.labelFile.index = [i for i in range(len(self.labelFile.index))]
            self.labelFile["Tool bounding box"] = [self.convertBBoxes(img_files["Bounding boxes"][i]) for i in img_files.index]
            self.labelFile.to_csv(label_file_path,index=False)

    def convertBBoxes(self,bboxes):
        bboxes = eval(str(bboxes))
        for bbox in bboxes:
            bbox["xmin"] = int(bbox["xmin"]*self.originalImageShape[1])
            bbox["xmax"] = int(bbox["xmax"] * self.originalImageShape[1])
            bbox["ymin"] = int(bbox["ymin"] * self.originalImageShape[0])
            bbox["ymax"] = int(bbox["ymax"] * self.originalImageShape[0])
        return bboxes

    def getCurrentIndex(self,idx,indexes):
        idx_found = False
        i = -1
        while not idx_found and i<len(indexes)-1:
            i += 1
            if indexes[i] == idx:
                idx_found = True
        return i

    def checkForNoneBBoxes(self):
        bboxes = eval(str(self.currentBBoxes))
        found_None = False
        for box in bboxes:
            if box["class"] is None:
                found_None = True
        return found_None



    def showNextImage(self):
        images_to_review = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & ((self.imageLabelFile["Status"] == "Review") | (self.imageLabelFile["Status"] == "Reviewed"))]
        completed_imgs =  self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Complete")]
        current_image = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        none_bboxes = self.checkForNoneBBoxes()
        if not none_bboxes:
            if not self.imageLabelFile["Status"][current_image.index[0]] == "Complete":
                self.imageLabelFile["Status"][current_image.index[0]] = "Reviewed"
            if not images_to_review.empty:
                img_idxs = images_to_review.index
                curr_idx = self.getCurrentIndex(current_image.index[0],img_idxs)
                if curr_idx < len(img_idxs)-1:
                    next_idx = img_idxs[curr_idx+1]
                    self.updateModelButton.setEnabled(False)
                else:
                    next_idx = img_idxs[curr_idx]
                    self.updateLabelFile()
                    self.updateModelButton.setEnabled(True)
            else:#if len(completed_imgs.index)==len(self.imageFiles):
                img_idxs = completed_imgs.index
                curr_idx = self.getCurrentIndex(current_image.index[0], img_idxs)
                if curr_idx < len(img_idxs)-1:
                    next_idx = img_idxs[curr_idx+1]
                    self.updateModelButton.setEnabled(False)
                else:
                    next_idx = img_idxs[curr_idx]
            self.currentImage = self.imageLabelFile["FileName"][next_idx]
            self.currentBBoxes = self.imageLabelFile["Bounding boxes"][next_idx]
            self.updateWidget()
        else:
            self.messageLabel.setText("All bounding boxes must be assigned a class")


    def updateWidget(self):
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)

        self.updateBBoxSelector()
        if len(eval(str(self.currentBBoxes))) > 0:
            self.currentBoxSelector.setCurrentIndex(2)
        else:
            self.currentBoxSelector.setCurrentIndex(0)
        self.onCurrentBoxChanged()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage)
        msg = ""
        imgs_remaining = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) &( self.imageLabelFile["Status"]=="Review")]
        reviewed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) &( self.imageLabelFile["Status"]=="Reviewed")]
        completed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) &( self.imageLabelFile["Status"]=="Complete")]
        if len(imgs_remaining.index) > 0:
            msg += "{}/{} images remaining in current set\n".format(len(imgs_remaining.index),len(imgs_remaining.index) + len(reviewed_imgs.index))
        elif len(completed_imgs.index)==len(self.imageFiles):
            msg+= "Review complete! Please update model, then select next video."
        else:
            msg += "Current set completed. Please update model for next image set.\n"
        msg += "{}/{} images annotated".format(len(reviewed_imgs.index) + len(completed_imgs.index),len(self.imageFiles))
        self.messageLabel.setText(msg)

    def showPreviousImage(self):
        images_to_review = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & ((self.imageLabelFile["Status"] == "Review") | (self.imageLabelFile["Status"] == "Reviewed"))]
        current_image = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        completed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Complete")]
        if not images_to_review.empty:
            img_idxs = images_to_review.index
            curr_idx = self.getCurrentIndex(current_image.index[0],img_idxs)
            if curr_idx > 0:
                next_idx = img_idxs[curr_idx-1]
                self.updateModelButton.setEnabled(False)
            else:
                next_idx = img_idxs[curr_idx]
                self.updateModelButton.setEnabled(True)
        elif len(completed_imgs.index) == len(self.imageFiles):
            img_idxs = completed_imgs.index
            curr_idx = self.getCurrentIndex(current_image.index[0], img_idxs)
            if curr_idx > 0:
                next_idx = img_idxs[curr_idx - 1]
                self.updateModelButton.setEnabled(False)
            else:
                next_idx = img_idxs[curr_idx]
        self.currentImage = self.imageLabelFile["FileName"][next_idx]
        self.currentBBoxes = self.imageLabelFile["Bounding boxes"][next_idx]
        self.updateWidget()

    def onFlipAllImageHClicked(self):
        for img in self.imagefiles:
            imagePath = os.path.join(self.imageDirectory, img)
            image = cv2.imread(imagePath)
            image = cv2.flip(image, 1)
            cv2.imwrite(imagePath, image)
        self.setImage(self.currentImage)

    def onFlipAllImageVClicked(self):
        for img in self.imagefiles:
            imagePath = os.path.join(self.imageDirectory, img)
            image = cv2.imread(imagePath)
            image = cv2.flip(image, 0)
            cv2.imwrite(imagePath,image)
        self.setImage(self.currentImage)

    def createModelWidget(self):
        self.modelwindow = QWidget()
        self.modelwindow.setWindowTitle("Create new model")
        layout = QVBoxLayout()
        create_model_name_label = QLabel("Model name:")
        self.create_model_line_edit = QLineEdit()
        accept_reject_layout = QHBoxLayout()
        self.accept_button = QPushButton("Ok")
        self.reject_button = QPushButton("Cancel")
        accept_reject_layout.addWidget(self.accept_button)
        accept_reject_layout.addWidget(self.reject_button)
        layout.addWidget(create_model_name_label)
        layout.addWidget(self.create_model_line_edit)
        layout.addLayout(accept_reject_layout)
        self.modelwindow.setLayout(layout)

        self.accept_button.clicked.connect(self.onModelCreated)
        self.reject_button.clicked.connect(self.onModelCreationCancelled)

    def createClassWidget(self):
        self.classwindow = QWidget()
        self.classwindow.setWindowTitle("Create new model")
        layout = QVBoxLayout()
        create_class_name_label = QLabel("Class name:")
        self.create_class_line_edit = QLineEdit()
        accept_reject_layout = QHBoxLayout()
        self.class_accept_button = QPushButton("Ok")
        self.class_reject_button = QPushButton("Cancel")
        accept_reject_layout.addWidget(self.class_accept_button)
        accept_reject_layout.addWidget(self.class_reject_button)
        layout.addWidget(create_class_name_label)
        layout.addWidget(self.create_class_line_edit)
        layout.addLayout(accept_reject_layout)
        self.classwindow.setLayout(layout)

        self.class_accept_button.clicked.connect(self.onClassAdded)
        self.class_reject_button.clicked.connect(self.onClassAddedCancelled)

    def createProgressWidget(self,num_images):
        self.prog_bar_window = QWidget()
        self.prog_bar_window.setWindowTitle("Selecting initial images")
        layout = QVBoxLayout()
        self.prog_bar = QProgressBar(self.prog_bar_window)
        #self.prog_bar.setGeometry(30,40,200,25)
        self.prog_bar.setMinimum(0)
        self.prog_bar.setMaximum((num_images//10)+1+1)
        self.prog_bar.setValue(0)
        self.prog_message_label = QLabel("Initializing model")
        layout.addWidget(self.prog_bar)
        layout.addWidget(self.prog_message_label)
        self.prog_bar_window.setLayout(layout)

    def onClassSelected(self):
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        if self.classSelector.currentText() == "Add new class":
            self.classwindow.show()
            #self.updateBBoxSelector()
        elif self.classSelector.currentText() != "Select class name":
            self.updateBBoxClass()
            self.updateBBoxSelector()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage)

    def updateBBoxClass(self):
        self.currentBBoxes = eval(str(self.currentBBoxes))
        className = self.classSelector.currentText()
        bbox_index = self.currentBoxSelector.currentIndex()-2
        print(bbox_index)
        self.currentBBoxes[bbox_index]["class"] = className
        self.updateLabelFile()

    def updateBBoxCoordinates(self):
        self.currentBBoxes = eval(str(self.currentBBoxes))
        bbox_index = self.currentBoxSelector.currentIndex()-2
        xmin = self.xminSelector.value()
        ymin = self.yminSelector.value()
        xmax = self.xmaxSelector.value()
        ymax = self.ymaxSelector.value()
        self.currentBBoxes[bbox_index]["xmin"] = xmin
        self.currentBBoxes[bbox_index]["xmax"] = xmax
        self.currentBBoxes[bbox_index]["ymin"] = ymin
        self.currentBBoxes[bbox_index]["ymax"] = ymax
        self.setImage(self.currentImage)
        self.updateLabelFile()

    def onCurrentBoxChanged(self):
        box_index = self.currentBoxSelector.currentIndex()
        self.currentBBoxes = eval(str(self.currentBBoxes))
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        if self.currentBoxSelector.currentText() == "Add new box":
            bbox = {"class":None,"xmin":0.45,"ymin":0.45,"xmax":0.55,"ymax":0.55}
            self.setBBoxCoordinates(bbox)
            self.currentBBoxes.append(bbox)
            self.classSelector.setCurrentIndex(0)
            self.updateBBoxSelector()
            self.currentBoxSelector.setCurrentIndex(self.currentBoxSelector.count()-1)
        elif self.currentBoxSelector.currentText() != "Select box":
            bbox = self.currentBBoxes[box_index-2]
            self.setBBoxCoordinates(bbox)
            self.classSelector.setCurrentText(bbox["class"])
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage)

    def onDeleteBox(self):
        box_index = self.currentBoxSelector.currentIndex()-2
        self.currentBBoxes = eval(str(self.currentBBoxes))
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        xmin_signals = self.xminSelector.blockSignals(True)
        xmax_signals = self.xmaxSelector.blockSignals(True)
        ymin_signals = self.yminSelector.blockSignals(True)
        ymax_signals = self.ymaxSelector.blockSignals(True)
        if self.currentBoxSelector!="Select box":
            self.currentBBoxes.pop(box_index)
            self.updateBBoxSelector()
            self.updateLabelFile()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.xminSelector.blockSignals(xmin_signals)
        self.xmaxSelector.blockSignals(xmax_signals)
        self.yminSelector.blockSignals(ymin_signals)
        self.ymaxSelector.blockSignals(ymax_signals)
        self.setImage(self.currentImage)


    def setBBoxCoordinates(self,bbox):
        self.xminSelector.setValue(bbox["xmin"])
        self.xmaxSelector.setValue(bbox["xmax"])
        self.yminSelector.setValue(bbox["ymin"])
        self.ymaxSelector.setValue(bbox["ymax"])

    def onModelCreated(self):
        model_signals = self.selectModelComboBox.blockSignals(True)
        self.modelwindow.hide()
        modelName = self.create_model_line_edit.text()
        if not os.path.exists(os.path.join(self.modelDir,modelName)):
            os.mkdir(os.path.join(self.modelDir,modelName))
            self.imageLabelFile = pandas.DataFrame(columns=["Folder","FileName","Status","Bounding boxes"])
            self.imageLabelFile.to_csv(os.path.join(self.modelDir,modelName,"image_labels.csv"),index=False)
            self.selectModelComboBox.addItem(modelName)
            self.selectModelComboBox.setCurrentText(modelName)
            self.messageLabel.setText("")
            self.selectImageDirButton.setEnabled(True)
        else:
            self.messageLabel.setText("A model named {} already exists".format(modelName))
            self.modelwindow.show()
        self.selectModelComboBox.blockSignals(model_signals)

    def onModelCreationCancelled(self):
        self.selectModelComboBox.setCurrentIndex(0)
        self.modelwindow.hide()

    def onClassAdded(self):
        bbox_signals = self.currentBoxSelector.blockSignals(True)
        class_signals = self.classSelector.blockSignals(True)
        self.classwindow.hide()
        className = self.create_class_line_edit.text()
        class_names = sorted([self.classSelector.itemText(i) for i in range(2,self.classSelector.count()-1)])
        if not className in class_names:
            self.classSelector.addItem(className)
            self.classSelector.setCurrentText(className)
            self.updateBBoxClass()
            self.updateBBoxSelector()
        else:
            self.messageLabel.setText("A class named {} already exists".format(className))
            self.classwindow.show()
        self.currentBoxSelector.blockSignals(bbox_signals)
        self.classSelector.blockSignals(class_signals)
        self.setImage(self.currentImage)

        if not os.path.exists(os.path.join(self.modelDir,self.selectModelComboBox.currentText(),"class_mapping.yaml")):
            class_mapping = dict(zip([i for i in range(len(class_names))],class_names))
            with open(os.path.join(self.modelDir,self.selectModelComboBox.currentText(), "class_mapping.yaml"), "w") as f:
                yaml.dump(class_mapping, f)

    def onClassAddedCancelled(self):
        self.classSelector.setCurrentIndex(0)
        self.classwindow.hide()

    def onModelSelected(self):
        if self.selectModelComboBox.currentText() == "Create new model":
            self.modelwindow.show()
        elif self.selectModelComboBox.currentText() != "Select model":
            self.selectImageDirButton.setEnabled(True)
            modelName = self.selectModelComboBox.currentText()
            self.imageLabelFile = pandas.read_csv(os.path.join(self.modelDir,modelName,"image_labels.csv"))
            with open(os.path.join(self.modelDir,modelName,"class_mapping.yaml"),"r") as f:
                class_mapping = yaml.safe_load(f)
            class_signals = self.classSelector.blockSignals(True)
            for key in class_mapping:
                self.classSelector.addItem(class_mapping[key])
            self.classSelector.blockSignals(class_signals)
        else:
            self.selectImageDirButton.setEnabled(False)


    def onSelectImageDirectory(self):
        window = QWidget()
        window.setWindowTitle("Select Image Directory")
        self.imageDirectory = QFileDialog.getExistingDirectory(window,"C://")

        self.imageDirectoryLabel.setText("Image Directory: \n{}".format(self.imageDirectory))

        label_filepath = os.path.join(self.modelDir, self.selectModelComboBox.currentText(), "image_labels.csv")
        if os.path.exists(label_filepath):
            self.imageLabelFile = pandas.read_csv(label_filepath)
            imgs = self.imageLabelFile.loc[self.imageLabelFile["Folder"] == self.imageDirectory]
            if not imgs.empty:
                self.imageFiles = [self.imageLabelFile["FileName"][i] for i in imgs.index]
            else:
                self.getImageFileNames()
        else:
            self.getImageFileNames()
        if len(self.imageFiles) > 0:
            self.addImagesToLabelFile()
        else:
            self.messageLabel.setText("No images found in directory")
            self.setImage()

    def getImageFileNames(self):
        videoId = os.path.basename(os.path.dirname(self.imageDirectory))
        subtype = os.path.basename(self.imageDirectory)
        if os.path.exists(os.path.join(self.imageDirectory,"{}_{}_Labels.csv".format(videoId,subtype))):
            self.videoID = videoId
            self.subtype = subtype
            self.labelFile = pandas.read_csv(os.path.join(self.imageDirectory,"{}_{}_Labels.csv".format(videoId,subtype)))
            self.imageFiles = [self.labelFile["FileName"][i] for i in self.labelFile.index]

        elif os.path.exists(os.path.join(self.imageDirectory,"{}_Labels.csv".format(subtype))):
            self.videoID = subtype
            self.subtype = None
            self.labelFile = pandas.read_csv(os.path.join(self.imageDirectory, "{}_Labels.csv".format(subtype)))
            self.imageFiles = [self.labelFile["FileName"][i] for i in self.labelFile.index]
        else:
            self.imageFiles = [x for x in os.listdir(self.imageDirectory) if (".jpg" in x) or (".png" in x)]


    def addImagesToLabelFile(self):
        img_labels = self.imageLabelFile.loc[self.imageLabelFile["Folder"]==self.imageDirectory]
        message = ""
        if img_labels.empty:
            new_df = pandas.DataFrame({"Folder":[self.imageDirectory for i in self.imageFiles],
                                       "FileName": self.imageFiles,
                                       "Status":["Incomplete" for i in self.imageFiles],
                                       "Bounding boxes": [[] for i in self.imageFiles]})
            self.imageLabelFile = pandas.concat([self.imageLabelFile,new_df])
            self.imageLabelFile.index = [i for i in range(len(self.imageLabelFile.index))]
            modelName = self.selectModelComboBox.currentText()
            message += self.selectInitialImages()
            self.imageLabelFile.to_csv(os.path.join(self.modelDir, modelName, "image_labels.csv"), index=False)
        first_image = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Review")]
        if not first_image.empty:
            message += "{} images remaining".format(len(first_image.index))
            self.updateModelButton.setEnabled(False)
        else:
            first_image = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (self.imageLabelFile["Status"] == "Reviewed")]
            if not first_image.empty:
                message += "Review stage complete. Please update model for next set of images"
                self.updateModelButton.setEnabled(True)
            else:
                first_image = self.imageLabelFile.loc[(self.imageLabelFile["Folder"] == self.imageDirectory) & (
                            self.imageLabelFile["Status"] == "Complete")]
                message += "Video complete. Please select the next video to annotate"
                self.updateModelButton.setEnabled(True)

        self.messageLabel.setText(message)
        self.currentImage = first_image["FileName"][first_image.index[0]]
        self.currentBBoxes = first_image["Bounding boxes"][first_image.index[0]]
        self.updateWidget()

    def updateBBoxSelector(self):
        prev_index = self.currentBoxSelector.currentIndex()
        initial_count = self.currentBoxSelector.count()
        for i in range(self.currentBoxSelector.count() - 1, -1, -1):
            self.currentBoxSelector.removeItem(i)
        self.currentBoxSelector.addItem("Select box")
        self.currentBoxSelector.addItem("Add new box")
        bboxes = eval(str(self.currentBBoxes))
        boxNames = ["{}. {}".format(i+1,bboxes[i]["class"]) for i in range(len(bboxes))]
        if len(boxNames) > 0:
            self.currentBoxSelector.addItems(boxNames)
        if self.currentBoxSelector.count() < initial_count and self.currentBoxSelector.count()>2:
            self.currentBoxSelector.setCurrentIndex(prev_index-1)
        elif self.currentBoxSelector.count() == 2:
            self.currentBoxSelector.setCurrentIndex(0)
        else:
            self.currentBoxSelector.setCurrentIndex(prev_index)

    def updateLabelFile(self):
        entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==self.currentImage]
        if not entry.empty:
            self.imageLabelFile["Bounding boxes"][entry.index[0]] = eval(str(self.currentBBoxes))
            self.imageLabelFile.to_csv(os.path.join(self.modelDir,self.selectModelComboBox.currentText(),"image_labels.csv"),index = False)
        class_names = sorted([self.classSelector.itemText(i) for i in range(2,self.classSelector.count())])
        class_mapping = dict(zip([i for i in range(len(class_names))], class_names))
        with open(os.path.join(self.modelDir, self.selectModelComboBox.currentText(), "class_mapping.yaml"), "w") as f:
            yaml.dump(class_mapping, f)

    def selectInitialImages(self):
        best_images = Cluster().getBestImages(self.imageFiles,self.imageDirectory)
        print(len(best_images))
        msg = "{}/{} images selected for annotation\n".format(len(best_images),len(self.imageFiles))
        if not self.imageFiles[0] in best_images:
            best_images.append(self.imageFiles[0])
        predict = False
        if os.path.exists(os.path.join(self.modelDir,self.selectModelComboBox.currentText(),"train")):
            self.yolo = YOLOv8("detect")
            self.yolo.loadModel(os.path.join(self.modelDir, self.selectModelComboBox.currentText()))
            predict = True
        for fileName in best_images:
            entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"]==fileName]
            self.imageLabelFile["Status"][entry.index[0]] = "Review"
            self.getPrediction(self.imageLabelFile["FileName"][entry.index[0]])
        return msg

    def setImage(self,fileName=None):
        if fileName == None:
            img = numpy.zeros(self.imgShape)

        else:
            img = cv2.imread(os.path.join(self.imageDirectory, fileName))
            self.originalImageShape = img.shape
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.imgShape[1], self.imgShape[0]), interpolation=cv2.INTER_AREA)
            bboxes = eval(str(self.currentBBoxes))
            for i in range(len(bboxes)):
                bbox = bboxes[i]
                if i == self.currentBoxSelector.currentIndex()-2:
                    colour = (0,255,0)
                else:
                    colour = (255,0,0)
                img = cv2.rectangle(img, (int(bbox["xmin"]*self.imgShape[1]), int(bbox["ymin"]*self.imgShape[0])), (int(bbox["xmax"]*self.imgShape[1]), int(bbox["ymax"]*self.imgShape[0])), colour, 2)
                img = cv2.putText(img, "{}. {}".format(i+1, bbox["class"]),
                                    (int(bbox["xmin"]*self.imgShape[1]), int(bbox["ymin"]*self.imgShape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2,
                                    cv2.LINE_AA)

        height, width, channel = img.shape
        bytesPerLine = 3 * width
        qImage = QImage(img.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
        pixelmap = QPixmap.fromImage(qImage)
        self.imageLabel.setPixmap(pixelmap)

    def updateModel(self):
        self.markReviewedAsComplete()
        self.createTrainingCSV()
        trainData = self.imageLabelFile.loc[(self.imageLabelFile["Folder"]==self.imageDirectory) & (self.imageLabelFile["Status"] == "Complete")]
        if len(trainData.index) == len(self.imageFiles):
            trainData = self.imageLabelFile.loc[self.imageLabelFile["Status"] == "Complete"]
        if len(trainData.index)>1000:
            balance = False
        else:
            balance=True
        if os.path.exists(os.path.join(self.modelDir,self.selectModelComboBox.currentText(),"train")):
            epochs = 2
        else:
            epochs = 100

        Train_YOLOv8.train_yolo_model(os.path.join(self.modelDir,self.selectModelComboBox.currentText()),epochs,balance)
        self.yolo = YOLOv8("detect")
        self.yolo.loadModel(os.
                            path.join(self.modelDir,self.selectModelComboBox.currentText()))
        self.selectNextImages()
        imgs_to_review = self.imageLabelFile.loc[self.imageLabelFile["Status"]=="Review"]
        if not imgs_to_review.empty:
            self.currentImage = imgs_to_review["FileName"][imgs_to_review.index[0]]
            self.currentBBoxes = imgs_to_review["Bounding boxes"][imgs_to_review.index[0]]
        else:
            completed_imgs = self.imageLabelFile.loc[(self.imageLabelFile["Folder"]==self.imageDirectory) & (self.imageLabelFile["Status"] == "Complete")]
            self.currentImage = completed_imgs["FileName"][completed_imgs.index[0]]
            self.currentBBoxes = completed_imgs["Bounding boxes"][completed_imgs.index[0]]
        self.updateWidget()

    def getPrediction(self,imageFile):
        image = cv2.imread(os.path.join(self.imageDirectory, imageFile))
        preds = eval(self.yolo.predict(image))
        for bbox in preds:
            bbox["xmin"] = bbox["xmin"] / image.shape[1]
            bbox["xmax"] = bbox["xmax"] / image.shape[1]
            bbox["ymin"] = bbox["ymin"] / image.shape[0]
            bbox["ymax"] = bbox["ymax"] / image.shape[0]
        entry = self.imageLabelFile.loc[self.imageLabelFile["FileName"] == imageFile]
        self.imageLabelFile["Bounding boxes"][entry.index[0]] = preds
        self.imageLabelFile["Status"][entry.index[0]] = "Review"


    def selectNextImages(self):
        trainData = self.imageLabelFile.loc[(self.imageLabelFile["Folder"]==self.imageDirectory) & (self.imageLabelFile["Status"]=="Complete")]
        allData = self.imageLabelFile.loc[(self.imageLabelFile["Folder"]==self.imageDirectory)]
        train_imgs = [trainData["FileName"][i] for i in trainData.index]
        all_imgs = [allData["FileName"][i] for i in allData.index]

        if len(allData.index) - len(trainData.index) > max(200,len(all_imgs)*0.1):
            for i in range(len(train_imgs)):
                if i%10==0:
                    print("updated {}/{} images".format(i+1,len(train_imgs)))
                idx = all_imgs.index(train_imgs[i])
                updated_images = 0
                while updated_images < 4 and idx<len(all_imgs)-1:
                    idx +=1
                    if not all_imgs[idx] in train_imgs:
                        self.getPrediction(all_imgs[idx])
                        updated_images +=1
        else:
            remaining_images = self.imageLabelFile.loc[(self.imageLabelFile["Folder"]==self.imageDirectory) & (self.imageLabelFile["Status"]=="Incomplete")]
            for i in remaining_images.index:
                self.getPrediction(remaining_images["FileName"][i])
        self.updateModelButton.setEnabled(False)

        self.imageLabelFile.to_csv(os.path.join(self.modelDir, self.selectModelComboBox.currentText(), "image_labels.csv"),index=False)


    def markReviewedAsComplete(self):
        img_entries = self.imageLabelFile.loc[self.imageLabelFile["Folder"]==self.imageDirectory]
        for i in img_entries.index:
            if self.imageLabelFile["Status"][i] == "Reviewed":
                self.imageLabelFile["Status"][i] = "Complete"
        self.imageLabelFile.to_csv(os.path.join(self.modelDir,self.selectModelComboBox.currentText(),"image_labels.csv"),index=False)

    def createTrainingCSV(self):
        entries = self.imageLabelFile.loc[self.imageLabelFile["Status"]=="Complete"]
        trainCSV = pandas.concat([entries.copy(),entries.copy(),entries.copy()])
        trainCSV.index = [i for i in range(3*len(entries.index))]
        trainCSV["Fold"] = [0 for i in range(3*len(entries.index))]
        set_names = ["Train" for i in range(len(entries.index))] + ["Validation" for i in range(len(entries.index))] + ["Test" for i in range(len(entries.index))]
        trainCSV["Set"] = set_names
        trainCSV = self.formatBBoxesForTraining(trainCSV)
        trainCSV.to_csv(os.path.join(self.modelDir,self.selectModelComboBox.currentText(),"Train_data.csv"),index=False)

    def formatBBoxesForTraining(self,trainCSV):
        for i in trainCSV.index:
            bboxes = eval(str(trainCSV["Bounding boxes"][i]))
            for bbox in bboxes:
                bbox["xmin"] = int(bbox["xmin"]*self.originalImageShape[1])
                bbox["xmax"] = int(bbox["xmax"]*self.originalImageShape[1])
                bbox["ymin"] = int(bbox["ymin"] * self.originalImageShape[0])
                bbox["ymax"] = int(bbox["ymax"] * self.originalImageShape[0])
            trainCSV["Bounding boxes"][i] = bboxes
        return trainCSV


if __name__ == "__main__":
    app = QApplication([])
    anReviewer = Automated_Annotator()
    sys.exit(app.exec())