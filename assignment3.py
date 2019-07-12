import matplotlib.patches as patches
from PIL import Image, ImageDraw
from cv2 import VideoWriter, VideoWriter_fourcc
import cv2
import os
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image, ImageOps
import tensorflow as tf
from torchvision import datasets, models, transforms
from torch.autograd import Variable
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
# The structure of the basic fully connected neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(1024, 1024)
        
        self.relu1 = nn.ReLU(inplace = True)
        
        self.fc2 = nn.Linear(1024, 1024)
        
        self.relu2 = nn.ReLU(inplace = True)
        
        
        self.fc8 = nn.Linear(1024, 4)
        
    def forward(self, x):
      x = self.fc1(x)
      
      x = self.relu1(x)
      
      x = self.fc2(x)
     
      x = self.relu2(x)
     
      x = self.fc8(x)
      return x

# the custom loader to take the features and the boxes as batches
# takes the array of features and the array of boxes as parameter
class GetFeatures(Dataset):

    def __init__(self, df_feat, df_box):
        """
        Args:
            df_feat (DataFrame): array of the features
            df_box (DataFrame): array of the bounding boxes
        """
        self.df_feat = df_feat
        self.df_box = df_box
    
    def __len__(self):
        return len(self.df_feat)

    def __getitem__(self, idx):
#         print(idx)
        feature = self.df_feat[idx].rstrip("\n").split()
        feature = list(np.float_(feature))
        
        box =  self.df_box[idx].rstrip("\n").split()
        box = list(np.float_(box))                
        
        feature = Variable(torch.tensor(feature))
        box = Variable(torch.tensor(box))
#         print(feature)
        return feature, box
# The custom loader to take the images as pairs.
# to call custom loader that takes images as pairs give the csv files as dataframe, the path of the folder that has the images, transform value and a boolean value if this is runned on Google Colab or not.
class GetImages(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, transform=None, isColab = False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.isColab = isColab
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if(self.isColab): 
          img_name1 = "drive/My Drive/"+self.df.iloc[idx, 0] #ilk frame path
          img_name2 = "drive/My Drive/"+self.df.iloc[idx, 1] #ikinci frame path
        else:
          img_name1 = self.df.iloc[idx, 0] #ilk frame path
          img_name2 = self.df.iloc[idx, 1] #ikinci frame path
        
        class_name = get_class(img_name1)
       
        
        image1 = Image.open(img_name1)
        image2 = Image.open(img_name2)

        landmarks1 = self.df.iloc[idx, 2:6].as_matrix()
        landmarks2 = self.df.iloc[idx, 6:].as_matrix()
        enl, rel_box = crop_im(image1, landmarks1, landmarks2)
        
       
        image1 = image1.crop(enl)
        image2 = image2.crop(enl)
        
        
        
        img_tensor1 = data_transforms(image1).float()
        img_tensor2 = data_transforms(image2).float()
        
        img_tensor1 = img_tensor1.squeeze()
        img_tensor2 = img_tensor2.squeeze()
        
        img1 = Variable(img_tensor1)
        img2 = Variable(img_tensor2)
        
        
        
       
       
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        sample = [img1, img2]
        return sample, class_name, np.array(rel_box)
      
# return the coordinates of the frames as dictionary
# Takes 2 parameters which are path of the folder and a boolean value if the code is running on Google Colab or not
def ann_file(path, isColab = True):
  if(isColab):
    path = "drive/My Drive/" + path
  filenames = os.listdir(path)
  
  ann_dict = {}
  for file in filenames:
    label = file.split(".")[0]
    ann_file = open(path + file, "r")
    ann = ann_file.read().split("\n")
    
    sub = []
    for i in ann:
      if(i != ""):
        params = i.split()
        sub.append(params)
        
    ann_dict[label] = sub
  return ann_dict

# creates csv extension files which contains the frame1_path, frame2_path, bounding_box_of_frame1 and bounding_box_of_frame2 respectively in each line
# Takes path of the videos folder, dictionary of the annotations that is returned from ann_file function, and 3 boolean values respectively
# make isTrain true if want to take train images
# make isTest true and isTrain false if want to take test images
# make isTest false and isTrain false if want to take validation images
# make isColab true if the code running on Google Colab otherwise false
def csv_files(path, ann, isTrain=True, isTest=False, isColab = True):
  if(isColab):
    path_detail = "drive/My Drive/" + path
  else:
    path_detail = path
  if(isTrain):
    path_detail = path_detail + "train"
    path = path + "train"
    csv_file = open("train.csv", "w")
  elif(isTest):
    path_detail = path_detail + "test"
    path = path + "test"
    csv_file = open("test.csv", "w")
  else: 
    path = path + "val"
    path_detail = path_detail + "val"
    csv_file = open("val.csv", "w")
  csv_file.write("f1, f2, x1_1, y1_1, x2_1, y2_1, x1_2, y1_2, x2_2, y2_2\n")
  filenames = os.listdir(path_detail)
  filenames.sort()
  for file in filenames:
    label = file # video name
    ann_infos = ann[label]
    frames = os.listdir(path_detail + "/" + label)
    frames.sort()
    path_label = path + "/" + label
    
    arr_len = len(frames)
    for fr in range(arr_len - 1):
      line = ""
      line += path_label + "/" + frames[fr]+", "
      line += path_label + "/" + frames[fr + 1]+", "
      coords = ann[label]
      
      for coor in coords:
        if(coor[0] == (frames[fr].split('.')[0]).lstrip("0")):
          line += str(coor[1]) + ", " + str(coor[2]) + ", " + str(coor[3]) + ", " + str(coor[4])+ ", " 
        if(coor[0] == (frames[fr+1].split('.')[0]).lstrip("0")):
          line += str(coor[1]) + ", " + str(coor[2]) + ", " + str(coor[3]) + ", " + str(coor[4])
      csv_file.write(line)
      csv_file.write("\n")
      
  csv_file.close()
# makes the bounding box 2 times large and return the coordinates of the enlarged box as an array
# takes the image and the ground truth bounding box as parameter
def enlarged(im, box):
  width, height = im.size 
  # box width-height 
  w = box[2] - box[0]
  h = box[3] - box[1]
  
  half_w = w/2
  half_h = h/2
  
  x1 = max(box[0] - half_w, 0)
  y1 = max(box[1] - half_h, 0)
  x2 = min(box[2] + half_w, width)
  y2 = min(box[3] + half_h, height)
  
  return [x1, y1, x2, y2]
# finds the relative position of the bounding box and returns its coordinates as an array
# takes the coordinates of the enlarged box and the ground truth bounding box of the second frame
def relative(enlarge, box):
  x1 = max(box[0] - enlarge[0], 0)
  y1 = max(box[1] - enlarge[1], 0)
  x2 = min(x1 + (box[2]- box[0]), enlarge[2]-enlarge[0])#max(enlarge[2] - box[2], 0)
  y2 = min(y1 + (box[3]- box[1]),enlarge[3]-enlarge[1])#max(enlarge[3] - box[3], 0)
  
  return [x1, y1, x2, y2]
# reorganize the coordinates of the bounding box according to the scaling and return  its coordinates as an array
# takes coordinates of the enlarge box and the coordinates of the relative box
def scaled(enl, box):
  w = enl[2] - enl[0]
  h = enl[3] - enl[1]
  
  scale_w = 224 / w
  scale_h = 224 / h
  
  x1 = box[0] * scale_w
  y1 = box[1] * scale_h
  x2 = box[2] * scale_w
  y2 = box[3] * scale_h
  
  return [x1, y1, x2, y2]
# returns the coordnates of the enlarged and relative boxes. 
# takes image, ground truth bounding box of the first frame and ground truth bounding box of the second frame as parameter
def crop_im(im, box1, box2):
  w,h = im.size
  
  enlarge = enlarged(im, box1)
  relative_box2 = relative(enlarge, box2)
#   s_enlarge = scaled(w, h, enlarge)
  s_box = scaled(enlarge, relative_box2)
  
  return enlarge, s_box

# takes the name of the video from the given path and return it
# takes the path of the image as parameter
def get_class(img_path):
  path = img_path.split("/")
  return path[-2]  

# extracts the features from the model and return the combination of the feaatures and the boxes
# takes model, dataloader and a boolean value if it is a train dataset or not
def feature_extract(model, dataloader, isTrain):
  comb = []
  box = []
  for inputs, labels, area in dataloader:
    input1 = inputs[0].to(device)
    input2 = inputs[1].to(device)
    
    output1 = model.features(input1)
    output2 = model.features(input2)
    
    output1 = model.avgpool(output1)
    output1 = output1.view(output1.size(0), -1)
    
    output2 = model.avgpool(output2)
    output2 = output2.view(output2.size(0), -1)
    
    combine_tensors = torch.cat((output1, output2), 1)
    
    combined = np.array(combine_tensors.cpu())
    area = np.array(area.cpu())
    comb.extend(combined)
    box.extend(area)
    
#     np.savetxt(feature, combined, fmt='%10.5f')
#     np.savetxt(out, area, fmt='%10.5f')
    
  return comb, box
  
# Trains the fully connected neural network and returns the network, loss of the validation and loss of the train
# Takes the model, the device(gpu or not), train dataloader, validation dataloader, optimizer, criterion and the number of epochs as parameter
def train(net, device, tr_dataloader, val_dataloader, optimizer, criterion, epoch_num):
  
  loss_tr = []
  loss_val = []
  
  
  
  for epoch in range(epoch_num):
    print('\nEpoch {}/{}'.format(epoch, epoch_num - 1))
    print('-' * 10)
    running_tr_loss = 0
    running_val_loss = 0
    # train network
    net.train()
    phase = "train"
    for inputs, output in tr_dataloader:
      phase = "train"
      inputs = inputs.to(device)
      output = output.to(device)
      
      optimizer.zero_grad()
      with torch.set_grad_enabled(phase == 'train'):

        net_out = net(inputs)

        loss = criterion(net_out, output)

        if phase == 'train':
          loss.backward() # just for train

          optimizer.step() # just for train

      running_tr_loss += loss.item() 
      
    epoch_loss = running_tr_loss / len(tr_dataloader.dataset)
    
    loss_tr.append(epoch_loss)
    
    print('{} Loss: {:.4f} '.format("Train", epoch_loss))  
    
   
    
    # validation
    net.eval()
    for inputs, output in val_dataloader:
      phase = "val"
      inputs = inputs.to(device)
      output = output.to(device)
      
      optimizer.zero_grad()
      with torch.no_grad():
        net_out = net(inputs)

        loss = criterion(net_out, output)

      running_val_loss += loss.item() 
   
    epoch_loss = running_val_loss / len(val_dataloader.dataset)
    
    loss_val.append(epoch_loss)
    
    print('{} Loss: {:.4f} '.format("Validation", epoch_loss))
    
  return net, loss_tr, loss_val      

# draw the plot of losses annd save them
# takes array of validation losses, array of train losses and number of epochs as parameter
def plot(val_loss, tr_loss, num_epochs):
  plt.subplot(211)
  plt.title("Loss plots vs. Number of Training Epochs")
  plt.plot(range(1,num_epochs+1),val_loss,label="validation")
  plt.plot(range(1,num_epochs+1),tr_loss,label="train")
  
  plt.xticks(np.arange(1, num_epochs+1, 1.0))
  plt.legend()
  
 
  plt.tight_layout()
  plt.savefig("LossPlot.png")

# prepare the given path according to running environment
# if it is run in colab, make the isColab = True (default = True)
def colab_path(path, isColab = True):
  if(isColab):
    return "drive/My Drive/"+path
  else:
    return path


​
# take test dataframe and the network 
# return the average loss
# test the network
def test(csv_test, net):
  num_test = len(csv_test)
  count = 0
  video_name = ""
  loss = 0.0
  net.eval()
  for tst in range(num_test - 1):
    im1_path = csv_test.iloc[tst, 0]
    real_box1 = csv_test.iloc[tst, 2:6].as_matrix()
    
    im2_path = csv_test.iloc[tst, 1]
    real_box2 = csv_test.iloc[tst, 6:].as_matrix()
​
    class_Name = im1_path.split("/")[-2] # name of the video
#     print(class_Name)
    # read frames
    im1 = Image.open(colab_path(im1_path, True))
    im2 = Image.open(colab_path(im2_path, True))
    
    width, height = im1.size
    
    if(video_name != class_Name):
      video_name = class_Name
      count += 1
      #initialize the first frame with real bounding box
      if(count <= 3):
        if(count>1):
          gif.release()
          
        FPS = 24
        fourcc = VideoWriter_fourcc(*'MP42')
        gif = cv2.VideoWriter(class_Name+".avi",  fourcc, float(FPS),(width,height)) #create wideo
        land = (real_box1[0], real_box1[1], real_box1[2], real_box1[3])
        rimg = im1.copy()
        rimg_draw = ImageDraw.Draw(rimg)
        rimg_draw.rectangle(land, fill=None, outline=(255, 0, 0))
        im_cv = np.array(rimg)
        gif.write(im_cv)
        
        
        
      enlarge, rel_box = crop_im(im1, real_box1, real_box2) # compare with rel_box
      
    else:
      
      if(count <= 3):
        rimg = im1.copy()
        rimg_draw = ImageDraw.Draw(rimg)
        rimg_draw.rectangle(last_box, fill=None, outline=(255, 0, 0))
        im_cv = np.array(rimg)
        gif.write(im_cv)
      
     
      enlarge, rel_box = crop_im(im1, next_box, real_box2) # compare with rel_box
       
    im1 = im1.crop(enlarge)
    img1 = in_transform(im1)[:3,:,:].unsqueeze(0)
    im1 = img1.to(device)
    feat1 = model.features(im1)
    feat1 = model.avgpool(feat1)
    feat1 = feat1.view(feat1.size(0), -1)
​
​
    im2 = im2.crop(enlarge)
    img2 = in_transform(im2)[:3,:,:].unsqueeze(0)
    im2 = img2.to(device)
    feat2 = model.features(im2)
    feat2 = model.avgpool(feat2)
    feat2 = feat2.view(feat2.size(0), -1)
​
​
    combined_feats = torch.cat((feat1, feat2), 1) # concatanete the features
​
    # send the network
    box = Variable(torch.from_numpy(np.array(rel_box).astype(np.float32)))
    
     
    box = box.to(device)
    net_out = net(combined_feats)
#     print(np.array(net_out.cpu().detach().numpy()).dtype)
    loss += criterion(net_out, box)
​
    # convert the tensor to the array
    next_box = np.array(net_out.squeeze().cpu().detach().numpy())
  
  return loss / num_test
​


path_ann = "dataset/annotations/"
ann = ann_file(path_ann, isColab = True)    

path = "dataset/videos/"
csv_files(path,ann, isTrain = False, isTest = False, isColab = True)
csv_files(path,ann, isTrain = True, isTest = False, isColab = True)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

csv_train = pd.read_csv("train.csv", sep=", ")   
my_dataset_train = GetImages(csv_train, root_dir='drive/My Drive/dataset/videos/train',transform = normalize, isColab = True)
train_loader = torch.utils.data.DataLoader(my_dataset_train, batch_size=64, shuffle=True, num_workers=4)
​
csv_val = pd.read_csv("val.csv", sep=", ")   
my_dataset_val = GetImages(csv_val, root_dir='drive/My Drive/dataset/videos/val',transform = normalize, isColab = True)
val_loader = torch.utils.data.DataLoader(my_dataset_val, batch_size=64, shuffle=True, num_workers=4)
​

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # check if cuda is available
# create vgg model to extract features
model = models.vgg16(pretrained=True)
model.avgpool = nn.AvgPool2d((7,7)) # change the avgpool 

for param in model.parameters():
    param.requires_grad = False

model = model.to(device) # send the model to gpu

# extract features
comb_tr, box_tr = feature_extract(model, train_loader, isTrain=True)
comb_val, box_val = feature_extract(model, val_loader, isTrain=False)

# save the features and the boxes to the txt files
feature_val = open("val_feature.txt", "w")
out_val = open("val_out.txt", "w")

feature_tr = open("train_feature.txt", "w")
out_tr = open("train_out.txt", "w")


np.savetxt(feature_tr, comb_tr, fmt='%10.5f')
np.savetxt(out_tr, box_tr, fmt='%10.5f')


np.savetxt(feature_val, comb_val, fmt='%10.5f')
np.savetxt(out_val, box_val, fmt='%10.5f')


feature_tr.close()
out_tr.close()

feature_val.close()
out_val.close()

# reaad the files
file_features_tr = open("train_feature.txt", "r")
file_boxes_tr = open("train_out.txt", "r")
file_features_val = open("val_feature.txt", "r")
file_boxes_val = open("val_out.txt", "r")

# takes them as 2d arrays
features_tr = file_features_tr.readlines()
boxes_tr = file_boxes_tr.readlines()
features_val = file_features_val.readlines()
boxes_val = file_boxes_val.readlines()

# call custom loader and take them as batch_size
feats_tr = GetFeatures(features_tr, boxes_tr)
tr_dataloader = torch.utils.data.DataLoader(feats_tr, batch_size=16, shuffle=False, num_workers=4)

feats_val = GetFeatures(features_val, boxes_val)
val_dataloader = torch.utils.data.DataLoader(feats_val, batch_size=16, shuffle=False, num_workers=4)

# construct fully connected network
net = Network()

criterion = torch.nn.MSELoss(size_average = True)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)
net = net.to(device) # send the network to device

# train the network
net, loss_tr, loss_val = train(net, device, tr_dataloader, val_dataloader, optimizer, criterion, 20)

plot(loss_val, loss_tr, 20) # draw the plots 

# test the network
in_transform = transforms.Compose([
  transforms.Resize(224),
  transforms.CenterCrop(224),
  transforms.ToTensor(),
  transforms.Normalize((0.485, 0.456, 0.406),
  (0.229, 0.224, 0.225))])

csv_files(path,ann, isTrain = False, isTest = True, isColab = True)
csv_test = pd.read_csv("test.csv", sep=", ")  

test(csv_test, net)

































