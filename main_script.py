# Load required packages
import os
import torch # for working with NN
import cv2 # for working with images and NN
import pytesseract
#from torch.autograd import Variable
from PIL import Image
#from PIL import ImageEnhance
from PIL import ImageFilter
#import matplotlib.pyplot as plt # tenp
#import matplotlib.image as mpimg # temp
import numpy as np
import pandas as pd
import re

# Load own libraries
import imgproc
from craft import CRAFT
import craft_utils
import file_utils
import textproc
import progressbar

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++ Set global variables +++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Set work directory
wd='D:\\OCR\\)'
os.chdir(wd)
# Set path to tessaract
td='C:\\Users\\Admin\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
# Set the maximum size of image that will be used (if it is larger the image will be resized to square_size)
square_size=350 # the larger size the more memory we need
mag_ratio=1 # Image magnification ratio
text_threshold=0.05 # Text confidence threshold
link_threshold=0.05 # Link confidence threshold
low_text=0.05 # Text low-bound score
resize_factor_V=np.linspace(2,4,3)  # resize factors
rotate_factor_V=np.linspace(-8,8,17) # roate factors

print("The main parameter values are:")
print("Image size:",square_size)
print("Image magnification ratio",mag_ratio)
print("Text confidence threshold",text_threshold)
print("Link confidence threshold",link_threshold)
print("Text low-bound score:",low_text)
print("Resize factors:",resize_factor_V)
print("Rotate factors:",rotate_factor_V)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Loading neural network for finding text areas
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

net = CRAFT() # Initialize net
net.load_state_dict(craft_utils.copyStateDict(torch.load(os.getcwd()+'\\craft_mlt_25k.pth', map_location='cpu'))) # load pretrained weights
net.eval()

print("The required neural network has been successfully loaded...")

# Set tessaract path
pytesseract.pytesseract.tesseract_cmd = td

# Load paths for all images in Input folder
image_list, _, _ = file_utils.get_files(wd+'\\Input')

print("All file names has been obtained...")

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++ Function section +++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Find coordinates of boxes where is a text
def get_boxes(img_c):
    # Resize
    img_r, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(img_c, 
                                                                square_size=square_size, 
                                                                interpolation=cv2.INTER_LINEAR, 
                                                                mag_ratio=mag_ratio)
    # Save ratio index for height
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing of the image
    x = imgproc.normalizeMeanVariance(img_r)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)                # [c, h, w] to [b, c, h, w]
    # forward pass
    y, _ = net(x)
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()
    # Post-processing
    boxes, _ = craft_utils.getDetBoxes(score_text, score_link, text_threshold=text_threshold, 
                                   link_threshold=link_threshold, low_text=low_text, poly=False)
    # Coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    return boxes


# Take founded boxes of texts and recognize the text with set parameter values
def get_text(img_c,boxes_c, resize_factor=4, rotate_factor=0, sep=""):
    img_c=Image.fromarray(img_c)
    text_line=''
    #text_line=list()
    for j in range(len(boxes_c)):
        l = int(min(boxes_c[j][:,0])) # left side of box
        r = int(max(boxes_c[j][:,0])) # right side of box
        t = int(min(boxes_c[j][:,1])) # top side of box
        b = int(max(boxes_c[j][:,1])) # bottom side of box
        if l==r==t==b:
            continue
        # Crop source image to the current text field
        text_img = img_c.crop((l,t,r,b))
        text_img = text_img.convert('L')
        #plt.imshow(text_img)
        text_img = text_img.resize((int(text_img.size[0]*resize_factor),
                                    int(text_img.size[1]*resize_factor)),
                                   Image.BICUBIC)
        text_img = text_img.filter(ImageFilter.GaussianBlur(1))
        text_img = text_img.rotate(rotate_factor)
        #text_img = ImageEnhance.Contrast(text_img).enhance(1)
        #text_img = ImageEnhance.Brightness(text_img).enhance(1)
        # for debugging
        #plt.imshow(text_img)
        #plt.show()
        #text recognition
        text = pytesseract.image_to_string(text_img,lang = 'rus')
        #text_line=text_line+sep+text.replace(" ","")
        #print(text)
        # clear text
        #text = text.upper()
        #text = "".join(re.split("[^А-Я0-9-.]*",text)) 
        # save text
        if text!='':
            #text_line.append(text)
            #text_line=text_line+text
            text_line=text_line+sep+text.replace(" ","")
        #print(j)
    text_line = text_line.upper()
    text_line = "".join(re.split("[^А-Я0-9-. ]*",text_line)) 
    return(text_line)
    
# Parsing text in different scales and rotations
def parse_text_iter2(img0_c,img90_c,boxes0_c, boxes90_c, 
                    resize_factor_V=[4], rotate_factor_V=[0], pb=True):
    l=len(resize_factor_V)*len(rotate_factor_V)
    res=list()
    bar = progressbar.ProgressBar(max_value=l)
    k=1
    for i in resize_factor_V:
        for j in rotate_factor_V:
            text0_c=get_text(img0_c,boxes0_c,i,j)
            text90_c=get_text(img90_c,boxes90_c,i,j)
            p_text=textproc.parse_text2(text0_c,text90_c)
            #res.append(list(p_text.items()))            
            res.append(p_text)
            #print(i,j)
            #print(text0_c)
            #print(p_text.transpose())            
            if pb==True:
                bar.update(k)
            k=k+1
    res=pd.concat(res,ignore_index=True)
    return res

## Final post processing for one image
def get_final_data(df):
    stat=np.array([])
    for i in range(df.shape[0]):
        stat=np.append(stat,np.sum(df.iloc[i,:]!=''))
    s_df=df.iloc[list(stat==np.max(stat)),:] 
    res=list()
    for i in range(s_df.shape[1]):
        x=s_df.iloc[:,i]
        x=x[x!='']
        if len(x)>0:
            x=x.value_counts().index[0] # take the most frequent value
        else:
            x=''
        res.append(x)
    res=pd.DataFrame(res).transpose()
    res.columns=df.columns
    for i in range(res.shape[1]):
        if res.iloc[0,i]=='':
           res.iloc[0,i]=df.iloc[:,i].value_counts().index[0]            
    return res

 
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++ Main cycle ++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Only for debugging
"""
i=0
img0 = imgproc.loadImage(image_list[i])
img90 = np.rot90(img0)  
boxes0 = get_boxes(img0)
boxes90 = get_boxes(img90)
text0=get_text(img0,boxes0,resize_factor_V[0],rotate_factor_V[1])
text90=get_text(img90,boxes90,resize_factor_V[0],rotate_factor_V[2], sep=" ")
#textproc.get_patronymic_sex2(text0) #66
#textproc.get_city2(text0,66) # 83
#textproc.get_name2(text0, "МУЖ",66)# 58
#textproc.get_surname2(text0, 58) #
#textproc.get_dates2(text0) #
#textproc.get_code_place2(text0)
#textproc.get_number2(text90) # 7
#textproc.get_series2(text90, 7)
p_text=textproc.parse_text2(text0,text90)
p_text.transpose()
text=parse_text_iter2(img0,img90,boxes0,boxes90,resize_factor_V,rotate_factor_V,True)    
#text
#text.to_csv("text.csv")
final_res=get_final_data(text)
final_res.to_csv("text.csv")
"""

final_table=list() # final table

for i in range(len(image_list)):
    print("Load image:",image_list[i])
    #  Load the current image as numpy array
    img0 = imgproc.loadImage(image_list[i])
    img90 = np.rot90(img0)
    print("The image has been loaded...")
    # Get boxes of text
    print("Find text boxes...")
    boxes0 = get_boxes(img0)
    boxes90 = get_boxes(img90)
    # Get text
    print("Recognize boxes in different variants...")
    text=parse_text_iter2(img0,img90,boxes0,boxes90,resize_factor_V,rotate_factor_V,True)
    print("")
    #all_res.to_csv(wd+'\\all_results.csv',index = None, header=True)
    # Get all data
    final_res=get_final_data(text)
    # Print
    #print(final_res.transpose())
    # Add to final table
    final_table.append(final_res)
    print("The work has been done for this image!")
    print("")

# Save as csv and json formats in files
indV=list()
for i in range(len(image_list)):
    indV.append(os.path.basename(image_list[i]))

final_table=pd.concat(final_table,ignore_index=True)
final_table.index=indV
final_table.to_csv(wd+'\\output\\all_results.csv')
final_table.to_json(wd+'\\output\\all_results.json')
print("The work has been finished!")