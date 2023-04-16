import torch
from torch import nn
import torch.nn.functional as F
import torch
from PIL import Image
import numpy as np
import base64 
import io 
from skimage.transform import resize

# load a pre-trained Model and convert it to eval mode. 
class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 3 channels - RGB
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 6, stride = 1, padding = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.LazyConv2d(32, kernel_size = 3, stride = 1, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.LazyConv2d(64, kernel_size = 3, stride = 1, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv4 = nn.LazyConv2d(16, kernel_size = 2, stride = 1, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv5 = nn.LazyConv2d(8, kernel_size = 1, stride = 1, padding = 1)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.fc1 = nn.LazyLinear(100)
        self.fc2 = nn.LazyLinear(10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x

model = CNN()
model.load_state_dict(torch.load('./best_model.pt', map_location=torch.device('cpu')))

def base64str_to_PILImage(base64str):
    base64_img_bytes = base64str.encode('utf-8')
    base64bytes = base64.b64decode(base64_img_bytes)
    bytesObj = io.BytesIO(base64bytes)
    img = Image.open(bytesObj) 
    return img

def change_color(box, color, colors):
    n,m,k = box.shape
    for i in range(n):
        for j in range(m):
            for lit in colors:
                if not (lit == color).all():
                    if (box[i,j] == lit).all():
                        box[i,j] = np.array([255,255,255])

    return box

def get_most_common_colors(im):
    colors, counts = np.unique(im.reshape((im.shape[0]*im.shape[1]), im.shape[2]), axis = 0, return_counts = True)
    color_to_counts = list(zip(colors,counts))
    ind = np.argsort(-counts)

    ordered_colors = colors[ind]

    out = ["2m", "kona"]

    for color in ordered_colors:
        if not (color[0] > 240 and color[1] > 240 and color[2] > 240):
            out.append(color)

    return np.array(out[2:7])

def get_pre_bound(im,color):
    x = []
    y = []
    for i in range(im.shape[0]):
        colors = np.unique(im[i,:,:], axis=0)
        if color in colors:
            for j in range(im.shape[1]):
                wont = True
                for k in range(3):
                    if im[i,j,k] != color[k]:
                        wont = False
                if wont:
                    x.append(i)
                    break

    for i in range(im.shape[1]):
        colors = np.unique(im[:,i,:], axis=0)
        if color in colors:
            for j in range(im.shape[0]):
                vont = True
                for k in range(3):
                    if im[j,i,k] != color[k]:
                        vont = False
                if vont:
                    y.append(i)
                    break

    return x,y

def get_square_size(x,y):
    return max(max(x)-min(x),max(y)-min(y))

def get_bounding_boxes_from_img(im):
    boxes = dict()
    square_sizes = []
    colors = get_most_common_colors(im)
    final_boxes = dict()
    box_add = 5
    color_dict = dict()
    for i,color in enumerate(colors):
        x,y = get_pre_bound(im,color)
        im_x,im_y = im.shape[:2]
        size_x_min = max(0,min(x)-box_add)
        size_x_max = min(im_x,max(x)+box_add)
        new_x = [size_x_min,size_x_max]
        size_y_min = max(0,min(y)-box_add)
        size_y_max = min(im_y-1,max(y)+box_add)
        new_y = [size_y_min,size_y_max]
        org_box = im[size_x_min:size_x_max, size_y_min:size_y_max]
        square_sizes.append(get_square_size(new_x,new_y)) 
        color_dict[min(y)] = color
        boxes[min(y)] = org_box
        square_size = max(square_sizes)

    for i,index in enumerate(sorted(boxes.keys())):
        box = 255 * np.ones((square_size,square_size,3))
        box = box.astype(np.int64)
        n,m,k = boxes[index].shape
        color = color_dict[index]
        box[:n,:m] = change_color(boxes[index].astype(np.int64),color,colors)
        final_boxes[i] = box

    return final_boxes


def predict(base64str):
    pil_img = base64str_to_PILImage(base64str)
    img = np.array(pil_img, dtype=np.float32)

    bounding_boxes = get_bounding_boxes_from_img(img)

    image_list = []

    for i in range(5):
        resized_image = np.array(resize(bounding_boxes[i].astype(np.uint8), (64, 64)), dtype=np.float32)
        image_list.append(torch.tensor(resized_image).permute(2,0,1).unsqueeze(0))

    input = torch.cat(image_list)

    output = model(input.to("cpu"))

    preds = torch.argmax(output, dim=1)

    probs = torch.exp(output).data.max(1, keepdim=True)[0]

    return "".join(np.array(preds, dtype=str)), torch.exp(output)
