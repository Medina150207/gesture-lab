from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms
CLASSES=['peace','stop','thumbs_up']
MEAN=[.485,.456,.406];STD=[.229,.224,.225];SIZE=160

def build(name, pretrained=False):
    if name=='resnet18':
        net=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        net.fc=nn.Linear(net.fc.in_features,3)
    elif name=='mobilenet_v2':
        net=models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        net.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(net.last_channel,3))
    else: raise ValueError('Unknown architecture')
    return net

def prepare(image):
    return ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'),(SIZE,SIZE),color=(128,128,128))
def tensor(image):
    return transforms.Normalize(MEAN,STD)(transforms.ToTensor()(prepare(image)))
def head(net,name):return net.fc if name=='resnet18' else net.classifier
def features(net,name,x):
    if name=='resnet18':
        x=net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x=net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return net.avgpool(x).flatten(1)
    return nn.functional.adaptive_avg_pool2d(net.features(x),1).flatten(1)
def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['classes']!=CLASSES:raise ValueError('Class order mismatch')
    net=build(ck['architecture']);net.load_state_dict(ck['state_dict']);net.eval()
    return net,ck

def predict(net,image,explain=True):
    x=tensor(image).unsqueeze(0)
    with torch.no_grad():net(x) # warm-up excluded from timing
    start=time.perf_counter()
    with torch.no_grad():probs=net(x).softmax(1)[0].numpy()
    elapsed=(time.perf_counter()-start)*1000
    overlay=None;cam=None;cam_ms=None
    if explain:
        start=time.perf_counter();saved={}
        target=net.layer4 if hasattr(net,'layer4') else net.features[-1]
        def capture(module,inputs,output):saved['activation']=output
        handle=target.register_forward_hook(capture)
        try:
            with torch.enable_grad():
                logits=net(x.requires_grad_(True));a=saved['activation']
                g=torch.autograd.grad(logits[0,int(probs.argmax())],a)[0]
                cam=(a*g.mean((2,3),keepdim=True)).sum(1).relu()[0].detach().numpy()
            if cam.max()>0:cam/=cam.max()
            cam=np.array(Image.fromarray(cam).resize((SIZE,SIZE),Image.Resampling.BILINEAR))
            from matplotlib import colormaps
            heat=colormaps['jet'](cam)[:,:,:3]
            overlay=Image.fromarray(np.uint8(np.clip(.6*np.asarray(prepare(image))+.4*255*heat,0,255)))
            cam_ms=(time.perf_counter()-start)*1000
        finally:handle.remove()
    return probs,elapsed,overlay,cam,cam_ms
