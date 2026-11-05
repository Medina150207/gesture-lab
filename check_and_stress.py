from pathlib import Path
import json
import pandas as pd,numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter
from sklearn.metrics import accuracy_score,f1_score
from core import *
ROOT=Path(__file__).parent;DATA=ROOT.parent/'Gesture_A2_Data';torch.set_num_threads(2)
rows=pd.read_csv(DATA/'test.csv');records=[]
for name in ['resnet18','mobilenet_v2']:
 net,ck=load(ROOT/'results'/f'{name}.pt')
 for condition in ['original','dark_50pct','blur_radius2']:
  preds=[];actual=[]
  for r in rows.itertuples():
   with Image.open(DATA/r.path) as im:im=prepare(im)
   if condition=='dark_50pct':im=ImageEnhance.Brightness(im).enhance(.5)
   elif condition=='blur_radius2':im=im.filter(ImageFilter.GaussianBlur(2))
   with torch.no_grad():preds.append(net(tensor(im).unsqueeze(0)).argmax(1).item())
   actual.append(CLASSES.index(r._asdict()['class']) if 'class' in r._asdict() else 0)
  actual=[CLASSES.index(c) for c in rows['class']]
  records.append(dict(model=name,condition=condition,accuracy=accuracy_score(actual,preds),macro_f1=f1_score(actual,preds,average='macro'),n=len(actual)))
  if condition=='original':
   old=pd.read_csv(ROOT/'results'/f'{name}_test_predictions.csv');assert [CLASSES[i] for i in preds]==old.predicted.tolist()
 with Image.open(DATA/rows.iloc[0].path) as im:
  probs,ms,overlay,cam,cam_ms=predict(net,im)
 assert np.isfinite(probs).all() and abs(probs.sum()-1)<1e-5 and np.isfinite(cam).all() and cam.shape==(160,160)
 overlay.save(ROOT/'results'/f'{name}_gradcam.png')
 print(name,'checkpoint + predictions + Grad-CAM verified',flush=True)
pd.DataFrame(records).to_csv(ROOT/'results'/'stress_tests.csv',index=False)
print(records,flush=True)
