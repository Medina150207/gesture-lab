from pathlib import Path
import pandas as pd,numpy as np,torch
from PIL import Image,ImageEnhance,ImageFilter
from sklearn.metrics import accuracy_score,f1_score
from core import *
r=Path(__file__).parent;data=r.parent/'Gesture_A2_Data';torch.set_num_threads(2)
a=pd.read_csv(r/'control_results/comparison.csv');b=pd.read_csv(r/'synthetic_results/comparison.csv');a['training']='real_only';b['training']='real_plus_36_synthetic';table=pd.concat([a,b]);table.to_csv(r/'ablation.csv',index=False)
records=[];test=pd.read_csv(data/'test.csv')
for variant in ['control_results','synthetic_results']:
 net,ck=load(r/variant/'mobilenet_v2.pt')
 for condition in ['original','dark_50pct','blur_radius2']:
  pred=[]
  for row in test.to_dict('records'):
   with Image.open(data/row['path']) as im:im=prepare(im)
   if condition=='dark_50pct':im=ImageEnhance.Brightness(im).enhance(.5)
   if condition=='blur_radius2':im=im.filter(ImageFilter.GaussianBlur(2))
   with torch.no_grad():pred.append(int(net(tensor(im).unsqueeze(0)).argmax(1)))
  y=[CLASSES.index(c) for c in test['class']];records.append(dict(variant=variant,condition=condition,accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,average='macro')))
 with Image.open(data/test.iloc[0].path) as im:
  _,_,overlay,cam,_=predict(net,im);assert np.isfinite(cam).all();overlay.save(r/variant/'gradcam.png')
stress=pd.DataFrame(records);stress.to_csv(r/'ablation_stress.csv',index=False)
text='''<html><meta charset="utf-8"><style>body{max-width:950px;margin:40px auto;font:17px/1.6 Arial}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse;font-size:14px}</style><h1>Synthetic-frame ablation — Medina and Aiganym</h1><p>We received 1026 images: 171 original frames and 855 derived augmentations. Seven source sequences were inferred from filenames. Transition frames with an absent target gesture were excluded. We selected 36 evenly spaced eligible original frames, 12 per class, before training. Derived augmentation copies were not used. Source generator, prompts and seeds were not provided; attribution to Step-Video-T2V or ArtAug is unverified.</p><p>Both branches use the same MobileNetV2 ImageNet initialization, seed 42, frozen backbone, input 160, 40 head epochs and minimum validation loss selection. Training has 535 real images versus 535 real plus 36 synthetic (6.3% synthetic before augmentation). Each receives an original and one augmented training view. Validation (114) and test (112) remain real and unchanged. Fixed epochs imply slightly more optimizer steps in the larger training branch. A control run was reproduced in the current environment.</p>'''+table.round(4).to_html(index=False)+'<h2>Controlled image shifts</h2>'+stress.round(4).to_html(index=False)+'''<p>Darkening and blur are artificial image shifts, not real environment labels. They are not ArtAug. Do not claim performance by real background or new participant without new labelled captures. Results use one split and seed and do not establish a general improvement.</p><h2>Application</h2><p>Select mobilenet_v2_synthetic to use the new weights. The app includes Grad-CAM, a camera snapshot, and an interactive brightness/blur stress test. Physical webcam operation still needs checking on the user device. Original MP4s were not provided, so this package does not add video playback or live video tracking.</p></html>'''
(r/'SYNTHETIC_REPORT.html').write_text(text)
print(table.to_string(index=False));print(stress.to_string(index=False))
