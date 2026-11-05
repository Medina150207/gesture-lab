"""Reproduce both A2 reference models. No test-based model selection."""
import argparse,json,random,time,platform
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from PIL import Image
from torchvision import transforms
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent/'results');args=p.parse_args()
out=args.output;out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(2)
random.seed(42);np.random.seed(42);torch.manual_seed(42)
manifest=pd.read_csv(args.data/'manifest.csv');assert manifest.groupby('group')['split'].nunique().max()==1
aug=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomAffine(10,translate=(.05,.05),fill=128),transforms.ColorJitter(brightness=.2,contrast=.1)])
class Photos(torch.utils.data.Dataset):
 def __init__(self,rows,augment=False):self.rows=rows.reset_index(drop=True);self.augment=augment
 def __len__(self):return len(self.rows)*(2 if self.augment else 1)
 def __getitem__(self,i):
  r=self.rows.iloc[i%len(self.rows)]
  with Image.open(args.data/r.path) as im:im=prepare(im)
  if self.augment and i>=len(self.rows):im=aug(im)
  return tensor(im),CLASSES.index(r['class'])
rows={s:manifest[manifest.split==s] for s in ['train','validation','test']}
# Keep original and one reproducibly augmented train view; no augmentation in val/test.
metrics=[]
for name in ['resnet18','mobilenet_v2']:
 torch.manual_seed(42);net=build(name,True).eval()
 for param in net.parameters():param.requires_grad=False
 for param in head(net,name).parameters():param.requires_grad=True
 cache={}
 for s in ['train','validation','test']:
  loader=torch.utils.data.DataLoader(Photos(rows[s],s=='train'),batch_size=32,shuffle=False)
  fs=[];ys=[]
  with torch.no_grad():
   for x,y in loader:fs.append(features(net,name,x));ys.append(y)
  cache[s]=(torch.cat(fs),torch.cat(ys));print(name,s,'features ready',flush=True)
 h=head(net,name);opt=torch.optim.Adam(h.parameters(),lr=.001);hist=[];best=float('inf');state=None
 for epoch in range(1,41):
  h.train();x,y=cache['train'];idx=torch.randperm(len(y));total=0;correct=0
  for ids in idx.split(32):
   opt.zero_grad();logits=h(x[ids]);loss=nn.functional.cross_entropy(logits,y[ids]);loss.backward();opt.step();total+=loss.item()*len(ids);correct+=(logits.argmax(1)==y[ids]).sum().item()
  h.eval()
  with torch.no_grad():
   vx,vy=cache['validation'];vl=h(vx);v_loss=nn.functional.cross_entropy(vl,vy).item();v_acc=(vl.argmax(1)==vy).float().mean().item()
  hist.append(dict(epoch=epoch,train_loss=total/len(y),train_accuracy=correct/len(y),validation_loss=v_loss,validation_accuracy=v_acc))
  if v_loss<best:best=v_loss;best_epoch=epoch;state={k:v.clone() for k,v in h.state_dict().items()}
 h.load_state_dict(state);net.eval()
 torch.save({'architecture':name,'classes':CLASSES,'size':SIZE,'mean':MEAN,'std':STD,'state_dict':net.state_dict(),'best_epoch':best_epoch,'protocol':'frozen ImageNet backbone; 40 head epochs; original + one augmented train view'},out/(name+'.pt'))
 pd.DataFrame(hist).to_csv(out/(name+'_history.csv'),index=False)
 fig,axs=plt.subplots(1,2,figsize=(10,4))
 for ax,key in zip(axs,['loss','accuracy']):
  for subset in ['train','validation']:ax.plot([v['epoch'] for v in hist],[v[subset+'_'+key] for v in hist],label=subset)
  ax.set(xlabel='Head epoch',ylabel=key,title=name);ax.legend()
 fig.tight_layout();fig.savefig(out/(name+'_curves.png'),dpi=150);plt.close(fig)
 with torch.no_grad():tx,ty=cache['test'];probs=h(tx).softmax(1).numpy()
 pred=probs.argmax(1);pr,re,f1,_=precision_recall_fscore_support(ty,pred,average='macro',zero_division=0)
 timings=[]
 for r in list(rows['test'].itertuples())[:20]:
  with Image.open(args.data/r.path) as im:_,ms,_,_,_=predict(net,im,False)
  timings.append(ms)
 metrics.append(dict(model=name,accuracy=accuracy_score(ty,pred),macro_precision=pr,macro_recall=re,macro_f1=f1,best_epoch=best_epoch,mean_forward_ms=np.mean(timings),test_images=len(ty)))
 predictions=rows['test'][['id','path','class','source']].copy();predictions['predicted']=[CLASSES[i] for i in pred]
 for k,c in enumerate(CLASSES):predictions['p_'+c]=probs[:,k]
 predictions.to_csv(out/(name+'_test_predictions.csv'),index=False)
 cm=confusion_matrix(ty,pred,labels=[0,1,2]);fig,ax=plt.subplots(figsize=(5,4));ax.imshow(cm,cmap='Blues')
 for i in range(3):
  for j in range(3):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
 ax.set(xticks=range(3),yticks=range(3),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='True',title=name);fig.tight_layout();fig.savefig(out/(name+'_confusion.png'),dpi=150);plt.close(fig)
 print(metrics[-1],flush=True)
pd.DataFrame(metrics).to_csv(out/'comparison.csv',index=False)
(out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'device':'CPU','threads':2,'seed':42,'input_size':SIZE,'latency':'mean of 20 warmed single-image forwards; excludes decode, preprocessing and Grad-CAM'},indent=2))
print('DONE',flush=True)
