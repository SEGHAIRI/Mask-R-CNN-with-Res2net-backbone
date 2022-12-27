import torch
from skimage import io, transform
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np

def get_train_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        ToTensorV2()
        ])

#Dataset Loader
class LoadDataSet(torch.utils.data.Dataset):
        def __init__(self,path, transform=None):
            self.path = path
            self.folders = os.listdir(path)
            self.transforms = get_train_transform()
        
        def __len__(self):
            return len(self.folders)
              
        
        def __getitem__(self,idx):
            #print(self.folders[idx])
            image_folder = os.path.join(self.path,self.folders[idx],'images/')
            mask_folder = os.path.join(self.path,self.folders[idx],'masks/')
            image_path = os.path.join(image_folder,os.listdir(image_folder)[0])
            
            img = io.imread(image_path)[:,:,:3].astype('float32')
            img = transform.resize(img,(128,128))

            boxes, labels, image_id, area, iscrowd = self.get_labels(img, idx,mask_folder)
            mask = self.get_mask(mask_folder, 360, 360 ).astype('uint8')
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            mask = mask.permute(2, 0, 1)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = mask
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            return (img,target) 

        def get_labels(self, img, idx,mask_folder):

            mask = self.get_mask(mask_folder, 128, 128 ).astype('float32')
            augmented = self.transforms(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            mask = mask.permute(2, 0, 1)
            boxes, labels, image_id, area, iscrowd = self.get_boxes(mask,idx)

            return boxes, labels, image_id, area, iscrowd

        def get_mask(self,mask_folder,IMG_HEIGHT, IMG_WIDTH):

            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH,len(os.listdir(mask_folder))))
            num_obj = 0
            for mask_ in os.listdir(mask_folder):
                    mask_ = io.imread(os.path.join(mask_folder,mask_))
                    mask_ = transform.resize(mask_, (IMG_HEIGHT, IMG_WIDTH))
                    mask_ = np.expand_dims(mask_,axis=-1)
                    mask[:,:,num_obj] = mask_.reshape(mask_.shape[0],mask_.shape[1])
                    num_obj += 1
            return mask

        def get_boxes(self,masks,idx):

            boxes = []
            num_objs=0
            for mask in masks:
                num_objs+=1
                mask_ = np.array(mask)
                pos = np.where(mask_)
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            
            #creat targets
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            return boxes, labels, image_id, area, iscrowd