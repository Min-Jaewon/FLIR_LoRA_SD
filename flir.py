import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FLIRDataset(Dataset):
    def __init__(self, root_dir, transform=None, tokenizer=None):
        self.root_dir = root_dir
        self.train_dir = os.path.join(self.root_dir, 'train', 'thermal_8_bit')
        self.val_dir = os.path.join(self.root_dir, 'val', 'thermal_8_bit')
        self.transform = transform
        self.tokenizer=tokenizer
        self.image_list = self.get_image_list()
        
    def get_image_list(self):
        image_list=[]
        for image in os.listdir(self.train_dir):
            image_list.append(image)

        return image_list

    def __len__(self):
        return len(self.image_list)
    
    def tokenize_empty_caption(self, caption):
        if self.tokenizer:
            inputs = self.tokenizer(
                '', max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            return inputs.input_ids
        else:
            return []
    
    def preprocess(self, image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        max_size=max(image.size)
        center_crop=transforms.CenterCrop(max_size)
        image=center_crop(image)
        image = self.transform(image)[:3,:,:]

        return image
    
    def __getitem__(self, idx):
        cur_image=self.image_list[idx]
        img_dir = os.path.join(self.train_dir, cur_image)
        image = Image.open(img_dir)
        image= self.preprocess(image)
        image_num = cur_image.split('.')[0]
        token = self.tokenize_empty_caption(image)
        return {'pixel_values': image, 'input_ids': token, 'name' : image_num}



