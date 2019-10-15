import os
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image


class Images(Dataset):
    """ test dataset for images """
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_names = [n for n in os.listdir(img_dir)]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_names[idx])).convert('RGB')
        img = self.img_transform(img)
        return img

    def __len__(self):
        return len(self.img_names)

    @property
    def img_transform(self):
        return T.Compose([T.ToTensor()])
