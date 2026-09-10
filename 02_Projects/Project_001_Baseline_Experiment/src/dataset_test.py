from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, image_dir,transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = transform


    def __len__(self):
        return len(self.images)
    

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.image_dir, image_name)

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image





transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



dataset = ImageDataset("data/samples",transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in loader:
    print("Batch shape:", batch.shape)
    

print("Dataset size:", len(dataset))

image = dataset[0]

print("Image shape:", image.shape)
print("Image data type:", image.dtype)
print("First image:", image)
