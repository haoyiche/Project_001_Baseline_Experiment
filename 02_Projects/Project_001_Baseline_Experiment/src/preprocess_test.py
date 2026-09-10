from PIL import Image
import torchvision.transforms as transforms

image = Image.open("data/samples/test.jpg")

transform = transforms.Compose([transforms.Resize((224,224))
                                ,transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

tensor = transform(image)

print("Tensor shape:", tensor.shape)

print("Tensor data type:", tensor.dtype)

print("Tensor min value:", tensor.min().item())
print("Tensor max value:", tensor.max().item())
