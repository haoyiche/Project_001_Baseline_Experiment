from PIL import Image
import numpy as np
import torch

image_path = "data/samples/test.jpg"


image = Image.open(image_path)

print("Image:")
print(image)


print("Image size:", image.size)

image_array = np.array(image)
print("Image array shape:", image_array.shape)
print("Image array data type:", image_array.dtype)
tensor_image = torch.from_numpy(image_array)

print("Tensor image shape:", tensor_image.shape)

print(
    "Before permute:",
    tensor_image.shape
)


tensor_image = tensor_image.permute(
    2,0,1
)


print(
    "After permute:",
    tensor_image.shape
)