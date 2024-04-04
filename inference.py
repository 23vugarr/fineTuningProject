import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from models.mymodels import ResNet18Based

pth = 'checkpoints/saved_model.pth'

idx_to_classes_reversed = {0: '1000_back', 1: '1000_front', 2: '100_back', 3: '100_front', 4: '10_back', 5: '10_front', 6: '20_back', 7: '20_front', 8: '5000_back', 9: '5000_front', 10: '500_back', 11: '500_front', 12: '50_back', 13: '50_front'}


# model initialization
model = ResNet18Based()
#loading pretrained results for pakistani currecy dataset
model.load_state_dict(torch.load(pth)['state_dict'])
model.eval()

test_transforms = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]
)
#preprocess image and get results
img = Image.open('image.png')
img_tensor = test_transforms(img)
img_tensor.unsqueeze_(0)

res = model(img_tensor)
print(idx_to_classes_reversed[np.argmax(res.data.cpu().numpy())])