from model import Generator, Discriminator
from dataset import UnpairedDataset
from tqdm import tqdm
import torch
import lpips
from torchvision.utils import save_image

from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms import ToTensor
from PIL import Image

# create a dataloader
train_dataset = UnpairedDataset("train")
test_dataset = UnpairedDataset("test")

# compute the logits 
device = "cuda" if torch.cuda.is_available() else "cpu"

# create a dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=6, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=6, shuffle=False)

# create a model
model_horse = Generator(32).cuda()
model_zebra = Generator(32).cuda()

discriminator_horse = Discriminator(32).cuda()
discriminator_zebra = Discriminator(32).cuda()

# create an optimizer
optimizer_horse = torch.optim.AdamW(model_horse.parameters(), lr=1e-5)
optimizer_zebra = torch.optim.AdamW(model_zebra.parameters(), lr=1e-5)

# create a loss function
loss_l1 = torch.nn.L1Loss()
loss_perceptual = lpips.LPIPS(net='vgg').cuda().eval()
weights = [1.0, 1.0, 0.25, 0.25]
n_epochs = 25

def moving_average(x: torch.Tensor, window: int = 3):
    return torch.nn.functional.avg_pool2d(x, kernel_size=window, stride=1, padding=window // 2)

for epoch in range(n_epochs):

    model_horse.train()
    model_zebra.train()
    
    for (x, y) in tqdm(train_loader):

        with torch.cuda.amp.autocast():
           
            x, y = x.cuda(), y.cuda()
        
            optimizer_horse.zero_grad()
            optimizer_zebra.zero_grad()
        
            x_ = model_horse(x)
            x__ = model_zebra(x_)

            y_ = model_zebra(y)
            y__ = model_horse(y_)

            x = train_dataset.denormalize(x)
            y = train_dataset.denormalize(y)

            x_hat = torch.cat((x_, y_), dim=1)
            discriminator_horse_pred = discriminator_horse(x_hat)

            print(discriminator_horse_pred.shape)





        break

    break

# save_image(y, "imgs/y.png")
# save_image(y_hat, "imgs/y_hat.png")
# zebra
# zebra_pred