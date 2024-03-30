from model import Generator
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

# load CLIP
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", 
                                          do_normalize=True, 
                                          do_convert_rgb=True, 
                                          do_rescale=False)

img = Image.open("imgs/y.png")
img = ToTensor()(img)
with torch.cuda.amp.autocast():
    zebra_pred = processor(["realistic horse", "realistic zebra"], img, return_tensors="pt")
    zebra_pred['input_ids'] = zebra_pred['input_ids'].to(device)
    zebra_pred['attention_mask'] = zebra_pred['attention_mask'].to(device)
    zebra_pred['pixel_values'] = zebra_pred['pixel_values'].to(device)
    zebra_pred = clip_model(**zebra_pred)

argmax = torch.softmax(zebra_pred["logits_per_image"], dim=1)
print(argmax)


# train the model
for epoch in range(n_epochs):

    model_horse.train()
    model_zebra.train()
    
    for (x, y) in tqdm(train_loader):

        with torch.cuda.amp.autocast():
        
            # x is a horse, y is a zebra       
            x, y = x.cuda(), y.cuda()
        
            optimizer_horse.zero_grad()
            optimizer_zebra.zero_grad()
        
            y_hat = model_horse(x)
            x_hat = model_zebra(y_hat)

            z_hat = model_zebra(y)
            k_hat = model_horse(z_hat)

            x = train_dataset.denormalize(x)
            y = train_dataset.denormalize(y)

            zebra_pred = processor(["realistic horse", "realistic zebra"], y_hat, return_tensors="pt")
            zebra_pred['input_ids'] = zebra_pred['input_ids'].to(device)
            zebra_pred['attention_mask'] = zebra_pred['attention_mask'].to(device)
            zebra_pred['pixel_values'] = zebra_pred['pixel_values'].to(device)
            zebra_pred = clip_model(**zebra_pred)["logits_per_image"]

            zebra = processor(["realistic horse", "realistic zebra"], y, return_tensors="pt")
            zebra['input_ids'] = zebra['input_ids'].to(device)
            zebra['attention_mask'] = zebra['attention_mask'].to(device)
            zebra['pixel_values'] = zebra['pixel_values'].to(device)
            zebra = clip_model(**zebra)["logits_per_image"]

        break

    break

save_image(y, "imgs/y.png")
save_image(y_hat, "imgs/y_hat.png")
zebra
zebra_pred