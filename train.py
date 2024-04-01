from models import Generator, Discriminator
from dataset import UnpairedDataset
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from torch.cuda.amp import GradScaler
from util import seed_env, moving_average
from torch.utils.data import DataLoader

train_dataset = UnpairedDataset("train")
test_dataset = UnpairedDataset("test")

# compute the logits
device = "cuda" if torch.cuda.is_available() else "cpu"

seed_env(42)
batch_size = 2
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

generator_A_to_B = Generator(64).cuda()
generator_B_to_A = Generator(64).cuda()

discriminator_A = Discriminator(64).cuda()
discriminator_B = Discriminator(64).cuda()

gen_opt = torch.optim.AdamW(
    list(generator_A_to_B.parameters()) + list(generator_B_to_A.parameters()),
    lr=2e-5,
    betas=(0.5, 0.999),
)
disc_opt = torch.optim.AdamW(
    list(discriminator_A.parameters()) + list(discriminator_B.parameters()),
    lr=1e-5,
    betas=(0.5, 0.999),
)

loss_l1 = torch.nn.L1Loss()
mse = torch.nn.L1Loss()
n_epochs = 150

gen_scaler = GradScaler()
disc_scaler = GradScaler()


def train_one_epoch(
    generator_A_to_B,
    generator_B_to_A,
    discriminator_A,
    discriminator_B,
    dataloader,
    gen_opt,
    disc_opt,
    gen_scaler,
    disc_scaler,
    mse,
    loss_l1,
    epoch,
):
    generator_A_to_B.train()
    generator_B_to_A.train()

    disc_loss = []
    gen_loss = []

    torch.cuda.empty_cache()

    pbar = tqdm(dataloader)

    for A, B in pbar:

        A, B = A.cuda(), B.cuda()

        with torch.cuda.amp.autocast():

            b = A.size(0)

            # domain A
            generated_A_fake = generator_B_to_A(B)  # B -> A
            discriminated_A_fake = discriminator_A(generated_A_fake.detach())
            discriminated_A_real = discriminator_A(A)

            h, w = discriminated_A_fake.shape[-2:]

            loss_A = mse(discriminated_A_fake, torch.zeros(b, 1, h, w).cuda())
            loss_A += mse(discriminated_A_real, torch.ones(b, 1, h, w).cuda())

            # domain B
            generated_B_fake = generator_A_to_B(A)
            discriminated_B_fake = discriminator_B(generated_B_fake.detach())
            discriminator_B_real = discriminator_B(B)

            loss_B = mse(discriminated_B_fake, torch.zeros(b, 1, h, w).cuda())
            loss_B += mse(discriminator_B_real, torch.ones(b, 1, h, w).cuda())

            loss = (loss_A + loss_B) / 2.0

            disc_loss.append(loss.item())

        disc_opt.zero_grad()
        disc_scaler.scale(loss).backward()
        disc_scaler.step(disc_opt)
        disc_scaler.update()

        with torch.cuda.amp.autocast():

            discriminated_A_fake = discriminator_A(generated_A_fake)
            discriminated_B_fake = discriminator_B(generated_B_fake)

            loss_A = mse(discriminated_A_fake, torch.ones(b, 1, h, w).cuda())
            loss_B = mse(discriminated_B_fake, torch.ones(b, 1, h, w).cuda())

            cycle_A = generator_A_to_B(generated_A_fake)
            cycle_B = generator_B_to_A(generated_B_fake)

            loss = (
                loss_A
                + loss_B
                + loss_l1(cycle_B, A) * 10.0
                + loss_l1(cycle_A, B) * 10.0
                + loss_l1(generator_B_to_A(A), A)
                + loss_l1(generator_A_to_B(B), B)
            ) / 6.0

            gen_loss.append(loss.item())

        gen_opt.zero_grad()
        gen_scaler.scale(loss).backward()
        gen_scaler.step(gen_opt)
        gen_scaler.update()

        pbar.set_description(
            f"Discriminator Loss: {moving_average(disc_loss)} Generator Loss: {moving_average(gen_loss)} Epoch: {epoch}"
        )

    return torch.mean(torch.tensor(disc_loss)), torch.mean(torch.tensor(gen_loss))


best_loss_gen = float("inf")

for epoch in range(n_epochs):

    disc_loss, gen_loss = train_one_epoch(
        generator_A_to_B,
        generator_B_to_A,
        discriminator_A,
        discriminator_B,
        train_loader,
        gen_opt,
        disc_opt,
        gen_scaler,
        disc_scaler,
        mse,
        loss_l1,
        epoch,
    )

    if gen_loss < best_loss_gen:
        best_loss_gen = gen_loss
        torch.save(generator_A_to_B.state_dict(), f"checkpoints/gen_A_{epoch}.pth")
        torch.save(generator_B_to_A.state_dict(), f"checkpoints/gen_B_{epoch}.pth")

    generator_A_to_B.eval()
    generator_B_to_A.eval()

    k = 0
    for x, y in tqdm(test_loader):

        x, y = x.cuda(), y.cuda()

        with torch.cuda.amp.autocast():
            x_ = generator_A_to_B(x)
            y_ = generator_B_to_A(y)

        x_ = test_dataset.denormalize(x_)
        y_ = test_dataset.denormalize(y_)

        x_ = torch.clamp(x_, 0, 1)
        y_ = torch.clamp(y_, 0, 1)

        save_image(x_, f"imgs/zebra_{k}.png")
        save_image(y_, f"imgs/horse_{k}.png")

        k += 1

    del x
    del y
    del x_
    del y_
