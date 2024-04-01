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

lambda_A = 10
lambda_B = 10
lambda_idt_A = 0.5
lambda_idt_B = 0.5
standard = False
wasserstein = True

# compute the logits
device = "cuda" if torch.cuda.is_available() else "cpu"

seed_env(42)
batch_size = 3
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
    lr=1e-5,
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

            # domain B
            generated_B_fake = generator_A_to_B(A)
            discriminated_B_fake = discriminator_B(generated_B_fake.detach())
            discriminator_B_real = discriminator_B(B)


            # ordinary GAN loss
            if standard:
                
                loss_A = mse(discriminated_A_fake, torch.zeros(b, 1, h, w).cuda())
                loss_A += mse(discriminated_A_real, torch.ones(b, 1, h, w).cuda())

                loss_B = mse(discriminated_B_fake, torch.zeros(b, 1, h, w).cuda())
                loss_B += mse(discriminator_B_real, torch.ones(b, 1, h, w).cuda())

                loss = (loss_A + loss_B) / 2.0

            elif wasserstein:
                
                # critic loss

                loss_A = torch.mean(discriminated_A_fake)
                loss_A -= torch.mean(discriminated_A_real)

                loss_B = torch.mean(discriminated_B_fake)
                loss_B -= torch.mean(discriminator_B_real)


                # gradient penalty
                alpha = torch.rand(b, 1, 1, 1).cuda()
                interpolated_A = (alpha * A + (1 - alpha) * generated_A_fake).requires_grad_(True)
                interpolated_B = (alpha * B + (1 - alpha) * generated_B_fake).requires_grad_(True)

                disc_A = discriminator_A(interpolated_A)
                disc_B = discriminator_B(interpolated_B)

                grad_A = torch.autograd.grad(
                    outputs=disc_A,
                    inputs=interpolated_A,
                    grad_outputs=torch.ones_like(disc_A),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

                grad_B = torch.autograd.grad(
                    outputs=disc_B,
                    inputs=interpolated_B,
                    grad_outputs=torch.ones_like(disc_B),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

                grad_penalty_A = (grad_A.norm(2, dim=1) - 1) ** 2
                grad_penalty_B = (grad_B.norm(2, dim=1) - 1) ** 2

                loss_A += grad_penalty_A.mean() * 10
                loss_B += grad_penalty_B.mean() * 10

                loss = (loss_A + loss_B) / 2.0

            disc_loss.append(loss.item())

        disc_opt.zero_grad()
        disc_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(disc_opt.parameters(), 0.5)
        disc_scaler.step(disc_opt)
        disc_scaler.update()

        with torch.cuda.amp.autocast():

            discriminated_A_fake = discriminator_A(generated_A_fake)
            discriminated_B_fake = discriminator_B(generated_B_fake)

            loss_A = mse(discriminated_A_fake, torch.ones(b, 1, h, w).cuda())
            loss_B = mse(discriminated_B_fake, torch.ones(b, 1, h, w).cuda())

            cycle_B = generator_A_to_B(generated_A_fake)
            cycle_A = generator_B_to_A(generated_B_fake)

            loss = (
                loss_A
                + loss_B
                + loss_l1(cycle_B, B) * lambda_A
                + loss_l1(cycle_A, A) * lambda_B
                + loss_l1(generator_B_to_A(A), A) * lambda_idt_B
                + loss_l1(generator_A_to_B(B), B) * lambda_idt_B
            )

            gen_loss.append(loss.item())

        gen_opt.zero_grad()
        gen_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(gen_opt.parameters(), 0.5)
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
    for A, B in tqdm(test_loader):

        A, B = A.cuda(), B.cuda()

        with torch.cuda.amp.autocast():
            B_fake = generator_A_to_B(A)
            A_fake = generator_B_to_A(B)

        A_fake = test_dataset.denormalize(A_fake)
        B_fake = test_dataset.denormalize(B_fake)

        A_fake = torch.clamp(A_fake, 0, 1)
        B_fake = torch.clamp(B_fake, 0, 1)

        save_image(A_fake, f"imgs/zebra_{k}.png")
        save_image(B_fake, f"imgs/horse_{k}.png")

        k += 1

    del A
    del B
    del A_fake
    del B_fake
