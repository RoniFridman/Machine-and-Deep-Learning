import torch
import os
import shutil
from torchvision.utils import save_image


def reproduce_hw3():
    img_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists('images'):
        shutil.rmtree('images', ignore_errors=True)
    train_loader = torch.load('dataloader.pt')
    model = torch.load('model.pkl').to(device)
    os.mkdir('images')
    sample_batch, _ = train_loader.__iter__().next()
    sample_batch = sample_batch.to(device)
    model.discrete_vae.to(device)

    for idx, img in enumerate(sample_batch):
        save_image(img, f"images/{idx + 1}.png")
        img = torch.reshape(img, [1, 3, img_size, img_size])
        recon_img_d = model.discrete_vae(img, 1, False)[0].reshape(-1, img_size, img_size)
        recon_img_c = model(img)[0].reshape(-1, img_size, img_size)
        save_image(recon_img_d, f"images/discrete_{idx + 1}.png")
        save_image(recon_img_c, f"images/continues_{idx + 1}.png")
        if idx == 4:
            break


if __name__ == "__main__":
    reproduce_hw3()
