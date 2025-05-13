import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# GAN Training Tool v1.8 – erweitert mit Checkpoint-Intervall 💾
# ─────────────────────────────────────────────────────────────────────────────

def prompt_settings():
    print("┌───────────────────────────────────┐")
    print("│      GAN Training Tool v1.8       │")
    print("└───────────────────────────────────┘")
    resume = input("Training fortsetzen? (j/n): ").strip().lower() == 'j'
    data_path = input("Pfad zum Bildordner: ").strip() or 'celebA/img_align_celeba'
    image_size = int(input("Bildgröße (px) [1024]: ") or 1024)
    latent_dim = int(input("Latente Dimension [128]: ") or 128)
    batch_size = int(input("Batch Size [2]: ") or 2)
    epochs = int(input("Epochen [100]: ") or 100)
    lr = float(input("Lernrate [0.0002]: ") or 0.0002)
    max_imgs = int(input("Max. Anzahl Bilder aus Dataset verwenden (0 = alle): ") or 0)
    checkpoint_interval = int(input("Alle wieviele Epochen Checkpoints speichern? [5]: ") or 5)
    return {
        'resume': resume,
        'data_path': data_path,
        'image_size': image_size,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr,
        'max_imgs': max_imgs,
        'checkpoint_interval': checkpoint_interval,
        'checkpoint_dir': 'checkpoints',
        'sample_dir': 'samples',
        'log_dir': 'runs'
    }

class FlatImageFolder(Dataset):
    def __init__(self, root, transform=None, max_imgs=0):
        self.files = [os.path.join(root, f) for f in sorted(os.listdir(root))
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if max_imgs > 0:
            self.files = self.files[:max_imgs]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # Dummy label

class Generator(nn.Module):
    def __init__(self, z_dim, image_size):
        super().__init__()
        self.init_size = image_size // 16
        self.l1 = nn.Sequential(nn.Linear(z_dim, 512 * self.init_size ** 2))
        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_feat, out_feat, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            block(512, 256),
            block(256, 128),
            block(128, 64),
            block(64, 32),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.size(0), 512, self.init_size, self.init_size)
        img = self.model(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            out = self.main(dummy)
            self.flattened_size = out.view(1, -1).shape[1]
        self.final = nn.Sequential(nn.Flatten(), nn.Linear(self.flattened_size, 1))

    def forward(self, x):
        out = self.main(x)
        return self.final(out)

def train():
    cfg = prompt_settings()
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['sample_dir'], exist_ok=True)
    os.makedirs(cfg['log_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(cfg['latent_dim'], cfg['image_size']).to(device)
    disc = Discriminator(cfg['image_size']).to(device)
    opt_G = optim.Adam(gen.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    opt_D = optim.Adam(disc.parameters(), lr=cfg['lr'], betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()
    scaler_G = torch.cuda.amp.GradScaler()
    scaler_D = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=cfg['log_dir'])

    transform = transforms.Compose([
        transforms.Resize((cfg['image_size'], cfg['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = FlatImageFolder(cfg['data_path'], transform, cfg['max_imgs'])
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)

    start_epoch = 0
    ckpt_path = os.path.join(cfg['checkpoint_dir'], 'last.pth')
    if cfg['resume'] and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        gen.load_state_dict(ckpt['gen'])
        disc.load_state_dict(ckpt['disc'])
        opt_G.load_state_dict(ckpt['opt_G'])
        opt_D.load_state_dict(ckpt['opt_D'])
        start_epoch = ckpt['epoch'] + 1
        print(f"✅ Checkpoint geladen – starte bei Epoche {start_epoch}")

    for epoch in range(start_epoch, cfg['epochs']):
        pbar = tqdm(loader, desc=f"Epoche {epoch+1}/{cfg['epochs']}")
        for i, (imgs, _) in enumerate(pbar):
            imgs = imgs.to(device)
            batch = imgs.size(0)
            valid = torch.ones(batch, 1, device=device)
            fake_label = torch.zeros(batch, 1, device=device)

            # Discriminator
            opt_D.zero_grad()
            with torch.cuda.amp.autocast():
                real_pred = disc(imgs)
                z = torch.randn(batch, cfg['latent_dim'], device=device)
                fake_imgs = gen(z)
                fake_pred = disc(fake_imgs.detach())
                d_loss = (criterion(real_pred, valid) + criterion(fake_pred, fake_label)) / 2
            scaler_D.scale(d_loss).backward()
            scaler_D.step(opt_D)
            scaler_D.update()

            # Generator
            opt_G.zero_grad()
            with torch.cuda.amp.autocast():
                gen_pred = disc(fake_imgs)
                g_loss = criterion(gen_pred, valid)
            scaler_G.scale(g_loss).backward()
            scaler_G.step(opt_G)
            scaler_G.update()

            # Logging
            step = epoch * len(loader) + i
            writer.add_scalar('Loss/Discriminator', d_loss.item(), step)
            writer.add_scalar('Loss/Generator', g_loss.item(), step)

            pbar.set_postfix(D=f"{d_loss.item():.4f}", G=f"{g_loss.item():.4f}")

        # Samples
        with torch.no_grad():
            z = torch.randn(16, cfg['latent_dim'], device=device)
            samples = gen(z)
            save_image((samples + 1) / 2, os.path.join(cfg['sample_dir'], f"epoch_{epoch+1:03}.png"), nrow=4)
            writer.add_images('Samples', (samples + 1) / 2, epoch+1)

        # Save "last" checkpoint
        torch.save({
            'gen': gen.state_dict(),
            'disc': disc.state_dict(),
            'opt_G': opt_G.state_dict(),
            'opt_D': opt_D.state_dict(),
            'epoch': epoch
        }, ckpt_path)

        # Save checkpoint nur alle X Epochen
        if (epoch + 1) % cfg['checkpoint_interval'] == 0:
            epoch_ckpt = os.path.join(cfg['checkpoint_dir'], f"epoch_{epoch+1:03}.pth")
            torch.save({
                'gen': gen.state_dict(),
                'disc': disc.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'epoch': epoch
            }, epoch_ckpt)

    writer.close()

if __name__ == '__main__':
    train()

