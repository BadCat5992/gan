import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Label, Button, Entry, Scale

# ─────────────────────────────
# Generator + Discriminator
# ─────────────────────────────

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

# ─────────────────────────────
# Utility Funktionen
# ─────────────────────────────

def extract_dims_from_checkpoint(path):
    ckpt = torch.load(path, map_location='cpu')
    gen_state = ckpt['gen']
    for key in gen_state:
        if "l1.0.weight" in key:
            linear_shape = gen_state[key].shape
            out_features, in_features = linear_shape
            latent_dim = in_features
            image_size = int(((out_features // 512) ** 0.5) * 16)
            return latent_dim, image_size
    raise ValueError("Ungültiger Checkpoint.")

def load_models(path):
    latent_dim, image_size = extract_dims_from_checkpoint(path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(latent_dim, image_size).to(device)
    disc = Discriminator(image_size).to(device)

    checkpoint = torch.load(path, map_location=device)
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
    gen.eval()
    disc.eval()

    return gen, disc, device, latent_dim, image_size

def apply_color_variation(images, strength):
    if strength <= 0:
        return images
    images_var = images.clone()
    for img in images_var:
        for c in range(3):
            factor = 1.0 + (random.uniform(-strength, strength))
            img[c] = torch.clamp(img[c] * factor, -1.0, 1.0)
    return images_var

def generate_images(gen, latent_dim, device, num_images, variation_strength, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    z = torch.randn(num_images, latent_dim).to(device)
    with torch.no_grad():
        images = gen(z)
        images = apply_color_variation(images, variation_strength)
    for i, img in enumerate(images):
        save_path = os.path.join(output_dir, f"generated_{i+1}.png")
        save_image((img + 1) / 2, save_path)
    messagebox.showinfo("Fertig 🎉", f"{num_images} Bilder wurden gespeichert!")

def discriminate_image(disc, image_path, image_size, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = disc(img_tensor)
        prob = torch.sigmoid(pred).item()
    messagebox.showinfo("Analyse", f"Echtheit: {prob:.4f}")

# ─────────────────────────────
# GUI Setup
# ─────────────────────────────

def start_gui():
    root = tk.Tk()
    root.title("GAN Generator v2.0 🧠🎨")
    root.geometry("400x400")

    checkpoint_path = tk.StringVar()
    image_path = tk.StringVar()
    num_images = tk.IntVar(value=1)
    variation_strength = tk.DoubleVar(value=0.1)

    def browse_ckpt():
        path = filedialog.askopenfilename(title="Checkpoint auswählen")
        checkpoint_path.set(path)

    def browse_image():
        path = filedialog.askopenfilename(title="Bild auswählen")
        image_path.set(path)

    def run_generation():
        if not checkpoint_path.get():
            messagebox.showerror("Fehler", "Bitte einen Checkpoint auswählen.")
            return
        try:
            gen, _, device, latent_dim, _ = load_models(checkpoint_path.get())
            generate_images(gen, latent_dim, device, num_images.get(), variation_strength.get())
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    def run_discrimination():
        if not checkpoint_path.get() or not image_path.get():
            messagebox.showerror("Fehler", "Bitte Checkpoint und Bild auswählen.")
            return
        try:
            _, disc, device, _, image_size = load_models(checkpoint_path.get())
            discriminate_image(disc, image_path.get(), image_size, device)
        except Exception as e:
            messagebox.showerror("Fehler", str(e))

    # GUI Elements
    Label(root, text="Checkpoint:").pack()
    Entry(root, textvariable=checkpoint_path, width=40).pack()
    Button(root, text="Durchsuchen", command=browse_ckpt).pack(pady=5)

    Label(root, text="Anzahl Bilder:").pack()
    Entry(root, textvariable=num_images).pack()

    Label(root, text="Farbvariation:").pack()
    Scale(root, variable=variation_strength, from_=0.0, to=1.0, orient="horizontal", length=200).pack()

    Button(root, text="Bilder generieren", command=run_generation).pack(pady=10)

    Label(root, text="Analyse-Modus – Bild auswählen:").pack()
    Entry(root, textvariable=image_path, width=40).pack()
    Button(root, text="Durchsuchen", command=browse_image).pack(pady=5)
    Button(root, text="Bild analysieren", command=run_discrimination).pack(pady=10)

    root.mainloop()

# ─────────────────────────────
# Entry Point
# ─────────────────────────────

if __name__ == "__main__":
    start_gui()

