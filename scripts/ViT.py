import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os


MODEL_WEIGHTS = 'best_vit_model.pth'
TEST_DIR = r'../StyleClassificationIndoors/test'
OUTPUT_CSV = "Vit_submission.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


class CustomViT(nn.Module):
    def __init__(self, num_classes=17, drop_rate=0.1):
        super().__init__()

        embed_dim = 768
        depth = 12
        heads = 12

        self.patch_embed = PatchEmbedding(embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=3072,
            dropout=drop_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        b = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)

        x = self.norm(x[:, 0])
        x = self.head(x)

        return x


# --------------------------
# Dataset for Test Set
# --------------------------
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        img_path = os.path.join(self.root_dir, file_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"Warning: corrupted image {file_name}")
            img = Image.new("RGB", (224, 224), color="black")

        if self.transform:
            img = self.transform(img)

        return img, file_name

def run_inference():
    print(f"Running on device: {DEVICE}")

    model = CustomViT(num_classes=17).to(DEVICE)

    if not os.path.exists(MODEL_WEIGHTS):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS}")

    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = TestDataset(TEST_DIR, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    preds = []
    names = []

    with torch.no_grad():
        for imgs, img_names in loader:
            imgs = imgs.to(DEVICE)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            preds.extend(predicted.cpu().tolist())
            names.extend(img_names)

    df = pd.DataFrame({
        "ImageName": names,
        "ClassLabel": preds
    })

    df.to_csv(OUTPUT_CSV, index=False)
    print("DONE. File saved:", OUTPUT_CSV)
    print(df.head())


# Run
if __name__ == "__main__":
    run_inference()
