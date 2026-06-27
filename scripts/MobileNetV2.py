import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os


TEST_DIR = r'./Practical_Test_Samples'
MODEL_WEIGHTS_PATH = 'best_mobilenet_v2_robust.pth'
OUTPUT_CSV = 'mobileNet_submission.csv'
NUM_CLASSES = 17
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=nn.BatchNorm2d):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear (Critical Fix matches your training)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        norm_layer = nn.BatchNorm2d
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
                [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
            ]

        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        # building first layer
        features = [ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))

        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def get_model_for_inference(num_classes):
    print("Building MobileNetV2 Architecture...")
    # 1. Initialize Base with 1000 classes (to match structure)
    model = MobileNetV2(num_classes=1000)

    # 2. Replace Head to match your Fine-Tuned 17 classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class RobustTestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Safe handling for corrupt images
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)
        return image, img_name


def run_test():
    print(f"Running Inference on: {DEVICE}")

    # A. Setup Model
    model = get_model_for_inference(NUM_CLASSES).to(DEVICE)

    # B. Load Your Weights
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"ERROR: Weights file '{MODEL_WEIGHTS_PATH}' not found!")
        return

    try:
        # map_location ensures it works on CPU if GPU is absent (crucial for exam laptops)
        state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)

        # Handle DataParallel if saved with 'module.' prefix
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        print("Weights loaded successfully.")
    except Exception as e:
        print(f"Weights loading failed: {e}")
        return

    model.eval()

    # C. Setup Data
    if not os.path.exists(TEST_DIR):
        print(f"WARNING: Test dir '{TEST_DIR}' not found. Please update the path inside the script.")
        # We stop here to alert you
        return

    test_dataset = RobustTestDataset(root_dir=TEST_DIR, transform=val_transforms)
    # num_workers=0 is SAFER for Windows/Local execution to avoid multiprocessing errors
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    predictions = []
    file_names = []

    print(f"Starting prediction loop on {len(test_dataset)} images...")

    # D. Inference with TTA (Test Time Augmentation)
    with torch.no_grad():
        for i, (images, names) in enumerate(test_loader):
            images = images.to(DEVICE)

            # Forward 1: Original
            out1 = model(images)
            # Forward 2: Horizontal Flip (TTA)
            out2 = model(torch.flip(images, dims=[3]))

            # Average
            avg_probs = (F.softmax(out1, dim=1) + F.softmax(out2, dim=1)) / 2.0
            _, preds = torch.max(avg_probs, 1)

            predictions.extend(preds.cpu().tolist())
            file_names.extend(names)
            # E. Save CSV
            df = pd.DataFrame({'ImageName': file_names, 'ClassLabel': predictions})
            df.to_csv(OUTPUT_CSV, index=False)

            if (i + 1) % 5 == 0:
                print(f"Processed batch {i + 1}/{len(test_loader)}")

    # E. Save CSV
    df = pd.DataFrame({'ImageName': file_names, 'ClassLabel': predictions})
    df.to_csv(OUTPUT_CSV, index=False)
    print("=" * 40)
    print(f"Done! Saved to {OUTPUT_CSV}")
    print(df.head())
    print("=" * 40)


if __name__ == '__main__':
    run_test()