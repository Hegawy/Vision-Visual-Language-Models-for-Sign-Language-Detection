import torch
import torch.nn as nn
from torchvision.models import resnet18
import clip

class HybridVisionEncoder(nn.Module):
    def __init__(self, use_clip=True, num_classes=7):
        super().__init__()
        self.use_clip = use_clip
        self.num_classes = num_classes

        if self.use_clip:
            # Load CLIP model and extract visual encoder
            self.model_clip, _ = clip.load("ViT-B/32", device="cpu", jit=False)
            self.visual = self.model_clip.visual.eval().float()
            for p in self.visual.parameters():
                p.requires_grad = False
            self.embed_dim = 768
        else:
            # Load pretrained ResNet18 and remove final classification head
            self.visual = resnet18(pretrained=True)
            self.visual.fc = nn.Identity()
            for p in self.visual.parameters():
                p.requires_grad = False
            self.embed_dim = 512

        # Classifier on pooled temporal features
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):  # x: [B, T, 3, 224, 224]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W).to(torch.float32)

        if self.use_clip:
            # Process with CLIP vision encoder
            x = self.visual.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
            cls_token = self.visual.class_embedding.unsqueeze(0).expand(x.shape[0], 1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = x + self.visual.positional_embedding
            x = self.visual.ln_pre(x)
            x = self.visual.transformer(x)
            x = self.visual.ln_post(x[:, 0, :])  # Take CLS token
        else:
            # Process with ResNet encoder
            x = self.visual(x)  # shape: [B*T, 512]

        # Temporal pooling and classification
        x = x.view(B, T, -1).mean(dim=1)
        return self.classifier(self.dropout(x))
