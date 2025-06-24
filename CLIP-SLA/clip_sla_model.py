import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType

class TransformerClassifierWithLoRA(nn.Module):
    def __init__(self, embed_dim=512, num_classes=50, num_layers=2, num_heads=4, dropout=0.3):
        super().__init__()

        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        # LoRA Config like CLIP-SLA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["out_proj"],
            lora_dropout=dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.transformer = nn.TransformerEncoder(get_peft_model(encoder_layer, lora_config), num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        if x.size(1) < 64:
            pad = torch.zeros(x.size(0), 64 - x.size(1), x.size(2)).to(x.device)
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > 64:
            x = x[:, :64, :]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.classifier(x)
