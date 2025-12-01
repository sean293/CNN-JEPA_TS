# pretrain/train_yolov12_jepa.py
import torch
import torch.nn as nn
from pretrain.trainer_common import LightlyModel, main_pretrain
from models.yolov12_backbone import YOLOv12Backbone
from models.sparse_encoder import dense_model_to_sparse

class YOLOv12JEPA(LightlyModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # 1️⃣ Build standard YOLOv12 backbone
        backbone = YOLOv12Backbone(in_ch=3)

        # 2️⃣ Convert to sparse
        self.backbone = dense_model_to_sparse(backbone, verbose=True)
        self.backbone_for_online_eval = self.backbone

        # 3️⃣ Add projection head
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.backbone(x)
        return self.projection_head(x)

    def train_val_step(self, batch, batch_idx, metric_label="train_metrics"):
        (x0, x1), _ = batch[:2]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1.detach())
        self.log(f"{metric_label}/loss", loss, on_epoch=True)
        return loss


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import sys
    cfg = OmegaConf.load(sys.argv[1]) if len(sys.argv) > 1 else OmegaConf.create({
        "seed": 42,
        "optimizer": {"lr": 1e-3, "batch_size": 64},
        "trainer": {"max_epochs": 100, "accelerator": "gpu", "devices": 1},
        "data": {"dataset_name": "imagenette", "num_workers": 4},
        "artifacts_root": "artifacts/yolov12",
        "wandb": False
    })
    main_pretrain(cfg, YOLOv12JEPA)
