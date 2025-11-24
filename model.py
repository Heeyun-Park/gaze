import math
import torch
import torch.nn as nn

from models.gaze360.resnet import (
    resnet18, resnet34, resnet50, resnet101, resnet152
)


# ============================================================
# 1. GazeLSTM
#    - ResNet feature → 256-D
#    - BiLSTM with 7-frame sequence
#    - Middle-frame regression (azimuth, elevation, variance)
# ============================================================
class GazeLSTM(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True,
                 img_feature_dim: int = 256, num_frames: int = 7):
        super().__init__()

        self.img_feature_dim = img_feature_dim
        self.num_frames = num_frames

        # Backbone
        if backbone == "resnet18":
            self.base_model = resnet18(pretrained=pretrained)
        elif backbone == "resnet34":
            self.base_model = resnet34(pretrained=pretrained)
        elif backbone == "resnet50":
            self.base_model = resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.base_model = resnet101(pretrained=pretrained)
        elif backbone == "resnet152":
            self.base_model = resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Original fc2 → 256-D output
        self.base_model.fc2 = nn.Linear(1000, self.img_feature_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.img_feature_dim,
            hidden_size=self.img_feature_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        # Final output: 3 (azimuth, elevation, variance_param)
        self.last_layer = nn.Linear(2 * self.img_feature_dim, 3)

    def forward(self, input_seq: torch.Tensor):
        """
        input_seq : (B, 7, 3, 224, 224)
        """
        B, T, C, H, W = input_seq.shape
        assert T == self.num_frames

        # (B*T,3,H,W)
        x = input_seq.view(-1, C, H, W)

        # (B*T,256)
        base_out = self.base_model(x)

        # (B,7,256)
        base_out = base_out.view(B, T, self.img_feature_dim)

        # (B,7,2*256)
        lstm_out, _ = self.lstm(base_out)

        # middle frame
        center_idx = T // 2
        lstm_center = lstm_out[:, center_idx, :]  # (B,512)

        output = self.last_layer(lstm_center).view(B, 3)

        # angle scaling
        ang = output[:, :2]
        ang[:, 0:1] = math.pi * torch.tanh(ang[:, 0:1])          # [-pi, pi]
        ang[:, 1:2] = (math.pi/2) * torch.tanh(ang[:, 1:2])      # [-pi/2, pi/2]

        # variance
        var = math.pi * torch.sigmoid(output[:, 2:3])            # (B,1)
        var = var.expand(-1, 2)                                  # (B,2)

        return ang, var


# ============================================================
# 2. SwitchGazeNet + Skeleton (HEAD_KP / UPPER_KP)
#    - Head branch + Body branch (GazeLSTM)
#    - Skeleton 기반 gating
#    - Residual fusion refinement (skeleton 포함)
# ============================================================
class SwitchGazeNet(nn.Module):
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()

        self.head_net = GazeLSTM(backbone=backbone, pretrained=pretrained)
        self.body_net = GazeLSTM(backbone=backbone, pretrained=pretrained)

        # skeleton: head_vec(15) + upper_vec(24) = 39
        skel_in_dim = 15 + 24
        self.skel_fc = nn.Sequential(
            nn.Linear(skel_in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )

        # gating: [skel_feat(32) + conf(2)] → 2 logits (head/body)
        self.gate_fc = nn.Sequential(
            nn.Linear(32 + 2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )

        # Fusion MLP (residual correction)
        # 입력: head_ang(2) + body_ang(2) + skel_feat(32) = 36
        self.fusion_fc = nn.Sequential(
            nn.Linear(36, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 3),
        )

    def forward(self,
                head_seq: torch.Tensor,
                body_seq: torch.Tensor,
                conf: torch.Tensor,
                head_vec: torch.Tensor,
                upper_vec: torch.Tensor):
        """
        head_seq : (B,7,3,224,224)
        body_seq : (B,7,3,224,224)
        conf     : (B,2)  [head_available, body_available]
        head_vec : (B,15)  5 head keypoints × (x,y,c)
        upper_vec: (B,24)  8 upper keypoints × (x,y,c)
        """

        # 1) Head / Body branches
        head_ang, head_var = self.head_net(head_seq)
        body_ang, body_var = self.body_net(body_seq)

        # 2) Skeleton feature
        skel_in = torch.cat([head_vec, upper_vec], dim=1)  # (B,39)
        skel_feat = self.skel_fc(skel_in)                  # (B,32)

        # 3) Gating weights (skeleton + conf 기반)
        # conf: (B,2)  [head_avail, body_avail]
        gate_in = torch.cat([skel_feat, conf], dim=1)      # (B,34)
        gate_logits = self.gate_fc(gate_in)                # (B,2)
        gate_weights = torch.softmax(gate_logits, dim=1)   # (B,2)

        w_h = gate_weights[:, 0:1]
        w_b = gate_weights[:, 1:2]

        # 4) confidence-aware masking (occlusion case 보호)
        #    conf가 0인 branch는 weight를 거의 0으로 보내도록 보정
        avail_sum = torch.clamp(conf[:, 0:1] + conf[:, 1:2], min=1e-6)
        conf_norm = conf / avail_sum                       # (B,2)
        w_h = w_h * conf_norm[:, 0:1]
        w_b = w_b * conf_norm[:, 1:2]
        w_sum = torch.clamp(w_h + w_b, min=1e-6)
        w_h = w_h / w_sum
        w_b = w_b / w_sum

        # 5) confidence + skeleton 기반 가중 평균
        ang = w_h * head_ang + w_b * body_ang
        var = w_h * head_var + w_b * body_var

        # 6) residual refinement (skeleton 포함)
        fused_in = torch.cat([head_ang, body_ang, skel_feat], dim=1)  # (B,36)
        fused_out = self.fusion_fc(fused_in)                          # (B,3)

        delta_az = fused_out[:, 0:1]
        delta_el = fused_out[:, 1:2]
        delta_v  = fused_out[:, 2:3]

        # corrected angles
        ang[:, 0:1] = 0.5 * ang[:, 0:1] + 0.5 * math.pi * torch.tanh(delta_az)
        ang[:, 1:2] = 0.5 * ang[:, 1:2] + 0.5 * (math.pi/2) * torch.tanh(delta_el)

        # corrected variance
        extra_var = math.pi * torch.sigmoid(delta_v)
        extra_var = extra_var.expand(-1, 2)
        var = 0.5 * var + 0.5 * extra_var

        return ang, var
