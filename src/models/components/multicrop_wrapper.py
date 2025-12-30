
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MultiCropWrapper(nn.Module):
    """
    MultiCropWrapper adapted for dict-based inputs (time series).
    Groups crops by resolution to forward efficiently.
    """
    def __init__(self, backbone, head):
        super().__init__()
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, crops):
        # Group by resolution (use length of 'time' as resolution proxy)
        grouped = {}
        for crop in crops:
            res = crop['time'].shape[-2]  # sequence length
            grouped.setdefault(res, []).append(crop)

        outputs = []
        for res, group in grouped.items():
            batch = self.concatenate_crops(group)
            outputs.append(self.backbone(**batch))

        return self.head(torch.cat(outputs, dim=0))

    def concatenate_crops(self, crops):
        keys = crops[0].keys()
        return {k: torch.cat([c[k] for c in crops], dim=0) for k in keys}