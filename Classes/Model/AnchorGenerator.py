from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch
from typing import List, Tuple


class AnchorGen(AnchorGenerator):
    # def __init__(self, sizes, aspect_ratios):
    #     super().__init__(sizes, aspect_ratios)

    def generate_anchors(self,
                         sizes: List[Tuple[int, int]],
                         aspect_ratios: Tuple[float],
                         dtype: torch.dtype = torch.float32,
                         device: torch.device = torch.device("cpu"),
                         ) -> torch.Tensor:
        base_anchors = []
        sizes = torch.as_tensor(sizes,
                                dtype=dtype,
                                device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios,
                                        dtype=dtype,
                                        device=device)
        for size in sizes:
            ws = size[0] / 2
            hs = size[1] / 2
            for ratio in aspect_ratios:
                w = ws * torch.sqrt(torch.as_tensor(ratio))
                h = hs / torch.sqrt(torch.as_tensor(ratio))
                base_anchors.append([-w, -h, w, h])
        return torch.tensor(base_anchors).round()
