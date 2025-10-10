from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        masks: Tensor = None,
        language_feats: Tensor = None,
    ) -> Float[Tensor, ""]:
        if language_feats is not None:
            delta = prediction.language_image - language_feats
        else:
            if masks is not None:
                delta = prediction.color* masks[..., None,:,:] - batch["target"]["image"] * masks[..., None,:,:]
            else:
                delta = prediction.color - batch["target"]["image"]
        return self.cfg.weight * (delta**2).mean()
