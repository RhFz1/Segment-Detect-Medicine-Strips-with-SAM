from dataclasses import dataclass

@dataclass
class ImageConfig:
    img_scale: float = 2.0
    pix_scale: int = 15
    area_real: int = 7.068 # this is cm^2