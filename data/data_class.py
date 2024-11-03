from dataclasses import dataclass
import numpy as np
from typing import List


@dataclass
class ImageInfoPacket:
    message: str
    name: str
    image_list: List[np.ndarray]


@dataclass
class InfoPacket:
    message: str
    count: int
    object: str