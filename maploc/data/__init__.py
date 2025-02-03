from .kitti.dataset import KittiDataModule
from .mapillary.dataset import MapillaryDataModule
from .yyc.dataset import YYCDataModule

modules = {"mapillary": MapillaryDataModule, "kitti": KittiDataModule, "yyc": YYCDataModule}
