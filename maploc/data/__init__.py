from .kitti.dataset import KittiDataModule
from .mapillary.dataset import MapillaryDataModule
from .yyc.dataset import YYCDataModule

modules = {"kitti": KittiDataModule, "yyc": YYCDataModule}
