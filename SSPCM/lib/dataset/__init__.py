
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .mpii import MPIIDataset as mpii
from .coco import COCODataset as coco
from .ai_challenger import AIDataset as ai_challenger

from .mix_coco_coco import Mixed_COCO_COCO_Dataset as mix_coco_coco

from .mix_mpii_ai import Mixed_MPII_AI_Dataset as mix_mpii_ai


from .mix_fisheye_fisheye import Mixed_Fisheye_Fisheye_Dataset as mix_fisheye_fisheye
from .fisheye import FisheyeDataset as fisheye
