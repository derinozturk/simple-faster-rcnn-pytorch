import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image

VARIABILITY =  (
    'blur clear',
    'blur normal',
    'heavy blur',
    'typical expression',
    'exaggerate expression',
    'normal illumination',
    'extreme illumination',
    'no occlusion',
    'partial occlusion',
    'heavy occlusion',
    'typical pose',
    'atypical pose',
    'invalid image',
    'valid image',
)

VARIABILITY_CATEGORIES = (
    'blur',
    'expression',
    'illumination',
    'occlusion',
    'pose',
    'invalid',
)


WIDER_BBOX_LABEL_NAMES = (
    'face',
)

class WiderBBoxDataset:
    def __init__(self, data_dir, split='train',
                 variability=list(VARIABILITY)):
        assert set(variability).issubset(VARIABILITY)
        self.id_list_file = os.path.join(data_dir, 'images', '{}.txt'.format(split))
        self.ids = [id.strip() for id in open(id_list_file)]
        self.data_dir = data_dir
        self.variability = variability

    def __len__(self):
        return len(self.ids)

    def get_example(self, idx):
        id = self.ids[idx]
        anno = ET.parse(os.path.join(self.data_dir, 'annotations' , id + '.xml'))
        label = []
        bbox = []
        variability = []
        for obj in anno.findall('object'):
            anno_variability = [obj.find(var_category) for var_category in VARIABILITY_CATEGORIES]

            if set(anno_variability).issubset(set(self.variability)):
                continue

            bbox_anno = obj.find('bndbox')
            bbox.append([int(bbox_anno.find(tag).text) - 1
                         for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(WIDER_BBOX_LABEL_NAMES.index(name))
        variability = np.array(np.array(anno_variability))

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImage', id_ + '.jpg')
        img = read_image(img_file, color=True)

        return img, bbox, label, variability
