import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image

VARIABILITY_2_DIFFICUTLY = {
    'blur clear': 0,
    'blur normal': 1,
    'heavy blur': 2,
    'typical expression': 3,
    'exaggerate expression': 4,
    'normal illumination': 5,
    'extreme illumination': 6,
    'no occlusion': 7,
    'partial occlusion': 8,
    'heavy occlusion': 9,
    'typical pose': 10,
    'atypical pose': 11,
    'invalid image': 12,
    'valid image': 13,
}

WIDER_BBOX_LABEL_NAMES = (
    'human face':
)

class WiderBBoxDataset:
    def __init__(self, data_dir, split='train',
                 variability=['Scale', 'Pose', 'Expression']):
        self.id_list_file = os.path.join(data_dir, 'images', '{}.txt'.format(split))
        self.ids = [id.strip() for id in open(id_list_file)]
        self.data_dir = data_dir
        #self.label_names = WIDER_BBOX_LABEL_NAMES # WIDER doesnt have separate class names
        self.variability = variability

    def __len__(self):
        return len(self.ids)

    def get_example(self, idx):
        id = self.ids[idx]
        anno = ET.parse(os.path.join(self.data_dir), 'Annotations' , id + '.xml')
        label = []
        bbox = []
        for obj in anno.findall('object'):
            if not int(obj.find('difficulty') in self.variability):
                continue

            bbox_anno = obj.find('bndbox')
            bbox.append([int(bndbox_anno.find(tag).text) - 1
                         for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(WIDER_BBOX_LABEL_NAMES.index(name))
        variability = np.array(variability)

        # Load a image
        img_file = os.path.join(self.data_dir, 'JPEGImage', id_ + '.jpg')

        return img, bbox, label, variability
