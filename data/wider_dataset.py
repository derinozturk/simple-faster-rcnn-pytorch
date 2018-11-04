import os
import xml.etree.ElementTree as ET
import numpy as np
from .util import read_image
import settings

class WiderBBoxDataset:
    def __init__(self, data_dir, split='train',
                 variability=['Scale', 'Pose', 'Expression']):
        self.id_list_file = os.path.join(data_dir, 'images', '{}.txt'.format(split))
        self.ids = [id.strip() for id in open(id_list_files)]
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

            bbox.append