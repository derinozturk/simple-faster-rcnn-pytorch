import os
import glob
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

class WiderBboxDataset:
    def __init__(self, data_dir, split='train',
                 variability=list(VARIABILITY)):
        assert set(variability).issubset(VARIABILITY)
        self.id_list_file = os.path.join(data_dir, 'image_lists', '{}.txt'.format(split))
        self.image_paths = []
        self.image_ids = []
        self.data_dir = data_dir
        self.data_split_dir = os.path.join(data_dir, split, 'images')
        self.variability = variability
        self.label_names = WIDER_BBOX_LABEL_NAMES

        subdirs = [os.path.join(self.data_split_dir, sub_dir) for sub_dir in os.listdir(self.data_split_dir)
                                                   if os.path.isdir(os.path.join(self.data_split_dir, sub_dir))]
        for subdir in subdirs:
            self.image_paths.extend(glob.glob(os.path.join(subdir, '*.jpg')))

        for image_path in self.image_paths:
            image_id = image_path.split(os.sep)[-1]
            image_id = image_id.split('.jpg')[0]
            self.image_ids.append(image_id)



    def __len__(self):
        return len(self.image_ids)

    def get_example(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_paths[idx]
        anno = ET.parse(os.path.join(self.data_dir, 'annotations', image_id + '.xml'))
        label = []
        bbox = []
        anno_variability = []
        for obj in anno.findall('object'):
            anno_variability = [obj.find(var_category).text for var_category in VARIABILITY_CATEGORIES]
            #print('Anno var: {}'.format(anno_variability))
            if set(anno_variability).issubset(set(self.variability)):
                continue

            bbox_anno = obj.find('bndbox')
            bbox.append([int(bbox_anno.find(tag).text) - 1
                         for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
            name = obj.find('name').text.lower().strip()
            label.append(WIDER_BBOX_LABEL_NAMES.index(name))
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        variability = np.array(anno_variability)

        # Load a image
        img_file = os.path.join(image_path)
        img = read_image(img_file, color=True)
        #print('bbox: {}'.format(bbox))
        return img, bbox, label, variability

    __getitem__ = get_example
