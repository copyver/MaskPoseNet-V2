import itertools
from collections import defaultdict

import numpy as np
import torch.utils.data as data

from data_utils import _isArrayLike


class DatasetBase(data.Dataset):
    """
    Base dataset class.
    """

    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train
        self.num_classes = None
        self.class_ids = None
        self.class_names = None
        self.num_images = None
        self.image_ids = None
        self.scene_pose_gt_info = {}
        self.scene_camera_info = {}
        self.scene_instances_gt_info = {}
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.debug = False

    def __getitem__(self, index):
        if self.is_train:
            while True:
                train_data = self.get_train_data(index)
                if train_data is None:
                    index = self._rand_another(index)
                    continue
                return train_data
        else:
            return self.get_test_data(index)

    def __len__(self):
        return self.num_images  # Todo:modify

    def _rand_another(self, idx):
        pool = [i for i in range(self.__len__()) if i != idx]
        return np.random.choice(pool)

    def get_train_data(self, index):
        """
        Should be overridden by subclasses to return training data for a given index.
        """
        raise NotImplementedError

    def get_test_data(self, index):
        """
        Should be overridden by subclasses to return test data for a given index.
        """
        raise NotImplementedError

    def add_class(self, source, class_id, class_name):
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def createIndex(self):
        anns, cats, imgs = {}, {}, {}
        imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
        annotations = self.scene_instances_gt_info.get('annotations', [])
        images = self.scene_instances_gt_info.get('images', [])
        categories = self.scene_instances_gt_info.get('categories', [])

        for ann in annotations:
            imgToAnns[ann['image_id']].append(ann)
            anns[ann['id']] = ann

        for img in images:
            imgs[img['id']] = img

        for cat in categories:
            cats[cat['id']] = cat

        for ann in annotations:
            catToImgs[ann['category_id']].append(ann['image_id'])

        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getCatIds(self):
        cats = self.scene_instances_gt_info.get('categories', [])
        ids = [cat['id'] for cat in cats]
        return ids

    def loadCats(self, ids):
        if isinstance(ids, list):
            return [self.cats[id] for id in ids]
        else:
            return [self.cats[ids]]

    def loadAnns(self, ids=None):
        if ids is None:
            ids = []
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif isinstance(ids, int):
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=None, catIds=None, iscrowd=None):
        if catIds is None:
            catIds = []
        if imgIds is None:
            imgIds = []
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = list(self.anns.values())
        else:
            if len(imgIds) > 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = list(self.anns.values())
            if len(catIds) > 0:
                anns = [ann for ann in anns if ann['category_id'] in catIds]

        if iscrowd is not None:
            ids = [ann['id'] for ann in anns if ann.get('iscrowd', 0) == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def load_image_annotations(self, image_id):
        return self.image_info[image_id]['annotations']

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        if self.is_train:
            np.random.shuffle(self.image_ids)
