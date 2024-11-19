import numpy as np
import torch.utils.data as data


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
        self.imgs_list = None
        self.scene_pose_gt_info = {}
        self.scene_camera_info = {}
        self.scene_instances_gt_info = {}
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.debug = False

    def __getitem__(self, index):
        if self.is_train:
            return self.get_train_data(index)
        else:
            return self.get_test_data(index)

    def __len__(self):
        return len(self.imgs_list)

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

    def prepare(self):
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self.imgs_list = list(range(self.num_images))
        print("Number of images: %d" % self.num_images)
        print("Number of classes: %d" % self.num_classes)

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

    def on_epoch_end(self):
        """Shuffle the dataset at the end of each epoch."""
        if self.is_train:
            np.random.shuffle(self.imgs_list)
