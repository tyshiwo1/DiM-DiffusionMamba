from torch.utils.data import Dataset 
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import math
import random
from PIL import Image
import os
import glob
import einops
import torchvision.transforms.functional as F
import json
from absl import logging

class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = tuple(self.dataset[item][:-1])  # remove label
        if len(data) == 1:
            data = data[0]
        return data


class LabeledDataset(Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item], self.labels[item]


class CFGDataset(Dataset):  # for classifier free guidance
    def __init__(self, dataset, p_uncond, empty_token, resolution):
        self.dataset = dataset
        self.p_uncond = p_uncond
        self.empty_token = empty_token
        self.resolution = resolution

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset[item]
        if random.random() < self.p_uncond:
            y = self.empty_token
        return x, y


class DatasetFactory(object):

    def __init__(self):
        self.train = None
        self.test = None

    def get_split(self, split, labeled=False):
        if split == "train":
            dataset = self.train
        elif split == "test":
            dataset = self.test
        else:
            raise ValueError

        if self.has_label:
            return dataset if labeled else UnlabeledDataset(dataset)
        else:
            assert not labeled
            return dataset

    def unpreprocess(self, v):  # to B C H W and [0, 1]
        v = 0.5 * (v + 1.)
        v.clamp_(0., 1.)
        return v

    @property
    def has_label(self):
        return True

    @property
    def data_shape(self):
        raise NotImplementedError

    @property
    def data_dim(self):
        return int(np.prod(self.data_shape))

    @property
    def fid_stat(self):
        return None

    def sample_label(self, n_samples, device, **kwargs):
        raise NotImplementedError

    def label_prob(self, k):
        raise NotImplementedError


# CIFAR10
def center_crop_arr_32(pil_image):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    image_size = 32
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class CIFAR10(DatasetFactory):
    r""" CIFAR10 dataset

    Information of the raw dataset:
         train: 50,000
         test:  10,000
         shape: 3 * 32 * 32
    """

    def __init__(self, path, random_flip=False, cfg=False, p_uncond=None, is_crop=False, resolution=32):
        super().__init__()
        self.resolution = resolution
        transform_train = []
        if is_crop: ###!!!
            transform_train.append(
                transforms.Lambda(center_crop_arr_32),
            )
        
        if resolution != 32:
            transform_train.append(transforms.Resize(resolution))
        
        transform_train = transform_train + [
            transforms.ToTensor(), 
            transforms.Normalize(0.5, 0.5),
        ]

        transform_test = []
        if resolution != 32:
            transform_test.append(transforms.Resize(resolution))

        transform_test += [
            transforms.ToTensor(), 
            transforms.Normalize(0.5, 0.5), 
        ] # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),

        if random_flip:  # only for train
            transform_train.append(transforms.RandomHorizontalFlip())
        
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
        self.train = torchvision.datasets.CIFAR10(path, train=True, transform=transform_train, download=True)
        self.test = torchvision.datasets.CIFAR10(path, train=False, transform=transform_test, download=True)

        assert len(self.train.targets) == 50000
        self.K = max(self.train.targets) + 1
        self.cnt = torch.tensor([len(np.where(np.array(self.train.targets) == k)[0]) for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / 50000 for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt: {self.cnt}')
        print(f'frac: {self.frac}')

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K, resolution=resolution)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution # 32, 32

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_cifar10_train_pytorch.npz'

    def sample_label(self, n_samples, device, **kwargs):
        if kwargs.get('label', None) is not None:
            return torch.ones(n_samples, dtype=torch.long, device=device) * kwargs['label']
        
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


class FeatureDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        # names = sorted(os.listdir(path))
        # self.files = [os.path.join(path, name) for name in names]

    def __len__(self):
        return 1_281_167 * 2  # consider the random flip

    def __getitem__(self, idx):
        path = os.path.join(self.path, f'{idx}.npy.npz')
        _data = np.load(path, allow_pickle=True)
        z, label = _data['z'], _data['label']
        return z, label

class NestedFeatureDataset(FeatureDataset):
    def __init__(self, path, preprocessed_file_path=None):
        super().__init__(path)
        self.path = path
        self.files = []
        if (preprocessed_file_path is not None) and os.path.exists(preprocessed_file_path):
            with open(preprocessed_file_path, 'r') as f:
                self.files = json.load(f)
                f.close()
        else:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith('.npy'):
                        self.files.append(os.path.join(root, file))
            
            with open(preprocessed_file_path, 'w') as f:
                json.dump(self.files, f)
                f.close()
        
        class_names = [os.path.basename(path).split("_")[0] for path in self.files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        self.labels = [sorted_classes[x] for x in class_names]
        assert len(sorted_classes) == 1000, f'{self.files[-1]}, len(self.labels)={len(self.labels)}, {path}'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx): 
        _data = np.load(self.files[idx], allow_pickle=True)
        # z, label = _data['z'], _data['label']
        z = _data
        label = np.array(self.labels[idx], dtype=np.int64)
        return z, label

class ImageNet256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K, resolution=256)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet256_guided_diffusion.npz'

    def sample_label(self, n_samples, device, **kwargs):
        if kwargs.get('label', None) is not None:
            return torch.ones(n_samples, dtype=torch.long, device=device) * kwargs['label']

        return torch.randint(0, 1000, (n_samples,), device=device)

class NestedImageNet256Features(ImageNet256Features):
    def __init__(self, path, cfg=False, p_uncond=None, preprocessed_file_path=None):
        super().__init__(path, cfg, p_uncond)
        self.train = NestedFeatureDataset(path, preprocessed_file_path=preprocessed_file_path)
        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K, resolution=256)

class ImageNet512Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = FeatureDataset(path)
        print('Prepare dataset ok')
        self.K = 1000

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K, resolution=512)

    @property
    def data_shape(self):
        return 4, 64, 64

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet512_guided_diffusion.npz'

    def sample_label(self, n_samples, device, **kwargs):
        if kwargs.get('label', None) is not None:
            return torch.ones(n_samples, dtype=torch.long, device=device) * kwargs['label']

        return torch.randint(0, 1000, (n_samples,), device=device)


class NestedImageNet512Features(ImageNet512Features):
    def __init__(self, path, cfg=False, p_uncond=None, preprocessed_file_path=None):
        super().__init__(path, cfg, p_uncond)
        self.train = NestedFeatureDataset(path, preprocessed_file_path=preprocessed_file_path)
        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K, resolution=512)

class ImageNet(DatasetFactory):
    def __init__(self, path, resolution, random_crop=False, random_flip=True, cfg=False, p_uncond=None):
        super().__init__()

        self.random_crop = random_crop
        self.random_flip = random_flip

        print(f'Counting ImageNet files from {path}')
        train_files = _list_image_files_recursively(os.path.join(path, 'train'))
        class_names = [os.path.basename(path).split("_")[0] for path in train_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        train_labels = [sorted_classes[x] for x in class_names]
        print('Finish counting ImageNet files')

        self.train = ImageDataset(resolution, train_files, labels=train_labels, random_crop=random_crop, random_flip=random_flip)
        self.resolution = (512 if resolution[0] > 512 else resolution[0]) if isinstance(resolution, list) or isinstance(resolution, tuple) else resolution

        # assert False, "{}, {}".format(self.resolution, resolution[0])
        if len(self.train) != 1_281_167:
            print(f'Missing train samples: {len(self.train)} < 1281167')

        self.K = max(self.train.labels) + 1
        cnt = dict(zip(*np.unique(self.train.labels, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()
        self.frac = [self.cnt[k] / len(self.train.labels) for k in range(self.K)]
        print(f'{self.K} classes')
        print(f'cnt[:10]: {self.cnt[:10]}')
        print(f'frac[:10]: {self.frac[:10]}')

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.K, resolution=resolution)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_imagenet{self.resolution}_guided_diffusion.npz'

    def sample_label(self, n_samples, device, **kwargs):
        if kwargs.get('label', None) is not None:
            return torch.ones(n_samples, dtype=torch.long, device=device) * kwargs['label']
            
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device)

    def label_prob(self, k):
        return self.frac[k]


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif "." in entry:
            print(f"Skipping {entry}")
            pass
        elif os.listdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        labels,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.image_paths = image_paths
        self.labels = labels
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        resolution_idx = None
        if isinstance(idx, tuple) or isinstance(idx, list):
            idx, resolution_idx = idx
        
        # print(f'{idx}, {resolution_idx}')
        
        path = self.image_paths[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        if isinstance(self.resolution, tuple) or isinstance(self.resolution, list):
            resolution = self.resolution[ resolution_idx ]
        else:
            resolution = self.resolution

        arr = process_img_input(
            pil_image, resolution, 
            random_crop=self.random_crop, 
            random_flip=self.random_flip,
        )

        label = np.array(self.labels[idx], dtype=np.int64)
        return arr, label
    
def process_img_input(pil_image, resolution, random_crop=False, random_flip=True, ):
    if random_crop:
        arr = random_crop_arr(pil_image, resolution, )
    else:
        arr = center_crop_arr(pil_image, resolution, )

    if random_flip and random.random() < 0.5:
        arr = arr[:, ::-1]

    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    return arr

class MultiscaleBatchSampler(torch.utils.data.BatchSampler):

    def __init__(self, resolution, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution
        assert self.drop_last
    
    def __iter__(self):
        resolution = self.resolution
        resolution_idx = np.random.choice(len(resolution), 1,)[0] 
        print(resolution_idx , 'batchsampler')

        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [
                        [next(sampler_iter)[0], resolution_idx] for _ in range(self.batch_size)
                    ]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]


class MultiscaleDistSampler(torch.utils.data.distributed.DistributedSampler):

    def __init__(self, resolution, mini_batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolution = resolution
        self.mini_batch_size = mini_batch_size
    
    def __iter__(self):
        resolution = self.resolution
        resolution_idx = np.random.choice(len(resolution), self.num_replicas,)
        if (resolution_idx == 0).sum() == 0:
            resolution_idx[0] = 0

        if ((resolution_idx == 1).sum() == 0) and len(resolution_idx) >= 2:
            resolution_idx[1] = 1
        
        if ((resolution_idx == 2).sum() == 0) and len(resolution_idx) >= 3:
            resolution_idx[2] = 2
        
        logging.info(f'resolution_idx num_replicas {resolution_idx} self.num_replicas {self.num_replicas}')
        
        resolution_idx = np.tile(resolution_idx, len(self.dataset) // self.num_replicas + 1 )
        resolution_idx = resolution_idx[:len(self.dataset)]

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        
        indices = np.array(indices).reshape(-1, 1)
        indices = np.concatenate([indices, resolution_idx.reshape(-1, 1)], axis=1)
        indices = indices.tolist()
        # print(resolution_idx[:10], len(indices))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def multiscale_collate_fn(batch, resolution=[512], random_crop=False, random_flip=True):
    '''
        collate_fn=partial(
            multiscale_collate_fn, 
            resolution=train_dataset.resolution,
            random_crop=dataset.random_crop if hasattr(dataset, 'random_crop') else False,
            random_flip=dataset.random_flip if hasattr(dataset, 'random_flip') else False,
        )
    '''
    data, label = zip(*batch) 
    data = np.array(data)
    label = np.array(label)
    assert False, f'{data.shape}, {label.shape}'

    if isinstance(resolution, list) or isinstance(resolution, tuple):
        width, height = pil_image.size
        min_size = min(width, height)
        # print(f'width: {width}, height: {height}, min_size: {min_size}')

        p = np.array([abs(i - min_size) for i in resolution])
        p = p / p.sum()

        resolution_idx = np.random.choice(len(resolution), 1, p=p)[0]
        resolution = resolution[resolution_idx]

        data = process_img_input(
            pil_image, resolution, 
            random_crop=random_crop,
            random_flip=random_flip, 
        )

    data = torch.tensor(data, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)
    return data, label

def center_crop_arr(pil_image, image_size, ):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


# CelebA


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


class CelebA(DatasetFactory):
    r""" train: 162,770
         val:   19,867
         test:  19,962
         shape: 3 * width * width
    """

    def __init__(self, path, resolution=64):
        super().__init__()

        self.resolution = resolution

        cx = 89
        cy = 121
        x1 = cy - 64
        x2 = cy + 64
        y1 = cx - 64
        y2 = cx + 64

        transform = transforms.Compose([Crop(x1, x2, y1, y2), transforms.Resize(self.resolution),
                                        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                        transforms.Normalize(0.5, 0.5)])
        self.train = torchvision.datasets.CelebA(root=path, split="train", target_type=[], transform=transform, download=True)
        self.train = UnlabeledDataset(self.train)

    @property
    def data_shape(self):
        return 3, self.resolution, self.resolution

    @property
    def fid_stat(self):
        return 'assets/fid_stats/fid_stats_celeba64_train_50000_ddim.npz'

    @property
    def has_label(self):
        return False


# MS COCO


def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)

    return np.array(img).astype(np.uint8)


class MSCOCODatabase(Dataset):
    def __init__(self, root, annFile, size=None):
        from pycocotools.coco import COCO
        self.root = root
        self.height = self.width = size

        self.coco = COCO(annFile)
        self.keys = list(sorted(self.coco.imgs.keys()))

    def _load_image(self, key: int):
        path = self.coco.loadImgs(key)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, key: int):
        return self.coco.loadAnns(self.coco.getAnnIds(key))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        image = self._load_image(key)
        image = np.array(image).astype(np.uint8)
        image = center_crop(self.width, self.height, image).astype(np.float32)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = einops.rearrange(image, 'h w c -> c h w')

        anns = self._load_target(key)
        target = []
        for ann in anns:
            target.append(ann['caption'])

        return image, target


def get_feature_dir_info(root):
    files = glob.glob(os.path.join(root, '*.npy.npz'))
    files_caption = glob.glob(os.path.join(root, '*_*.npy.npz'))
    num_data = len(files) - len(files_caption)
    n_captions = {k: 0 for k in range(num_data)}
    for f in files_caption:
        name = os.path.split(f)[-1]
        k1, k2 = os.path.splitext(name)[0].split('_')
        n_captions[int(k1)] += 1
    return num_data, n_captions


class MSCOCOFeatureDataset(Dataset):
    # the image features are got through sample
    def __init__(self, root):
        self.root = root
        self.num_data, self.n_captions = get_feature_dir_info(root)

    def __len__(self):
        return self.num_data

    def __getitem__(self, index):
        z = np.load(os.path.join(self.root, f'{index}.npy.npz'))
        k = random.randint(0, self.n_captions[index] - 1)
        c = np.load(os.path.join(self.root, f'{index}_{k}.npy.npz'))
        return z, c


class MSCOCO256Features(DatasetFactory):  # the moments calculated by Stable Diffusion image encoder & the contexts calculated by clip
    def __init__(self, path, cfg=False, p_uncond=None):
        super().__init__()
        print('Prepare dataset...')
        self.train = MSCOCOFeatureDataset(os.path.join(path, 'train'))
        self.test = MSCOCOFeatureDataset(os.path.join(path, 'val'))
        assert len(self.train) == 82783
        assert len(self.test) == 40504
        print('Prepare dataset ok')

        self.empty_context = np.load(os.path.join(path, 'empty_context.npy'))

        if cfg:  # classifier free guidance
            assert p_uncond is not None
            print(f'prepare the dataset for classifier free guidance with p_uncond={p_uncond}')
            self.train = CFGDataset(self.train, p_uncond, self.empty_context, resolution=256)

        # text embedding extracted by clip
        # for visulization in t2i
        self.prompts, self.contexts = [], []
        for f in sorted(os.listdir(os.path.join(path, 'run_vis')), key=lambda x: int(x.split('.')[0])): 
            _data = np.load(os.path.join(path, 'run_vis', f), allow_pickle=True)
            prompt, context = _data['prompt'], _data['context']
            self.prompts.append(prompt)
            self.contexts.append(context)
        self.contexts = np.array(self.contexts)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return f'assets/fid_stats/fid_stats_mscoco256_val.npz'


class FFHQ1024DatasetFactory(DatasetFactory):

    def __init__(self, path, resolution=1024, random_crop=False, random_flip=True, preprocessed_file_path=None):
        super().__init__()
        self.resolution = resolution
        self.train = FFHQ1024Dataset(
            path, 
            resolution=resolution,
            random_crop=random_crop, 
            random_flip=random_flip, 
            preprocessed_file_path=preprocessed_file_path
        )

    @property
    def fid_stat(self):
        # return f'assets/fid_stats/ffhq{self.resolution}.npz'
        return f'assets/fid_stats/ffhq1024.npz'
    
    @property
    def data_shape(self):
        return 4, self.resolution, self.resolution

    @property
    def has_label(self):
        return False

class FFHQ1024Dataset(Dataset):
    def __init__(self, path, resolution=1024, random_crop=False, random_flip=True, preprocessed_file_path=None):
        super().__init__()
        self.path = path
        self.resolution = resolution
        self.random_flip = random_flip
        self.random_crop = random_crop
        self.files = []
        if (preprocessed_file_path is not None) and os.path.exists(preprocessed_file_path):
            with open(preprocessed_file_path, 'r') as f:
                self.files = json.load(f)
                f.close()
        else:
            for root, dirs, files in os.walk(self.path):
                for file in files:
                    if file.endswith('.png'):
                        self.files.append(os.path.join(root, file))
            
            with open(preprocessed_file_path, 'w') as f:
                json.dump(self.files, f)
                f.close()
        
        # class_names = [os.path.basename(path).split("_")[0] for path in self.files]
        # sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        # self.labels = [sorted_classes[x] for x in class_names]
        # assert len(sorted_classes) == 1000, f'{self.files[-1]}, len(self.labels)={len(self.labels)}, {path}'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        pil_image = Image.open(path)
        pil_image.load()
        pil_image = pil_image.convert("RGB")

        arr = np.array(pil_image)

        if self.resolution == arr.shape[0] and self.resolution == arr.shape[1]:
            pass
        elif self.random_crop:
            arr = random_crop_arr(pil_image, self.resolution)
        else:
            arr = center_crop_arr(pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]

        arr = arr.astype(np.float32) / 127.5 - 1

        # label = np.array(self.labels[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]) #, label

def get_dataset(name, **kwargs):
    if name == 'cifar10':
        return CIFAR10(**kwargs)
    elif name == 'imagenet':
        return ImageNet(**kwargs)
    elif name == 'imagenet256_features':
        return ImageNet256Features(**kwargs)
    elif name == 'imagenet512_features':
        return ImageNet512Features(**kwargs)
    elif name == 'celeba':
        return CelebA(**kwargs)
    elif name == 'mscoco256_features':
        return MSCOCO256Features(**kwargs)
    elif name == 'imagenet256_nested_features':
        return NestedImageNet256Features(**kwargs)
    elif name == 'imagenet512_nested_features':
        return NestedImageNet512Features(**kwargs)
    elif name == 'ffhq1024':
        return FFHQ1024DatasetFactory(**kwargs)
    else:
        raise NotImplementedError(name)
