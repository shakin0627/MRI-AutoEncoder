import torch
import pytorch_lightning as pl
from datasets import get_dataset

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, root,
                 dataset_name="mri_contrastive",
                 batch_size=8, num_workers=4,
                 modalities=None, fixed_pair=False,
                 transform=None, pin_memory=True, allow_same_modality=True):
        """
        root: 数据根目录（例如 ./data/MRI），该目录下应包含 train/ val/ test 子目录
        dataset_name: 在 datasets.get_dataset 中注册的名字
        batch_size, num_workers: dataloader 参数
        modalities: 可选的模态列表，默认由 dataset 自动推断
        fixed_pair: 是否固定用 modalities[:2] 做配对
        transform: 传给 dataset 的 transform
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.pin_memory = pin_memory
        self.modalities = modalities
        self.fixed_pair = fixed_pair
        self.allow_same_modality = allow_same_modality

        self.train_set = None
        self.val_set = None
        self.test_set = None

    
    def setup(self, stage=None):
        # stage: "fit", "test", or None (both)
        if stage == "fit" or stage is None:
            self.train_set = get_dataset(self.dataset_name, root=self.root, split="train", modalities=self.modalities, fixed_pair=self.fixed_pair, transform=self.transform, allow_same_modality=self.allow_same_modality)
            self.val_set   = get_dataset(self.dataset_name, root=self.root, split="val", modalities=self.modalities, fixed_pair=self.fixed_pair, transform=self.transform, allow_same_modality=self.allow_same_modality)

        if stage == "test" or stage is None:
            self.test_set  = get_dataset(self.dataset_name, root=self.root, split="test",  modalities=self.modalities, fixed_pair=self.fixed_pair, transform=self.transform, allow_same_modality=self.allow_same_modality)

    @staticmethod
    def _collate_fn(batch):
        # batch 是 list of (anchor_tensor, pos_tensor)
        anchor, pos = zip(*batch)
        anchor = torch.stack(anchor, dim=0)
        pos = torch.stack(pos, dim=0)
        assert anchor.shape == pos.shape, \
            f"Anchor/Pos batch mismatch: {anchor.shape} vs {pos.shape}"
        return anchor, pos
    
    ## _get_item方法返回的x_pos和x_anchor是来源于一个随机选取的患者的样本对，均为（C，H，W）
    ## _collate_fn这个静态方法负责将B=batch_size个tensor堆叠成一个batch
    
    def train_dataloader(self):
        """shuffle=True 打乱 dataset 的索引顺序（self.patients 的 index）"""
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    

    def val_dataloader(self):
        """shuffle=False"""
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
