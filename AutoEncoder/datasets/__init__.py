from .mri_contrastive import MRIContrastiveDataset
def get_dataset(dataset_name, root, split="train", modalities=None, transform=None, fixed_pair=False, allow_same_modality=True):
    """
    返回对应的数据集实例
    dataset_name: "mri_contrastive" 或别名
    root: dataset 根目录（上一级），例如 "./data/MRI"
    split: "train"/"val"/"test"
    modalities: 可选的模态列表
    transform: torchvision transform，若为 None 使用默认
    fixed_pair: 是否固定使用 modalities[:2] 作为配对
    """
    ds = dataset_name.lower()
    if ds in ("mri_contrastive", "mri_png", "mri"): ## 数据集种类可扩展
        return MRIContrastiveDataset(root=root, split=split, modalities=modalities, transform=transform, fixed_pair=fixed_pair, allow_same_modality=allow_same_modality)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
## 更合理的策略是先训练几百个epoch，使reconstruction loss能逐步收敛之后再加入对比损失 ##