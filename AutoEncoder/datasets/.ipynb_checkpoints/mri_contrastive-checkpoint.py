import os
import random
import torch
import warnings
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F

class ElasticDeformation2D():
    def __init__(self, alpha=5, sigma=8, p=0.3, kernel_size=17):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.kernel_size = kernel_size

    def make_grid(self, h, w, device=None):
        """生成采样 grid"""
        device = device or torch.device("cpu")
        dx = torch.randn(1, 1, h, w, device=device, dtype=torch.float32) * self.alpha
        dy = torch.randn(1, 1, h, w, device=device, dtype=torch.float32) * self.alpha

        # fallback if gaussian_blur not found
        if hasattr(F, "gaussian_blur"):
            dx = F.gaussian_blur(dx, kernel_size=(self.kernel_size, self.kernel_size), sigma=self.sigma)
            dy = F.gaussian_blur(dy, kernel_size=(self.kernel_size, self.kernel_size), sigma=self.sigma)
        else:
            pass

        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        xx = xx.float().unsqueeze(0).unsqueeze(0)
        yy = yy.float().unsqueeze(0).unsqueeze(0)

        xx = xx + dx
        yy = yy + dy

        xx = 2.0 * xx / (w - 1) - 1.0
        yy = 2.0 * yy / (h - 1) - 1.0
        grid = torch.cat([xx, yy], dim=1).permute(0, 2, 3, 1)  # (1,H,W,2)
        return grid
    
    def apply_grid(self, img, grid):
        x = img.unsqueeze(0)  # (1,C,H,W)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=True)
        return out.squeeze(0)
    
    def __call__(self, imgs):
        """
        Args:
            imgs: torch.Tensor 或 list[torch.Tensor]
                  - 单张图像: (C,H,W)
                  - 多张图像: [(C,H,W), (C,H,W), ...]，共用同一grid 
        """
        ## 对两张图像生成相同的随机位移场，否则会破坏两模态间的像素级解剖对齐，从而违背“解剖一致的流形”目标
        if torch.rand(1).item() > self.p:
            return imgs
        ## 传入list,返回list
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]

        c, h, w = imgs[0].shape
        grid = self.make_grid(h, w, device=imgs[0].device)
        out = [self.apply_grid(img, grid) for img in imgs]
        return out if len(out) > 1 else out[0]

from torchvision.transforms import functional

class PairTransform:
    def __init__(self, p_blur=0.2, p_jitter=0.3, p_elastic=0.3):
        """环境中的 torchvision >= 0.9 / README 标明依赖"""
        self.blur = transforms.GaussianBlur(3)
        self.jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1) ## 灰度图跳过 saturation/hue, torchvision>=0.8/0.9+
        self.elastic = ElasticDeformation2D(alpha=5, sigma=8, p=p_elastic)
        self.p_blur = p_blur
        self.p_jitter = p_jitter

    def __call__(self, img_pair):
        """输入: (x_anchor, x_pos)，都是 tensor (C,H,W)"""
        x_anchor, x_pos = img_pair

        # ---- 同步 GaussianBlur ----
        if torch.rand(1).item() < self.p_blur:
            x_anchor = self.blur(x_anchor)
            x_pos    = self.blur(x_pos)

        # ---- 同步 ColorJitter ----
        if torch.rand(1).item() < self.p_jitter:
            # ColorJitter 每次调用都会采样新参数
            fn_idx, brightness_factor, contrast_factor, _, _ = \
                transforms.ColorJitter.get_params(
                    self.jitter.brightness, self.jitter.contrast,
                    self.jitter.saturation, self.jitter.hue
                )

            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    x_anchor = transforms.functional.adjust_brightness(x_anchor, brightness_factor)
                    x_pos    = transforms.functional.adjust_brightness(x_pos, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    x_anchor = transforms.functional.adjust_contrast(x_anchor, contrast_factor)
                    x_pos    = transforms.functional.adjust_contrast(x_pos, contrast_factor)

        # ---- 同步 Elastic Deformation ----
        x_anchor, x_pos = self.elastic([x_anchor, x_pos])
        ## 传入list,返回list
        return x_anchor, x_pos

class MRIContrastiveDataset(Dataset):
    def __init__(self, root, split="train", transform=None, modalities=None, fixed_pair=False):
        """
        Args:
            root: 数据根目录, 例如 ./data/MRI
                  假设目录结构为 root/train/Patient001/T1.png, T2.png, FLAIR.png ...
            split: "train" / "val"
            modalities: 模态列表, 例如 ("T1", "T2", "FLAIR")
                        如果为 None，会自动根据第一个病人的文件夹推断。
            transform: 图像预处理
            fixed_pair: 如果 True，就只用前两个模态做配对 (modalities[0], modalities[1])
                        如果 False，就在所有模态中随机采样一对 (anchor, pos)
        """
        super().__init__()
        self.root = os.path.join(root, split)
        self.split = split
        if not os.path.exists(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        self.patients = sorted([d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d))])
        if len(self.patients) == 0:
            raise RuntimeError(f"No patient folders found under {self.root}")

        if transform == None:
            ## Note : 数据增强需要保持anatomical consistency
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

        self.pair_transform = PairTransform(p_blur=0.2, p_jitter=0.3, p_elastic=0.3) if split == "train" else None

        # 自动推断可用模态
        if modalities is None:
            first_patient = self.patients[0]
            patient_dir = os.path.join(self.root, first_patient)
            files = os.listdir(patient_dir)
            image_exts = [".png", ".jpg", ".jpeg"]
            modal_set = set()
            for f in files:
                name, ext = os.path.splitext(f)
                if ext.lower() in image_exts:
                    modal_set.add(name)
            if not modal_set:
                raise RuntimeError(f"No image modalities found under {patient_dir} (checked {image_exts})")
            self.modalities = sorted(modal_set)
        else:
            self.modalities = list(modalities)

        self.fixed_pair = fixed_pair


    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, idx): ## PIL.Image
        patient = self.patients[idx]
        patient_dir = os.path.join(self.root, patient)

        if self.fixed_pair:
            # 固定模态对 (T1, T2)
            anchor_mod, pos_mod = self.modalities[:2]
        else:
            # 在所有模态中随机选择不同的两个
            anchor_mod, pos_mod = random.sample(self.modalities, 2)

        ## 文件找不到时抛出FileNotFound Error
        def find_file(patient_dir, base_name):
            for ext in [".png", ".jpg", ".jpeg"]:
                p = os.path.join(patient_dir, f"{base_name}{ext}")
                if os.path.exists(p):
                    return p
            raise FileNotFoundError(f"Modality file not found for patient {patient}: {base_name} (tried .png/.jpg/.jpeg)")

        anchor_path = find_file(patient_dir, anchor_mod)
        pos_path = find_file(patient_dir, pos_mod)
        
        x_anchor = self.transform(Image.open(anchor_path).convert("L"))
        x_pos = self.transform(Image.open(pos_path).convert("L"))

        assert isinstance(x_anchor, torch.Tensor) and isinstance(x_pos, torch.Tensor), \
            f"Dataset must return tensors, got {type(x_anchor)}, {type(x_pos)}"
        assert x_anchor.dim() == 3 and x_pos.dim() == 3, \
            f"Expected (C,H,W), got {x_anchor.shape}, {x_pos.shape}"

        ## PIL.Image -> Tensor

        # 成对模态用同一个 elastic deformation
        if self.pair_transform is not None:
            x_anchor, x_pos = self.pair_transform((x_anchor, x_pos))
            assert isinstance(x_anchor, torch.Tensor) and isinstance(x_pos, torch.Tensor), \
                f"PairTransform must return tensors"
            assert x_anchor.shape == x_pos.shape, \
                f"Anchor/Pos shape mismatch: {x_anchor.shape} vs {x_pos.shape}"

        return x_anchor, x_pos

## 需要实现的是：
## 每次训练的一个组合：x_pos = {Patient A_T1, Patient B_T1, Patient C_T1, Patient D_T1...}
##                 x_anchor = {Patient A_T2, Patient B_T2, Patient C_T2, Patient D_T2...}
"""很难处理的问题：（1）z_pos和z_anchor之间的InfoNCE是一个batchsize的总体nce损失，需要标准化吗？此阶段需不需要训练discriminator，尽量拉远不同患者之间相同模态的分布距离"""
"""备注：采用t-SNE可视化不同患者的模态分布，应该是以一个一个患者为中心的簇，簇中相近的点代表T1w, T2w等不同模态，印证流形假设"""
"""关于训练策略的问题：每次随机选两个样本对作为两个模态的正负样本对？那如何考虑到对于每个患者，他的不同模态被采样的可能性一致，需要设置采样时的随机种子，还要保证stability"""