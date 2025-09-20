import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from callbacks.save_best_parts_reconstruction import SaveBestPartsAndReconstruction
from datamodules.mri_datamodule import MRIDataModule
from models.autoencoder import PreTrained_AutoEncoder
import os
import sys


def check_dependencies():
    """检查必要的依赖和文件是否存在"""
    try:
        # 检查数据目录
        root = "./data/MRI"
        if not os.path.exists(root):
            raise FileNotFoundError(f"MRI data directory not found: {root}")
        
        # 检查数据目录是否为空
        if not os.listdir(root):
            raise FileNotFoundError(f"MRI data directory is empty: {root}")
        
        # 检查必要的模块文件是否存在
        required_files = [
            "./datamodules/mri_datamodule.py",
            "./models/autoencoder.py",
            "./callbacks/save_best_parts_reconstruction.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print("Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            raise FileNotFoundError("Please ensure all required files exist")
        
        print("All dependencies and files check passed")
        return True
        
    except Exception as e:
        print(f"Dependency check failed: {e}")
        return False

def setup_device_info():
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"   GPU Available: {device_name}")
        print(f"   Device count: {device_count}")
        print(f"   Memory: {memory_total:.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
        
        # At least 4GB, otherwise cuda out of memory
        if memory_total < 4.0:
            print("Warning: GPU memory < 4GB, consider reducing batch_size")
        
        return "gpu", 1
    else:
        print("Using CPU (GPU not available)")
        return "cpu", 1
def create_callbacks():
    """创建训练回调函数"""
    callbacks = [
        # 进度条
        TQDMProgressBar(refresh_rate=20),
        
        # 模型检查点保存
        ModelCheckpoint(
            monitor="val_loss",
            filename="checkpoint-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,  # 保存最好的3个模型
            mode="min",
            save_last=True,  # 同时保存最后一个epoch
            verbose=True
        ),
        
        # 自定义的重建图像和最佳部件保存
        SaveBestPartsAndReconstruction(
            save_dir="./checkpoints", 
            monitor="val_loss", 
            n_samples=8  # 增加样本数用于更好的可视化
        )
    ]
    
    return callbacks
def create_optimized_datamodule():
    """创建优化的数据模块"""
    # 根据可用内存调整参数
    if torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if memory_gb >= 8:
            batch_size = 16
            num_workers = 8
        elif memory_gb >= 4:
            batch_size = 8
            num_workers = 4
        else:
            batch_size = 4
            num_workers = 2
    else:
        batch_size = 4
        num_workers = 2
    
    print(f"📊 Using batch_size={batch_size}, num_workers={num_workers}")
    
    datamodule = MRIDataModule(
        dataset_name="mri_contrastive",
        root="./data/MRI",
        batch_size=batch_size,
        modalities=("T1", "T2"),
        num_workers=num_workers,
        fixed_pair=False,   
        allow_same_modality=True,
    )
    
    return datamodule
def create_optimized_model():
    """创建优化的模型"""
    # 为MRI图像优化的参数
    model = PreTrained_AutoEncoder(
        lr=1e-4,           
        lambda_rec=1.0,     
        lambda_nce=0.0,     
        proj_dim=128,       # 256
        temperature=0.1,    
    )
    
    return model

def main():

    print("Starting MRI AutoEncoder Training")
    print("=" * 50)
    
    os.makedirs("./checkpoints", exist_ok=True)
    work_dir = os.path.abspath("./checkpoints")
    
    print(f"Working directory: {work_dir}")

    # 1. check dependencies
    if not check_dependencies():
        print("Please fix the dependency issues before training")
        sys.exit(1)
    
    # 2. setup
    accelerator, devices = setup_device_info()
    
    # 3. datamodule
    try:
        datamodule = create_optimized_datamodule()
        print("DataModule created successfully")
    except Exception as e:
        print(f"Failed to create DataModule: {e}")
        sys.exit(1)
    
    # 4. model
    try:
        model = create_optimized_model()
        print("Model created successfully")
    except Exception as e:
        print(f"Failed to create model: {e}")
        sys.exit(1)
    
    # 5. callback
    callbacks = create_callbacks()
    
    # 6. trainer
    trainer = pl.Trainer(
        max_epochs=100,           # 增加训练轮数
        accelerator=accelerator,
        devices=devices,
        logger=False,
        log_every_n_steps=10,
        callbacks=callbacks,
        default_root_dir=work_dir,
        
        gradient_clip_val=1.0,    # 梯度裁剪，防止梯度爆炸
        deterministic=False,      # 允许一些随机性以获得更好的性能
        enable_progress_bar=True,
        enable_model_summary=True,
        
        # 验证设置
        val_check_interval=0.5,   # 每半个epoch验证一次
        check_val_every_n_epoch=1,
        
    )
    

    try:
        print("\n Starting training...")
        print("=" * 50)
        trainer.fit(model, datamodule=datamodule)
        print("\n Training completed successfully!")
        
        if trainer.checkpoint_callback:
            best_path = trainer.checkpoint_callback.best_model_path
            best_score = trainer.checkpoint_callback.best_model_score
            print(f"Best model saved at: {best_path}")
            print(f"Best validation loss: {best_score:.4f}")
            
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
        print("   Partial results saved in checkpoints directory")
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        print("   Please check your data and model configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
