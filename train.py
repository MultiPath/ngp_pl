import torch
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from utils import slim_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.S = 16 # the interval to update density grid

    def forward(self, rays, split):
        kwargs = {'test_time': split!='train'}
        if self.hparams.dataset_name == 'colmap':
            kwargs['exp_step_factor'] = 1/512
        return render(self.model, rays, **kwargs)

    def setup(self, stage):
        hparams = self.hparams
        dataset = dataset_dict[hparams.dataset_name]
        kwargs = {'root_dir': hparams.root_dir,
                  'downsample': hparams.downsample}
        
        # setup dataset
        self.train_dataset = dataset(split=hparams.split, **kwargs)
        self.train_dataset.batch_size = hparams.batch_size
        self.test_dataset = dataset(split='test', **kwargs)

        # build loss
        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
        for p in self.val_lpips.net.parameters():
            p.requires_grad = False

        # build model
        self.model = NGP(scale=hparams.scale)
        
        # save grid coordinates for training
        G = self.model.grid_size
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))

    def configure_optimizers(self):
        hparams  = self.hparams
        self.opt = FusedAdam(self.model.parameters(), hparams.lr, eps=1e-15)
        self.sch = CosineAnnealingLR(self.opt,
                                     hparams.num_epochs,
                                     hparams.lr/30)
        return [self.opt], [self.sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        K = torch.cuda.FloatTensor(self.train_dataset.K)
        poses = torch.cuda.FloatTensor(self.train_dataset.poses)
        self.model.mark_invisible_cells(K, poses, self.train_dataset.img_wh)

    def training_step(self, batch, batch_nb):
        if self.global_step%self.S == 0:
            self.model.update_density_grid(self.hparams.density_threshold,
                                           warmup=self.global_step<256)

        rays, rgb = batch['rays'], batch['rgb']
        results = self(rays, split='train')
        loss_d = self.loss(results, rgb)
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], rgb)
        self.log('lr', self.opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, prog_bar=True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rays, rgb_gt = batch['rays'], batch['rgb']
        results = self(rays, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                       torch.clip(rgb_gt*2-1, -1, 1))
        logs['lpips'] = self.val_lpips.compute()
        self.val_lpips.reset()

        if not hparams.no_save_test: # save test image to disk
            idx = batch['idx']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs  = torch.stack([x['psnr'] for x in outputs])
        ssims  = torch.stack([x['ssim'] for x in outputs])
        lpipss = torch.stack([x['lpips'] for x in outputs])

        mean_psnr  = all_gather_ddp_if_available(psnrs).mean()
        mean_ssim  = all_gather_ddp_if_available(ssims).mean()
        mean_lpips = all_gather_ddp_if_available(lpipss).mean()

        self.log('test/psnr', mean_psnr, prog_bar=True)
        self.log('test/ssim', mean_ssim)
        self.log('test/lpips_vgg', mean_lpips)


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = NeRFSystem(hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir="logs",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False) if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if (not hparams.no_save_test) and hparams.dataset_name=='nsvf': # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = slim_ckpt(f'ckpts/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt')
        torch.save(ckpt_, f'ckpts/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')
