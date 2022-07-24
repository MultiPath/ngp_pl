import torch
import time
import numpy as np
from models.networks import NGP
from models.rendering import render
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from datasets import dataset_dict
from utils import load_ckpt
from train import depth2img
from opt import get_opts


def run_test(hparams):
    dataset = dataset_dict[hparams.dataset_name](
        hparams.root_dir, split='test', downsample=1.0
    )
    model = NGP(scale=hparams.scale).cuda()
    load_ckpt(model, f'ckpts/{hparams.exp_name}/epoch=19_slim.ckpt')
    
    psnrs = []; ts = []

    for img_idx in tqdm(range(len(dataset))):
        rays = dataset.rays[img_idx][:, :6].cuda()

        t = time.time()
        results = render(model, rays, **{'test_time': True, 'T_threshold': 1e-2})
        torch.cuda.synchronize()
        ts += [time.time()-t]

        if dataset.split != 'test_traj':
            rgb_gt = dataset.rays[img_idx][:, 6:].cuda()
            psnrs += [psnr(results['rgb'], rgb_gt).item()]

    if psnrs: 
        print(f'mean PSNR: {np.mean(psnrs):.2f}, min: {np.min(psnrs)}, max: {np.max(psnrs)}')
    print(f'mean time: {np.mean(ts):.4f} FPS: {1/np.mean(ts):.2f}')


if __name__ == '__main__':
    hparams = get_opts()
    run_test(hparams)
