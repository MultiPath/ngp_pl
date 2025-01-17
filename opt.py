import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # data args
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nsvf', 'colmap', 'rtmv', 'nerfpp'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')
    
    # model args
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--encoder_arch', type=str, default='standard', choices=['standard', 'split', 'tiled', 'volume'])

    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics (experimental)')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='whether to train with random bg color (real dataset only)')

    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    # validation args
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')
    parser.add_argument('--no_save_images', action='store_true', default=False,
                        help='whether to save test image and video')
    
    # checkpointing
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')

    return parser.parse_args()
