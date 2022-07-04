# Instant-NGP PyTorch

This repo is adapted from https://github.com/kwea123/ngp_pl


Instant-ngp (only NeRF) in pytorch+cuda trained with pytorch-lightning (**high quality with high speed**). This repo aims at providing a concise pytorch interface to facilitate future research, and am grateful if you can share it (and a citation is highly appreciated)!

* [Example Video1](https://user-images.githubusercontent.com/11364490/177025079-cb92a399-2600-4e10-94e0-7cbe09f32a6f.mp4), [Example Video2](https://user-images.githubusercontent.com/11364490/176821462-83078563-28e1-4563-8e7a-5613b505e54a.mp4)

*  [Official CUDA implementation](https://github.com/NVlabs/instant-ngp/tree/master)
*  [torch-ngp](https://github.com/ashawkey/torch-ngp) another pytorch implementation.

# :computer: Installation
Run 
```bash
bash bolt_setup.sh
```
Or you can check the following steps:
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n ngp_pl python=3.8` to create a conda environment and activate it by `conda activate ngp_pl`)
* Python libraries
    * Install pytorch by `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113`
    * Install `tinycudann` following their [instruction](https://github.com/NVlabs/tiny-cuda-nn#requirements) (compilation and pytorch extension)
    * Install `apex` following their [instruction](https://github.com/NVIDIA/apex#linux)
    * Install core requirements by `pip install -r requirements.txt`

* Cuda extension: Upgrade `pip` to >= 22.1 and run `pip install models/csrc/`

# :books: Data preparation

Download preprocessed datasets from [NSVF](https://github.com/facebookresearch/NSVF#dataset).

# :key: Training

Quickstart: `python train.py --root_dir <path/to/lego> --exp_name Lego`

It will train the lego scene for 20k steps (each step with 8192 rays), and perform one testing at the end. The training process should finish within about 5 minutes (saving testing image is slow, add `--no_save_test` to disable). Testing PSNR will be shown at the end.

More options can be found in [opt.py](opt.py).

# :mag_right: Testing

Use `test.ipynb` to generate images. Lego pretrained model is available [here](https://github.com/kwea123/ngp_pl/releases/tag/v1.0)

# :books: Benchmarks

To run benchmarks, use the scripts under `benchmarking`.

Followings are my results (qualitative results [here](https://github.com/kwea123/ngp_pl/issues/7)):

<details>
  <summary>Synthetic-NeRF</summary>

|       | Mic   | Ficus | Chair | Hotdog | Materials | Drums | Ship  | Lego  | AVG   |
| :---: | :---: | :---: | :---: | :---:  | :---:     | :---: | :---: | :---: | :---: |
| PSNR  | 35.23 | 33.64 | 34.78 | 36.76  | 28.77     | 25.61 | 29.57 | 34.69 | 32.38 |
| FPS   | 40.81 | 34.02 | 49.80 | 25.06  | 20.08     | 37.77 | 15.77 | 36.20 | 32.44 |

</details>

<details>
  <summary>Synthetic-NSVF</summary>

|       | Wineholder | Steamtrain | Toad | Robot | Bike | Palace | Spaceship | Lifestyle | AVG | 
| :---: | :---: | :---: | :---: | :---: | :---:  | :---:  | :---: | :---: | :---: |
| PSNR  | 31.06 | 35.65 | 34.49 | 36.23 | 36.99 | 36.36 | 35.48 | 33.96 | 35.03 |
| FPS   | 47.07 | 75.17 | 50.42 | 64.87 | 66.88 | 28.62 | 35.55 | 22.84 | 48.93 |

</details>

<details>
  <summary>Tanks and Temples</summary>

|      | Ignatius | Truck | Barn  | Caterpillar | Family | AVG   | 
|:---: | :---:    | :---: | :---: | :---:       | :---:  | :---: |
| PSNR | 28.90    | 28.21 | 28.92 | 26.30       | 33.77  | 29.22 |
| *FPS | 10.04    |  7.99 | 16.14 | 10.91       | 6.16   | 10.25 |

*Evaluated on `test-traj`

</details>

<details>
  <summary>BlendedMVS</summary>

|       | *Jade  | *Fountain | Character | Statues | AVG   | 
|:---:  | :---:  | :---:     | :---:     | :---:   | :---: |
| PSNR  | 25.69  | 26.91     | 30.16     | 26.93   | 27.42 |
| **FPS | 26.02  | 21.24     | 35.99     | 19.22   | 25.61 |

*I manually switch the background from black to white, so the number isn't directly comparable to that in the papers.

**Evaluated on `test-traj`

</details>


# TODO

- [ ] support custom dataset
