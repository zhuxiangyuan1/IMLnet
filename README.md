# IMLSSR
This is the official code of our paper "Stereoscopic Image Super-Resolution with Interactive Memory Learning"

## Requirements
- Python 3.6 (Anaconda is recommended)
- skimage
- imageio
- Pytorch 1.4.0
- torchvision  0.5.0
- tqdm 
- pandas
- cv2 (pip install opencv-python)

### Prepare test data
1. Download the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) dataset and put folders `testing/colored_0` and `testing/colored_1` in `data/test/KITTI2012/original` 
2. Cd to `data/test` and run `generate_testset.m` to generate test data.

## Test
```bash
python test.py 
``` 

## Train
```bash
python train.py 
``` 
