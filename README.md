# Learning to Separate: Detecting Heavily-Occluded Objects in Urban Scenes

This repository is the official implementation of our ECCV 2020 paper: [Learning to Separate: Detecting Heavily-Occluded Objects in Urban Scenes](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3024_ECCV_2020_paper.php)

## Requirement

Environment:
You only need to install [Tensorflow 1.13](https://www.tensorflow.org/), and some common-used tools like opencv-python and numpy.

Data:
KITTI: Download the dataset from [KITTI](http://www.cvlibs.net/datasets/kitti/). To run the code, you need to first split the dataset into training and validation set by your self.(Random partition is OK.)

CityPersons: Download the images from [CityScapes](https://www.cityscapes-dataset.com/) and the annotations from [here](https://bitbucket.org/shanshanzhang/citypersons/)


## Usage

### 1. Training

```shell
% train KITTI 
python bin/kitti/kitti_train.py --log True --gpu 0

% train CityPersons
python bin/citypersons/cp_trian.py --log True --gpu 0
```

### 2. Test

```shell
% train KITTI
python bin/kitti/kitti_test.py  --gpu 0

% train CityPersons
python bin/citypersons/cp_test.py --gpu 0
```

## Citation

```
@inproceedings{yang2020separate,
  title={ Learning to Separate: Detecting Heavily-Occluded Objects in Urban Scenes},
  author={Yang, Chenhongyi and Ablavsky, Vitaly and Wang, Kaihong and Feng, Qi and Betke, Margrit},
  booktitle={ECCV},
  year={2020}
}
```


