# VRDL_HW4 Super Resolution
This is the code for VRDL HW4.

## Dependencies

- python 3.8
- pytorch=1.8.0+cu111
- Pillow>=8.0.1
- scikit-learn>=0.24.0
- scikit-image>=0.18.1
- numpy=1.19.2
- opencv-python>=4.5.1.48

## Preparing and preprocessing
1. download the dataset in codalab
2. put the data under this repository
3. create json file for training
```shell
python3 create_data_lists.py
```

## Training
Run:
```shell
python3 train_srresnet.py
```

## Inference
1. download the weight [here](https://drive.google.com/file/d/1g2T9TrEV55liutONJPFe1WbSuareis8a/view?usp=sharing) and put it under the repo
2. run:
```shell
python3 inference.py
```

## Reference & Acknowledgement
sgrvinod: [a-PyTorch-Tutorial-to-Super-Resolution](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution#tutorial-in-progress)
