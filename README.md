# EMLight: Lighting Estimation via Spherical Distribution Approximation (AAAI 2021)
![Teaser](teaser1.png)

## Update
- *12/2021*: We release our [Virtual Object Relighting (VOR) Dataset](https://drive.google.com/drive/folders/1mI3ufHgZOmHeShoezk77Gr_GhdqCPGif?usp=sharing) for lighting estimation evaluation. Please refer to Virtual Object Insertion & Rendering section.
- *07/2021*: Our new work [Sparse Needlets for Lighting Estimation with Spherical Transport Loss](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhan_Sparse_Needlets_for_Lighting_Estimation_With_Spherical_Transport_Loss_ICCV_2021_paper.pdf) is accepted to ICCV 2021. This work introduces a new Needlets basis for lighting representation which allows to represent illumination in both spatial and frequency domains. The implementation code is available in `Needlets/` of this repository.

![Teaser](teaser2.png)


## Prerequisites
- Linux or macOS
- Python3, PyTorch
- CPU or NVIDIA GPU + CUDA CuDNN

## Dataset Preparation
[Laval Indoor HDR Dataset](http://indoor.hdrdb.com/#intro) <br>
Thanks to the intellectual property of Laval Indoor dataset, the original datasets and processed training data can not be released from me. Please get access to the dataset by contacting the dataset creator jflalonde@gel.ulaval.ca.

After getting the dataset, the raw illumination map can be processed to generate the training data of the regression network as below:
````bash
cd RegressionNetwork/representation/
python3 distribution_representation.py
````


## Pretrained Models
The pretrained regression model of EMLight (96 anchor points, without depth branch) as well as pretrained densenet-121 can be downloaded from [Google Drive](https://drive.google.com/file/d/1ziqu_hgmGzYXTQLQJPsS1AWLcVJWKzTN/view?usp=sharing). Saving the pretrained models in `RegressionNetwork/checkpoints`. The model parameters  should be adjusted accordingly for inference.

## Training

### Newest Update (Jan-2022)
The sigmoid in the output layers in DenseNet.py should be deleted. To avoid complex learning rate scheduling, I fix the learning rate to 0.0001 in the overfitting stage. The model is trained on subsets of 100, 1000, 2500, ... and the full set gradually. If you find the prediction get stuck in some points (may happen occasionally), you should stop it and load the weights trained on previous subset to retrain it.


Run the command 
````bash
cd RegressionNetwork/
python3 train.py
````
Training tip1: you may overfit the model on a small subset first, then train the model on the full set, to avoid divergence during training. 

Training tip2: you can try to reduce the number of anchor points (e.g., 96) in the model, which helps to converge during training.



## Virtual Object Insertion & Rendering
To evaluate the performance of lighting estimation, we create a Virtual Object Relighting (VOR) dataset to conduct object insertion & rendering in Blender.
The lighting estimaiton performance is evaluated by using the predicted illumination map as the environment light in Blender.

The background scenes of this set include images from [Laval Indoor HDR](http://indoor.hdrdb.com/), [Fast Spatially-Varying Indoor](https://lvsn.github.io/fastindoorlight/supplementary/index.html#), and some wild scenes.
This dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1mI3ufHgZOmHeShoezk77Gr_GhdqCPGif?usp=sharing).

![Teaser](teaser3.png)


### Citation
If you use this code for your research, please cite our papers.
```
@inproceedings{zhan2021emlight,
  title={EMLight: Lighting Estimation via Spherical Distribution Approximation},
  author={Zhan, Fangneng and Zhang, Changgong and Yu, Yingchen and Chang, Yuan and Lu, Shijian and Ma, Feiying and Xie, Xuansong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

```
@inproceedings{zhan2021emlight,
  title={Sparse Needlets for Lighting Estimation with Spherical Transport Loss},
  author={Zhan, Fangneng and Zhang, Changgong and Hu, Wenbo and Lu, Shijian and Ma, Feiying and Xie, Xuansong and Shao, Ling},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2021}
}
```
