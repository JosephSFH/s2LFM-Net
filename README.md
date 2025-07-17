# s²LFM-Net
High-speed multi-channel volumetric microscopy via spatial-spectral encoding(under review)
# Overview
Here, we proposed a deep-learning decoder, **s²LFM-Net**, to accurately reconstruct the 3D multi-spectral fluorescent distributions from ill-posed, RGB-encoded, light-field measurements. By jointly leveraging spatial, angular, and spectral cues, our decoder **s²LFM-Net** surpasses SOTA unmixing techniques for such ill-posed tasks in fidelity and inference speed across diverse imaging scenarios. With the help of **s²LFM-Net**, we can now reconstruct up to 8 channels information with high spectral fidelity and inference speed from light field RGB measurements. Next, we will show you the intruction of **s²LFM-Net** step by step.
# Environments
## Basic Hardware Configuration
1. a NVIDIA-RTX 3090 GPU or better
2. 128G RAM
3. ~500GB disk space
4. Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
5. Windows 10 Pro
## Basic Software Configuration
1. Python 3.9.12
2. Pytorch 1.11.0
3. conda 22.9.0
4. CUDA version 12.6
5. see _requirements.txt_ for the version of other packages.
## Startup
### Download s²LFM-Net code
There are three options for you to download our codes:
1. Download our code using
```
cd ~
git clone https://github.com/JosephSFH/s2LFM-Net.git
```
2. Click the **green** button <img width="114" height="44" alt="image" src="https://github.com/user-attachments/assets/1678941c-89cb-407e-a151-ce311bd5e25a" /> on the top right of the wedpage to download .zip.
3. Refer to [Zenodo](https://doi.org/10.5281/zenodo.15860987) for downloading.
### Prepare s²LFM-Net environment
```
conda create -n s2lfmnet python=3.9
conda avtivate s2lfmnet
pip install -r requirements.txt
```
s²LFM-Net is built upon Anaconda and Pytorch, please make sure you can successfully install both platform. You can refer to [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html) and [Pytorch](https://pytorch.org/) for guides and help.
# Demo
## Download pretrain model
Pretrain model can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1xuh2Tuk6kf2MCx6WWFNGJ5Ft0j_4cepn?usp=sharing) or [Zenodo](https://doi.org/10.5281/zenodo.15860987). Please suit yourself~
## Download dataset
Demo dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1qwZ-8G3QGqDtUgFMESaBcEfDcOcAPFi7?usp=sharing). Trainingset is quite large, you can also refer to [Zenodo](https://doi.org/10.5281/zenodo.15860987) for only codes and testset.
## Train s²LFM-Net with numerical simulated data
If you want to re-train the model, please using
```
cd ~
conda activate s2lfmnet
python demo_train_simulation.py
```
or just click the **run** button with the file _demo_train_simulation.py_ opened on your complier (recommended **Visual Studio Code**).

By a single NVIDIA-3090 GPU, the model would finish training after about 9.4 hours and be save at <ins>~/pretrain_model/simulation_8channels/epoch_125.pth.tar </ins>
## Then test and see what happens
Now you can just using
```
cd ~
python demo_inference_simulation.py
```
or again, just click the **run** button with the file _demo_inference_simulation.py_ opened on your complier.

Now your will see the unmixed 8 channels results waiting for your inside the folder <ins>~/output/simulation_8channels/</ins>

Quick and easy, isn't? ʕง•ᴥ•ʔง

## Bounes
We also provided extra demo for real scene. You can use
```
cd ~
python demo_inference_kidney.py
```
or again again, just click the **run** button to see the results from a 4-color human kidney slice.

# Results
Numerical simulated data was randomly synthesis with tubulins and beans in 8 channels (simulation codes are under preparation and would be provided soon). The centerview of RGB measurements are shown on the left. The right part shows the results from **s²LFM-Net**. Scale bars, 20μm.
![screenshot](https://github.com/user-attachments/assets/66c3f53a-5382-44ee-9002-99bb4d7cb4a2)

# Citation
Unavaliable now.
# Acknowledgement

# Correspondence
Should you have any questions regarding this project and the related results, please feel free to contact [Zhi Lu](luzhi@tsinghua.edu.cn) and [Feihao Sun](sfh21@mails.tsinghua.edu.cn).
