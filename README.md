# s²LFM-Net
High-speed multi-channel volumetric microscopy via spatial-spectral encoding(under review)
# Overview
Here, we proposed a deep-learning decoder, **s²LFM-Net**, to accurately reconstruct the 3D multi-spectral fluorescent distributions from ill-posed, RGB-encoded, light-field measurements. By jointly leveraging spatial, angular, and spectral cues, our decoder **s²LFM-Net** surpasses SOTA unmixing techniques for such ill-posed tasks in fidelity and inference speed across diverse imaging scenarios. With the help of **s²LFM-Net**, we can now reconstruct up to 8 channels information with high spectral fidelity and inference speed from light field RGB measurements. Next, we will show you the intruction of **s²LFM-Net** step by step.
# Environments
## Our hardware configuration
1. a NVIDIA-RTX 3090 GPU or better
2. 128G RAM
3. ~500GB disk space
4. Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
5. Windows 10 Pro
## Our software configuration
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
3. Refer to [Zenodo](https://doi.org/10.5281/zenodo.15905791) for downloading.
### Prepare s²LFM-Net environment
```
conda create -n s2lfmnet python=3.9
conda activate s2lfmnet
pip install -r requirements.txt
```
s²LFM-Net is built upon Anaconda and Pytorch, please make sure you can successfully install both platform. You can refer to [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html) and [Pytorch](https://pytorch.org/) for guides and help.
# Demo
## Download pretrain model
Pretrain model can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1uzhmvDSUzESFG0uTNmDv3njGoF4JRsHs?usp=sharing) or [Zenodo](https://doi.org/10.5281/zenodo.15905791). Please suit yourself~
## Download dataset
Demo dataset can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1u-rY2btQbRUWsy6I7mF9r_Qjni_8b5W0?usp=sharing) or [Zenodo](https://doi.org/10.5281/zenodo.15905791).
## Run the demo
After downloading the pretrian model and dataset, now you can just using
```
cd ~
python demo_inference_simulation.py
```
or again, just click the **run** button with the file _demo_inference_simulation.py_ opened on your complier (please make sure the model and data path are correct in the code).

Now your will see the unmixed 8 channels results waiting for your inside the folder <ins>~/output/simulation_8channels/</ins>

Quick and easy, isn't? ʕง•ᴥ•ʔง
## Train and test your own s²LFM-Net model
If you want to re-train the model, please using
```
cd ~
conda activate s2lfmnet
python demo_train_simulation.py
```
or just click the **run** button with the file _demo_train_simulation.py_ opened on your complier (recommended **Visual Studio Code**).

By a single NVIDIA-3090 GPU, the model would finish training after about 9.4 hours and be save at <ins>~/YourModel/simulation_8channels/epoch_125.pth.tar </ins>

Then your can use our demo inference code (modify with your own parameters if needed) and see what happens.

# Results
Numerical simulated data was randomly synthesis with tubulins and beans in 8 channels (simulation codes are under preparation and would be provided soon). The centerview of RGB measurements are shown on the left. The right part shows the results from **s²LFM-Net**. Scale bars, 20μm.
![screenshot](https://github.com/user-attachments/assets/66c3f53a-5382-44ee-9002-99bb4d7cb4a2)

# Citation
Unavaliable now.
# Acknowledgement

# Correspondence
Should you have any questions regarding this project and the related results, please feel free to contact Zhi Lu([luzhi@tsinghua.edu.cn](luzhi@tsinghua.edu.cn)) and Feihao Sun([sfh21@mails.tsinghua.edu.cn](sfh21@mails.tsinghua.edu.cn)).
