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
Download our code using
```
cd ~
git clone https://github.com/JosephSFH/s2LFM-Net.git
```
or click the **green** button <img width="114" height="44" alt="image" src="https://github.com/user-attachments/assets/1678941c-89cb-407e-a151-ce311bd5e25a" /> on the top right of the wedpage to download .Zip.
### Prepare s²LFM-Net environment
```
conda create -n s2lfmnet python=3.9
conda avtivate s2lfmnet
pip install -r requirements.txt
```
s²LFM-Net is built upon Anaconda and Pytorch, please make sure you can successfully install both platform. You can refer to [Anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/index.html) and [Pytorch](https://pytorch.org/) for guides and help.
# Demo

# Results

# Citation

# Acknowledgement

# Correspondence
