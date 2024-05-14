# Image Captioning with Attention

This project is a code implementation for the [attend show and tell](https://arxiv.org/abs/1502.03044) research paper with some minor changes in the original architecture.

## Requirements

- Python 3.10 or later

## Install python using MiniConda

1) Download and install MiniConda from [here](https://docs.anaconda.com/free/miniconda/#quick-command-line-install)
2) Create a new environment using the following command:
```bash
$ conda create -n image-captioning python=3.10
```
3) Activate the environment:
```bash
$ conda activate image-captioning
```
## Install Dependencies

1) Install pytorch (cpu version)
```bash
$ conda install pytorch torchvision cpuonly -c pytorch
```
2) Install required packages 
```bash
$ pip install -r requirements.txt
```
## Setup the environment variables
```bash
$ pip cp .env.example .env
```
Set your environment variablesin the `.env` file.