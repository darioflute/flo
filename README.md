# First Look Observations with the Nancy Grace Roman telescope

This repository contains a set of useful codes created to improve the reduction and coaddition of First Look Observations made with the Nancy Grace Roman telescope.

## Installation


# Environment

This section describes how the environment for simulations and data reduction was created.

## PSF data

First, download the latest PSF using the link in https://stpsf.readthedocs.io/en/latest/installation.html
```
mkdir ~/data
cd ~/data
tar xvfz ~/Download/stpsf-data-LATEST.tar.gz 
```
By unarchiving the file stpsf-data-LATEST.tar.gz the directory stpsf-data is created inside the directory data.

## Cache directory

In this directory RomanCal will store files from the CRDS

```
mkdir ~/data/crds_cache
```

## Conda environment

After cloning the distribution locally:

git clone git@github.com:darioflute/flo.git

Do the following:

cd flo

conda env create -n environment.yml

conda activate flo

To exit the environment:

conda deactivate flo

If you decide to remove this environment:

conda remove -n flo --all

answering "y" to the two questions.