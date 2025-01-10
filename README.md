![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmaxerh%2Fswin1d&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


# Swin1D-AD

This repository supplements our paper "Rethinking Time Series Anomaly Detection: A Scalable Transformer-Based Framework for Large Contexts".

## Installation

We are using Python-3.8 for running the code. 
```shell
pip3 install -r requirements.txt
```

## Datasets

We are training and testing our model on publicly available datasets:
- SMD: The dataset can be obtained at https://github.com/NetManAIOps/OmniAnomaly
- PSM: The dataset can be obtained at https://github.com/eBay/RANSynCoders
- SWaT/WADI: The distribution rights belong to iTrust. The data can be requested at https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
- SMAP/MSL: The dataset can be obtained using:
```shell
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

We converted the datasets to .pkl-files if necessary. 

## Training

The track different trainings with tensorboard and mlflow.
Start the mlflow server with the following command:
```shell
mlflow server
```

We included yaml-files with all the hyperparameters for each training. 

You can start a training with the settings in the file `settings/setting_1001.yaml` with the following command:

```shell
python main.py -s 1001
```

Multiple trainings can be started by running
```shell
sh start_trainings.sh
```

The results will be saved in `eval_all.csv`.

## Evalutaion

With the command
```shell
python main_eval_csv.py
```
tables the results of the csv are printed for each dataset and input window size.
