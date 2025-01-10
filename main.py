import yaml
import torch
import mlflow
from datetime import datetime
from argparse import ArgumentParser

from utils.helpers import *

from models.swinunet import SwinV2_Unet as mdl
from trainer_ad import TrainerSwin as mdltrainer
from visualization.visualize_training import VisualizerTraining


def load_config(config_number: str):
    with open(os.path.join("settings", f"setting_{config_number}.yaml")) as file:
        config = yaml.safe_load(file)
    config["data"]["main_data_path"] = "data/"
    config["data"]["config_number"] = config_number
    return config


def print_config(config):
    printok(f"*** setting {config['data']['config_number']} ***")
    printok(f"\t- dataset: {config['data']['dataset']} - {config['data']['entity']}")
    printok(f"\t- mode: {config['data']['mode']}")
    printok(f"\t- model: {config['model']['type']}")
    printok(f"\t- training window size: {config['data']['seq_len']}")
    printok(f"\t- epochs: {config['training']['n_epochs']}")


def create_folders(config):
    output_dir = os.path.join("outputs", config['data']['mode'], config['model']['type'], f"{config['data']['dataset']}",f"{config['data']['seq_len']}")
    checkpoints_dir = os.path.join("checkpoints", config['data']['mode'], config['model']['type'], f"{config['data']['dataset']}",f"{config['data']['seq_len']}")
    create_folder(os.path.join(output_dir, "figures"))
    create_folder(os.path.join(output_dir, "gif"))
    create_folder(os.path.join(output_dir, "npy"))
    create_folder(checkpoints_dir)
    create_folder("loggings")
    config["dirs"] = {
        "outputs": output_dir,
        "checkpoints": checkpoints_dir,
    }
    return config
    
def get_data_dim(dataset):
    if dataset == 'SMAP':
        return 25
    elif dataset == 'MSL':
        return 55
    elif dataset == 'SMD':
        return 38
    elif str(dataset).startswith("machine"):
        return 38
    elif str(dataset).startswith('HAI'):
        return 79
    elif dataset == 'PSM':
        return 25
    elif dataset == 'SWaT':
        return 50
    elif dataset == 'WADI':
        return 127
    else:
        raise ValueError('unknown dataset ' + str(dataset))


def main_train_eval(mode, config_number, config, device):
    config["data"]["n_channels"] = get_data_dim(config["data"]["dataset"])
    printokblue(f"-> creating model")
    mdltype = config["model"]["type"]
    printokblue(f"\t{mdltype}")

    model = mdl(config, in_channels=config["data"]["n_channels"], device=device).to(device)
    if mode == "train":
        n_plots = min(9, config["data"]["n_channels"])
        vis = VisualizerTraining(config, n_plots=n_plots, plot_loss=False)
        printokblue(f"-> creating trainer")
        trainer = mdltrainer(config, config_number, model, vis, device)
        printokblue(f"-> start training module")
        mlflow.end_run()
        mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
        mlflow.set_experiment(f"{config['data']['dataset']}_{config['data']['entity']}")
        with mlflow.start_run(run_name=str(datetime.now())):
            mlflow.log_params(config['data'])
            mlflow.log_params(config['model'])
            trainer.train()
        vis.close_plot()
    elif mode == "eval":
        printokblue(f"-> creating evaluation module")
        trainer = mdltrainer(config, config_number, model, None, device)
        printokblue(f"-> start evaluation")
        checkpoints_path = f"{config['dirs']['checkpoints']}/{config['data']['entity']}_checkpoint.pt"
        trainer.eval(checkpoints_path)


def main(args):
    if int(args.gpu) >= 0:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    config_number = str(args.setting)
    config = load_config(config_number)
    config["data"]["dataset"] = args.dataset
    config["data"]["seq_len"] = args.seq_len
    config["model"]["type"] = args.model
    if config["data"]["dataset"] == "SMD":
        data_set_number = []
        data_set_number += ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7", "1-8"]
        data_set_number += ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9"]
        data_set_number += ["3-1", "3-2", "3-3", "3-4", "3-5", "3-6", "3-7", "3-8", "3-9", "3-10", "3-11"]
        data_set_number = [f"machine-{e}" for e in data_set_number]
    elif config["data"]["dataset"] == "MSL":
        data_set_number = []
        data_set_number += ["C-1", "C-2", "D-14", "D-15", "D-16", "F-4", "F-5", "F-7", "F-8"]
        data_set_number += ["M-1", "M-2", "M-3", "M-4", "M-5", "M-6", "M-7", "P-10", "P-11", "P-14", "P-15"]
        data_set_number += ["S-2", "T-4", "T-5", "T-8", "T-9", "T-12", "T-13"]
    elif config["data"]["dataset"] == "SMAP":
        data_set_number = []
        data_set_number += ["A-1", "A-2", "A-3", "A-4", "A-5", "A-6", "A-7", "A-8", "A-9", "B-1"]
        data_set_number += ["D-1", "D-2", "D-3", "D-4", "D-5", "D-6", "D-7", "D-8", "D-9", "D-11", "D-12", "D-13"]
        data_set_number += ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "E-10", "E-11", "E-12", "E-13"]
        data_set_number += ["F-1", "F-2", "F-3", "G-1", "G-2", "G-3", "G-4", "G-6", "G-7"]
        data_set_number += ["P-1", "P-2", "P-3", "P-4", "P-7", "R-1", "S-1", "T-1", "T-2", "T-3"]
    else:
        data_set_number = [config["data"]["dataset"]]
    for e in data_set_number:
        config["data"]["entity"] = e
        config = create_folders(config)
        print_config(config)
        main_train_eval(args.mode, config_number, config, device)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--setting",
                        help="settings file number (str)",
                        default="0001",
                        type=str)
    parser.add_argument("--dataset",
                        help="dataset name [SMD, SMAP, MSL, PSM, SWaT, WADI]",
                        default="SMD",
                        type=str)
    parser.add_argument("--seq_len",
                        help="sequence length (int) [64,128,256,512,1024]",
                        default=1024,
                        type=int)
    parser.add_argument("--model",
                        help="model name (str)",
                        default="swin_unet",
                        type=str)
    parser.add_argument("--gpu",
                        help="gpu to use (int)",
                        default=0)
    parser.add_argument("--mode",
                        help="what to do [download, train, infer, display, eval, compare]",
                        default="train",
                        type=str)
    args = parser.parse_args()
    main(args)
    printok(f"\n*** END {args.setting} ***")
