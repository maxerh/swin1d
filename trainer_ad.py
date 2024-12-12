import os
import numpy as np
import torch
import mlflow
from tqdm.auto import tqdm
from pprint import pprint
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Mean
import torch.nn.functional as F

from utils.helpers import get_value_from_tensor

from utils.pot.pot import pot_eval


class Trainer:
    def __init__(self, config, setting, model, vis, device):
        super(Trainer, self).__init__()
        self.device = device
        self.config = config
        self.setting = setting
        self.model = model
        from data_loader import get_loader_segment
        self.datapath = os.path.join(config["data"]["main_data_path"], config["data"]["dataset"], config["data"]["entity"])

        self.train_dl = get_loader_segment(self.datapath, batch_size=self.config["data"]["bs"],
                                           win_size=self.config["data"]["seq_len"], step=config["data"]["seq_stride"],
                                           mode='train',
                                           dataset=config["data"]["dataset"])

        self.test_dl = get_loader_segment(self.datapath, batch_size=1,
                                          win_size=self.config["data"]["seq_len"], step=config["data"]["seq_len"],
                                          mode='test',
                                          dataset=config["data"]["dataset"])

        self.vis = vis
        self.n_epochs = config["training"]["n_epochs"]
        self.n_channels = config["data"]["n_channels"]
        self.now = datetime.now().strftime("%Y%m%d-%H%M%S")

    def train_step(self, x):
        pass

    def train(self):
        pass

    def do_evaluation(self, epoch, final_eval):
        pass


class TrainerSwin(Trainer):
    def __init__(self, config, setting, model, vis, device):
        super(TrainerSwin, self).__init__(config, setting, model, vis, device)

        self.checkpoint_name = f"{self.now}" \
                               f"-{self.setting}" \
                               f"-{config['data']['dataset'].lower()}" \
                               f"-{config['data']['entity'].lower()}" \
                               f"-{config['model']['type'].lower()}" \
                               f"-{config['data']['mode'].lower()}" \
                               f"-sw_{self.config['data']['seq_len']}" \
                               f"-dim_{self.config['model']['embed_dim']}" \
                               f"-lr_{self.config['training']['lr']*1000}"

        self.logdir = os.path.join("loggings", "tensorboard_logging", f"{self.checkpoint_name}")
        self.writer = SummaryWriter(log_dir=self.logdir)
        self.loss_tracker = Mean(device=self.device)
        self.ev = Evaluator(self.model, config, self.train_dl, self.test_dl, device, mode='unet')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config['training']['lr'])

    @torch.no_grad()
    def infer(self):
        checkpoints_path = f"/media/NAS_staff/hoh/model_weights/weights/{self.model.name}/{self.config['data']['dataset']}/{self.config['data']['seq_len']}/"
        entity = self.config['data']['entity'] or self.config['data']['dataset'].lower()
        self.load(f"{checkpoints_path}/{entity}_checkpoint.pt")
        for batch, label in self.test_dl:
            if sum(label[0]) > 0:
                print("anomaly")
            x = batch[:, :self.n_channels, :]
            x_pred = self.model(x.to(self.device))
            time_step_loss = torch.mean(torch.square(x_pred-x.to(self.device)), axis=1)
            for i in range(self.vis.n_plots):
                self.vis.update_subplot(i, signal=[x[0, i, :], get_value_from_tensor(x_pred)[0, i, :]], style=["--", "-"], y=label[0])
            self.vis.update_subplot(i+1, signal=[get_value_from_tensor(time_step_loss)[0]], style=["-"], y=label[0])
            self.vis.draw_plot()
            self.vis.update_subplot(i+1, signal=[get_value_from_tensor(time_step_loss)[0]], style=["-"], y=label[0])

    def eval(self, path):
        self.load(path)
        result, x_test, x_test_pred, y_test, y_test_pred, a_scores_test = self.ev.evaluate(model=self.model)
        return x_test, y_test, x_test_pred, y_test_pred


    def train_step(self, x):
        x_pred = self.model(x.to(self.device))
        loss = F.mse_loss(x.to(self.device), x_pred, reduction='mean')
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return loss

    def train(self):
        for test_batch, test_label in self.test_dl:
            for i, sample in enumerate(test_label):
                s = get_value_from_tensor(sum(sample))
                if s == 0:
                    break
            if s == 0:
                break
        signal_to_plot = torch.unsqueeze(test_batch[i, :, :self.n_channels], 0).transpose(1,2)
        test_label_plot = sample

        pbar = tqdm(range(self.n_epochs))
        for epoch in pbar:
            self.model.train()
            for batch, label in self.train_dl:
                self.opt.zero_grad()
                x = batch[:, :, :self.n_channels].transpose(1,2)
                loss = self.train_step(x)

                self.loss_tracker.update(loss)
                pbar.set_description(f"loss: {self.loss_tracker.compute()}")
            self.tensorboard_logging(epoch, {"loss": self.loss_tracker.compute()}, "training")
            self.loss_tracker.reset()

            out_x0 = self.model(signal_to_plot.to(self.device))
            for i in range(self.vis.n_plots):
                self.vis.update_subplot(epoch, i, signal=[get_value_from_tensor(signal_to_plot)[0, i, :], get_value_from_tensor(out_x0)[0, i, :]], style=["--", "-"], y=test_label_plot)
            self.vis.draw_plot()
            self.writer.add_figure("training/reconstruction", self.vis.fig, global_step=epoch, close=False)
        # end for

        self.vis.save_fig(f"{self.config['dirs']['outputs']}/figures/{self.setting}_{self.now}_{epoch:03d}")
        self.save()
        self.do_evaluation(epoch, final_eval=True)

    def do_evaluation(self, epoch, final_eval=False):
        results = self.ev.evaluate(model=self.model)    # result, x_test, x_test_pred, y_test, y_test_pred, a_scores_test
        result = results[0]
        results_dict = {'F1': result['f1'],
                        'ROC': result['ROC/AUC'],
                        'TP': result['TP'],
                        'TN': result['TN'],
                        'FP': result['FP'],
                        'FN': result['FN'],
                        'P': result['precision'],
                        'R': result['recall'],
                        'threshold': result['threshold'],
                        }
        self.tensorboard_logging(epoch, results_dict, mode="testing")

        if final_eval:
            mlflow.log_params(results_dict)
            evaluation_file = 'eval_all.csv'
            new_data = [self.setting,
                        self.config['data']['dataset'],
                        self.config['data']['entity'],
                        self.config['data']['seq_len'],
                        self.config['model']['embed_dim'],
                        result['TP'],
                        result['TN'],
                        result['FP'],
                        result['FN'],
                        result['threshold'],
                        result['precision'],
                        result['recall'],
                        result["f1"],
                        result["ROC/AUC"],
                        ]
            import csv
            csv_writer = csv.writer(open(evaluation_file, "a"))
            csv_writer.writerow(new_data)


    def tensorboard_logging(self, iteration, values_dict, mode):
        """
        Tensorboard logging of variables during training

        :param iteration: The current iteration (run_iteration)
        :param values_dict: a dictionary of the values that should be updated
        :param mode: a string "training" or "testing"
        """
        for k, v in values_dict.items():
            self.writer.add_scalar(f'{mode}/{k}', v, iteration)

    def save(self):
        data = {
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
        }

        name = f"{self.config['dirs']['checkpoints']}/{self.config['data']['entity']}_checkpoint.pt"
        torch.save(data, name)
        print(f'Saving milestone')

    def load(self, path_to_ckpt):
        data = torch.load(path_to_ckpt, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.opt.load_state_dict(data['opt'])



class Evaluator:
    def __init__(self, model, config, train_dl, test_dl, device, mode):
        super().__init__()
        self.seq_length = config["data"]["seq_len"]
        self.n_channels = config["data"]["n_channels"]
        self.seq_raw = np.zeros((self.seq_length, self.n_channels))
        self.recon_count = np.zeros((self.seq_length, self.n_channels))
        self.a_score = np.zeros(self.seq_length)
        self.stride = config["data"]["seq_stride"]
        self.model = model
        self.config = config
        self.batch_size = config["data"]["bs"]
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.device = device
        self.mode = mode

        self.highest_anomaly_score_train_idx = None
        self.highest_anomaly_score_test_idx = None
        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.x_train_pred = None
        self.x_test_pred = None

    def evaluate(self, model=None):
        print("\n-+-+-+- evaluation -+-+-+-")
        if model is not None:
            self.model = model
        self.model.eval()

        self.x_train = np.zeros((len(self.train_dl)*self.batch_size*self.seq_length, self.n_channels))
        self.x_train_pred = np.zeros_like(self.x_train)

        print(datetime.now(), "- calculating anomaly scores for training dataset")
        for idx_ds, (x, y) in enumerate(self.train_dl):
            x = x.transpose(1,2)
            x = x[:, :self.n_channels, :]
            self.x_train[idx_ds * self.seq_length * self.batch_size : (idx_ds + 1) * self.seq_length * self.batch_size, :] = x.numpy().transpose(0,2,1).reshape((-1,self.n_channels))
            if self.mode == "diffusion":
                _, out = self.model(x.to(self.device), training=False, return_diffusion_process=False)
                out = out[-1]
            else:
                out = self.model(x.to(self.device))
            out = get_value_from_tensor(out).transpose(0,2,1).reshape((-1, self.n_channels))
            self.x_train_pred[idx_ds * self.seq_length * self.batch_size : (idx_ds + 1) * self.seq_length * self.batch_size, :] = out
        a_scores_train =((self.x_train - self.x_train_pred)**2).mean(axis=-1)  # anomaly scores
        self.highest_anomaly_score_train_idx = np.argmax(a_scores_train)
        print(f"  -> Anomaly scores (min|max|avg): {np.min(a_scores_train)}, {np.max(a_scores_train)}, {np.mean(a_scores_train)}")
        print(f"  -> Index of max: {np.argmax(a_scores_train)} ({a_scores_train.shape[0]})")

        #####  Test data  #####
        batch_size=1
        self.x_test = np.zeros((len(self.test_dl)*batch_size*self.seq_length, self.n_channels))
        self.x_test_pred = np.zeros_like(self.x_test)
        self.y_test = np.zeros(self.x_test.shape[0])

        print(datetime.now(), "- calculating anomaly scores for test dataset")
        for idx_ds, (x, y) in enumerate(self.test_dl):
            x = x.transpose(1,2)
            x = x[:, :self.n_channels, :]
            self.x_test[idx_ds * self.seq_length * batch_size: (idx_ds + 1) * self.seq_length * batch_size,:] = x.numpy().transpose(0, 2, 1).reshape((-1, self.n_channels))
            self.y_test[idx_ds * self.seq_length * batch_size: (idx_ds + 1) * self.seq_length * batch_size] = y.numpy().reshape(-1)
            if self.mode == "diffusion":
                _, out = self.model(x.to(self.device), training=False, return_diffusion_process=False)
                out = out[-1]
            else:
                out = self.model(x.to(self.device))
            out = get_value_from_tensor(out).transpose(0, 2, 1).reshape((-1, self.n_channels))
            self.x_test_pred[idx_ds * self.seq_length * batch_size: (idx_ds + 1) * self.seq_length * batch_size,:] = out
        a_scores_test = ((self.x_test - self.x_test_pred) ** 2).mean(axis=-1)  # anomaly scores
        self.highest_anomaly_score_test_idx = np.argmax(a_scores_test)
        self.y_test = self.y_test[:a_scores_test.shape[0]]
        result, _ = pot_eval(a_scores_train, a_scores_test, self.y_test)
        print(f"  -> Anomaly scores (min|max|avg): {np.min(a_scores_test)}, {np.max(a_scores_test)}, {np.mean(a_scores_test)}")
        print(f"  -> Index of max: {np.argmax(a_scores_test)} ({a_scores_test.shape[0]})")
        print(datetime.now(), "- calculating results with POT")
        pprint(result)
        y_test_pred = np.zeros_like(self.y_test)
        y_test_pred[np.where(a_scores_test > result["threshold"])] = 1

        return result, self.x_test, self.x_test_pred, self.y_test, y_test_pred, a_scores_test
