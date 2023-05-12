import torch

import monai
from monai.networks.nets import ViTAutoEnc

import time
from functools import partial
import matplotlib.pyplot as plt
import os
import numpy as np

from timm.utils import AverageMeter
from helpers.utils import reduce_tensor


class SSLModel():
    def __init__(self,
                 max_epochs: int = 500,
                 val_interval: int = 2,
                 batch_size: int = 64,
                 lr: float = 1e-4,
                 recon_loss = torch.nn.L1Loss(),
                 local_rank: np.array = [0], #, 1, 2, 3],
                 **kwargs) -> None:
        super().__init__()
        # Training Config
        # Define Network ViT backbone & Loss & Optimizer
        self.model = ViTAutoEnc(
                in_channels=3,
                img_size=(224, 224),
                patch_size=(16, 16),
                pos_embed="conv",
                hidden_size=768,
                mlp_dim=3072,
        )

        self.model = self.model.cuda()
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank], 
                                                          broadcast_buffers=False, find_unused_parameters=True)
        model_without_ddp = self.model.module

        # Define Hyper-paramters for training loop
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.batch_size = batch_size
        self.lr = lr
        self.recon_loss = recon_loss
        self.contrastive_loss = monai.losses.ContrastiveLoss(temperature=0.05)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, **kwargs)
        
        self.epoch_loss_values = []
        self.step_loss_values = []
        self.epoch_cl_loss_values = []
        self.epoch_recon_loss_values = []
        self.val_loss_values = []
        self.best_val_loss = 1e3

        return


    def train(self, train_loader, val_loader, output_path):
        for epoch in range(self.max_epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{self.max_epochs}")
            self.model.train()
            epoch_loss = 0
            epoch_cl_loss = 0
            epoch_recon_loss = 0
            step = 0

            loss_l1_meter = AverageMeter()
            loss_cont_meter = AverageMeter()
            loss_meter = AverageMeter()

            for batch_data in train_loader:
                step += 1
                start_time = time.time()
                
                inputs, inputs_2, gt_input = (
                    batch_data["image"].cuda(non_blocking=True),
                    batch_data["image_2"].cuda(non_blocking=True),
                    batch_data["gt_image"].cuda(non_blocking=True),
                )
                self.optimizer.zero_grad()
                outputs_v1, hidden_v1 = self.model(inputs)
                outputs_v2, hidden_v2 = self.model(inputs_2)

                flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=4)
                flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=4)

                r_loss = self.recon_loss(outputs_v1, gt_input)
                cl_loss = self.contrastive_loss(flat_out_v1, flat_out_v2)

                # Adjust the CL loss by Recon Loss
                total_loss = r_loss + cl_loss * r_loss

                total_loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()

                epoch_loss += total_loss.item()
                self.step_loss_values.append(total_loss.item())

                # CL & Recon Loss Storage of Value
                epoch_cl_loss += cl_loss.item()
                epoch_recon_loss += r_loss.item()


                r_loss_t = reduce_tensor(r_loss)
                cl_loss_t = reduce_tensor(cl_loss)
                total_loss_t = reduce_tensor(total_loss)

                loss_l1_meter.update(r_loss_t.item(), inputs.size(0))
                loss_cont_meter.update(cl_loss_t.item(), inputs.size(0))
                loss_meter.update(total_loss_t.item(), inputs.size(0))

                end_time = time.time()
                print(
                    f"{step}/{len(train_loader.dataset) // train_loader.batch_size}, "
                    f"train_loss: {total_loss.item():.4f}, "
                    f"time taken: {end_time-start_time}s"
                )

            epoch_loss /= step
            epoch_cl_loss /= step
            epoch_recon_loss /= step

            self.epoch_loss_values.append(epoch_loss)
            self.epoch_cl_loss_values.append(epoch_cl_loss)
            self.epoch_recon_loss_values.append(epoch_recon_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if epoch % self.val_interval == 0:
                print("Entering Validation for epoch: {}".format(epoch + 1))
                total_val_loss = 0
                val_step = 0
                self.model.eval()
                for val_batch in val_loader:
                    val_step += 1
                    start_time = time.time()
                    inputs, gt_input = (
                        val_batch["image"].cuda(non_blocking=True),
                        val_batch["gt_image"].cuda(non_blocking=True),
                    )
                    print("Input shape: {}".format(inputs.shape))
                    outputs, outputs_v2 = self.model(inputs)
                    val_loss = self.recon_loss(outputs, gt_input)
                    total_val_loss += val_loss.item()
                    end_time = time.time()

                total_val_loss /= val_step
                self.val_loss_values.append(total_val_loss)
                print(f"epoch {epoch + 1} Validation avg loss: {total_val_loss:.4f}, " f"time taken: {end_time-start_time}s")

                if total_val_loss < best_val_loss:
                    print(f"Saving new model based on validation loss {total_val_loss:.4f}")
                    best_val_loss = total_val_loss
                    checkpoint = {"epoch": self.max_epochs, "state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
                    torch.save(checkpoint, os.path.join(output_path, "best_model.pt"))

        print("Done")

        return


    def plot(self, output_path) -> None:
        plt.figure(1, figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.epoch_loss_values)
        plt.grid()
        plt.title("Training Loss")

        plt.subplot(2, 2, 2)
        plt.plot(self.val_loss_values)
        plt.grid()
        plt.title("Validation Loss")

        plt.subplot(2, 2, 3)
        plt.plot(self.epoch_cl_loss_values)
        plt.grid()
        plt.title("Training Contrastive Loss")

        plt.subplot(2, 2, 4)
        plt.plot(self.epoch_recon_loss_values)
        plt.grid()
        plt.title("Training Recon Loss")

        plt.savefig(os.path.join(output_path, "loss_plots.png"))
        plt.close(1)

        return