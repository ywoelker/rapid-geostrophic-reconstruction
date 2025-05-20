
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from datetime import datetime

import torch 
import torch.nn as nn
import numpy as np
import xarray as xr
from copy import deepcopy

from amoc_reconstruction.reconstruction.dataset import merge_profiles_max_profiles

def move_data_to_device(data, device):
    if isinstance(data, dict):
        return {k: move_data_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list) or isinstance(data, tuple):
        return [move_data_to_device(x, device) for x in data]
    else:
        return data.to(device)

def train(model, train_dataloader, val_dataloader, device = 'cpu', seed = None, verbose = True, geostrophic_target = False, lr = 1e-3, wd = 1e-6, num_epochs = 80):

    if seed is not None:
        torch.random.manual_seed(seed)
        np.random.seed(seed)


    model.float()
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    optimizer_all = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # lr_scheduler_all = optim.lr_scheduler.MultiStepLR(optimizer_all, milestones=[15,30, 80, 100, 150], gamma=.5)
    lr_scheduler_all = optim.lr_scheduler.CosineAnnealingLR(optimizer_all, T_max=num_epochs, eta_min=1e-6)

    writer = SummaryWriter(f'../rapid-geostrophic-reconstruction/runs/susterv3-{datetime.now().strftime("%Y-%m-%d_%H-%M")}')
    best_val_loss = 1e10
    best_model = None
    best_epoch = 0

    train_losses = []

    torch.autograd.set_detect_anomaly(True)

    # model.node_assigner.requires_grad_(False)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        model.train()
        loss_per_epoch = 0.0

        
        # if epoch >= 15 and type(model) == ProfileModelSUSTeR5_fast:
        #     if (type(model.node_assigner) == DistanceAssignerModule) and ((epoch // 5) % 3) < 2  and epoch > 15:
        #         if model.node_assigner.sigmas.requires_grad:
        #             if verbose:
        #                 print('Setting sigmas to requires_grad=False')
        #             model.node_assigner.requires_grad_(False)
        #     else:
        #         if not model.node_assigner.sigmas.requires_grad:
        #             if verbose:
        #                 print('Setting sigmas to requires_grad=True')
        #             model.node_assigner.requires_grad_(True)
        #         # test_assignments(epoch, model.node_assigner)


        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            X, mask_padded, y, y_prev, lon, lat, dvdz, days, missing_indices, fs, ac, ws, total_moc = move_data_to_device(data, device)

            # if geostrophic_target:
            #     fs = torch.zeros_like(fs)
            #     ac = torch.zeros_like(ac)
            #     ws = torch.zeros_like(ws)
            
            target = total_moc
            if geostrophic_target:
                target = y


            # zero the parameter gradients
            optimizer_all.zero_grad()

            outputs, _ = model(X, mask_padded, lon, lat, dvdz, days, y_prev, missing_indices, fs, ac, ws)
            loss = criterion(outputs, target)
            loss.backward()

            if torch.isnan(loss):
                print('Loss is NaN')
                continue

            optimizer_all.step()
            loss_per_epoch += loss.detach().cpu().item()

            # check if all parameters are finite
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    print('Parameter ', name, ' comtains NaN values')
                    print(param)
                    # raise ValueError('Parameter is not finite')

        lr_scheduler_all.step()

        model.eval()
        total_val_loss = 0.

        with torch.no_grad():
            for data in val_dataloader:
                X, mask_padded, y, y_prev, lon, lat, dvdz, days, missing_indices, fs, ac, ws, total_moc = move_data_to_device(data, device)

                target = total_moc
                if geostrophic_target:
                    target = y

                val_prediction, _ = model(X, mask_padded, lon, lat, dvdz, days, y_prev, missing_indices, fs, ac, ws)
                # val_prediction = val_prediction.detach()
                val_loss = criterion(val_prediction, target).detach().cpu().item()

                total_val_loss += val_loss

        total_val_loss /= len(val_dataloader)


        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            if verbose:
                print('New best model found in epoch: ', epoch, 'with val loss: ', total_val_loss)
            
            # test_assignments(epoch, model.node_assigner)

        if (epoch+1) % 3 == 0 and verbose:
            print(f'Epoch {epoch}, Loss: {loss_per_epoch / len(train_dataloader)}, Val loss: {total_val_loss}, current learning rate: {optimizer_all.param_groups[0]["lr"]:.2E}')

        writer.add_scalars('Loss', {
            'train': loss_per_epoch / len(train_dataloader),
            'validation': total_val_loss}, epoch)

        train_losses.append([loss_per_epoch / len(train_dataloader), total_val_loss])

        if epoch - best_epoch > 15 and best_val_loss < 20:
            break

    return best_model



def add_noise(noise, X):

    for batch_i in range(len(X)):
        for compartment_i in range(len(X[batch_i])):
            X[batch_i][compartment_i] = X[batch_i][compartment_i] + noise * torch.randn_like(X[batch_i][compartment_i])

def make_predictions(dataset, model, temporal_values, mean_total_moc, std_total_moc, noise = 0.0, device = 'cpu', geostrophic_target = False):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=merge_profiles_max_profiles, num_workers=4)

    test_predictions = []
    test_hidden_spaces = []
    test_embedding_space = []
    ground_truth = []

    inner_values_lst = []

    model.eval()
    total_test_loss = 0.

    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    for data in dataloader:
        X, mask_padded, y, y_prev, lon, lat, dvdz, days, missing_indices, fs, ac, ws, total_moc = move_data_to_device(data, device)
        
        # if geostrophic_target:
        #     fs = torch.zeros_like(fs)
        #     ac = torch.zeros_like(ac)
        #     ws = torch.zeros_like(ws)

        target = total_moc
        if geostrophic_target:
            target = y


        if noise > 0:
            add_noise(noise, X)

        test_prediction, inner_values = model(X, mask_padded, lon, lat, dvdz, days, y_prev, missing_indices, fs, ac, ws)
        test_prediction = test_prediction.detach()
        # inner_values_lst.append([v.detach() for v in inner_values] + [total_moc.detach()])
        # test_hidden_space = inner_values[0].detach()
        # embedding_space = inner_values[1].detach()

        test_loss = criterion(test_prediction, target).item()

        total_test_loss += test_loss
        test_predictions.append(test_prediction)
        ground_truth.append(target)
        # test_hidden_spaces.append(test_hidden_space)
        # test_embedding_space.append(embedding_space)

    test_predictions = torch.cat(test_predictions).cpu().numpy()
    ground_truth = torch.cat(ground_truth).cpu().numpy()
    # test_hidden_spaces = torch.cat(test_hidden_spaces).cpu().numpy()
    # test_embedding_space = torch.cat(test_embedding_space).cpu().numpy()
    transport = xr.DataArray(
        test_predictions.squeeze() * std_total_moc.values[()] + mean_total_moc.values[()],
        coords={
            'time': temporal_values
        }, name='dv_dz_times_X'
    )

    gt_tranport = xr.DataArray(
        ground_truth.squeeze() * std_total_moc.values[()] + mean_total_moc.values[()],
        coords={
            'time': temporal_values
        }, name='dv_dz_times_X'
    )

    

    return transport, (test_hidden_spaces, test_embedding_space, gt_tranport, inner_values_lst)