import argparse
from datetime import datetime
import sys
from amoc_reconstruction.reconstruction.trainer import Trainer 
import torch
from amoc_reconstruction.reconstruction.data.profiledataset import ProfileDataset
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from copy import deepcopy
from amoc_reconstruction.reconstruction.model.profile_model import ProfileModelSUSTeR3
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import hvplot.xarray
from pathlib import Path


def main(args):


    

    # Load the argo and umo data
    suffixe = ['_70None', '_2nd_NoneNone']
    ds_argo_merged, t_umo_obs, t_delta = load_merged_argo_dataset_and_tumo_cycles(suffixe, args.time_smoothing)

    # Load dvdz 
    dv_dz_obs_1st = xr.open_dataset('../rapid-geostrophic-reconstruction/data/dv_dz_sim_miss_70None.nc')
    dv_dz_obs_1st = dv_dz_obs_1st.resample(time = args.time_smoothing).mean()
    dv_dz_obs_2nd = xr.open_dataset('../rapid-geostrophic-reconstruction/data/dv_dz_sim_miss_2nd_NoneNone.nc')
    dv_dz_obs_2nd = dv_dz_obs_2nd.resample(time = args.time_smoothing).mean()

    dv_dz_obs_2nd['time'] = dv_dz_obs_2nd['time'] + t_delta

    dv_dz_obs = xr.concat([dv_dz_obs_1st, dv_dz_obs_2nd], dim = 'time')

    ds_argo_merged_10days, t_umo_10days, t_delta = load_merged_argo_dataset_and_tumo_cycles(['_70None', '_2nd_NoneNone'], '90D')

    compartments = [
        ['west_p', -76.74, -70, 'west'],
        ['mar_west_p', -60, -47, 'east'],
        ['mar_east_p', -47, -40, 'west'],
        ['east_p', -30, -13.5, 'east']
    ]

    # VIKING contains more detailed information in the upper layers as the Argo dataset
    # Argo only measures each 20 meters while VIKING has measurments at 0, 3, 9, 12 ... meters

    boundaries = np.arange(-10000, 1, 20) + 10
    depth_grouped = ds_argo_merged_10days.groupby_bins('z', boundaries).mean()
    depth_grouped = depth_grouped.where(ds_argo_merged_10days.z.groupby_bins('z', boundaries).count() > 0)


    ds_argo_merged_10days = depth_grouped.assign_coords(z_bins = ('z_bins', np.arange(-10000, 1, 20)[1:])).rename({'z_bins': 'z'}).sel(z = ds_argo_merged_10days.z, method = 'nearest').assign_coords(z = ds_argo_merged_10days.z).transpose('time', 'pos', 'z')

    # We only take the first 2000 meters depth
    ds_argo_merged_10days = ds_argo_merged_10days.where(ds_argo_merged_10days.z > -2000, drop = True)

    ds_argo_merged_10days = ds_argo_merged_10days.where(ds_argo_merged_10days.salinity > 0)
    # ds_argo_merged_10days = ds_argo_merged_10days.where((ds_argo_merged_10days.z <= -1500) | (ds_argo_merged_10days.z > -100) , drop = True)
    # ds_argo_merged_10days = ds_argo_merged_10days.where((ds_argo_merged_10days.z <= -1500) , drop = True)

    ds_argo_merged_10days = ds_argo_merged_10days.sortby('z')





    test_indices = t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time)))).sel(time = slice('2030-01-01', None)).ti.values

    val_indices =  t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time)))).sel(time = slice('2020-01-01', '2030-01-01')).ti.values


    train_mask = ~np.isin(np.arange(t_umo_10days.time.shape[0]), np.concatenate((val_indices, test_indices)))

    train_indices = np.arange(t_umo_10days.time.shape[0])[train_mask]


    rho_mean = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).mean(['time', 'pos']).rho
    temperature_mean = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).mean(['time', 'pos']).temperature
    salinity_mean = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).mean(['time', 'pos']).salinity

    if args.load_std:
        temperature_std = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_temperature_std_argo.nc').temperature
        salinity_std = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_salinity_std_argo.nc').salinity
        rho_std = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_rho_std_argo.nc').rho
        std_transport = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_transport_std_argo.nc').dv_dz_times_X
    else:
        rho_std = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).std(['time', 'pos']).rho
        temperature_std = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).std(['time', 'pos']).temperature
        salinity_std = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).std(['time', 'pos']).salinity
        std_transport = t_umo_10days.isel(time = train_indices).std('time').dv_dz_times_X




    def standardize_rho(ds_pos, rho_mean=None, rho_std=None):
        ds_pos = deepcopy(ds_pos)
        ds_pos["rho"] = (ds_pos["rho"] - rho_mean) / rho_std
        return ds_pos

    def standardize_temperature(ds_pos, temperature_mean=None, temperature_std=None):
        ds_pos = deepcopy(ds_pos)
        ds_pos["temperature"] = (ds_pos["temperature"] - temperature_mean) / temperature_std
        return ds_pos

    def standardize_salinity(ds_pos, salinity_mean=None, salinity_std=None):
        ds_pos = deepcopy(ds_pos)
        ds_pos["salinity"] = (ds_pos["salinity"] - salinity_mean) / salinity_std
        return ds_pos


    ds_argo_merged_10days_std = standardize_rho(
        ds_argo_merged_10days,
        rho_mean=rho_mean, 
        rho_std=rho_std,
    )

    ds_argo_merged_10days_std = standardize_temperature(
        ds_argo_merged_10days_std,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
    )

    ds_argo_merged_10days_std = standardize_salinity(
        ds_argo_merged_10days_std,
        salinity_mean=salinity_mean,
        salinity_std=salinity_std,
    )

    ds_argo_merged_10days_std = ds_argo_merged_10days_std.fillna(0)

    nn_feature_ds = xr.concat(
        [
            ds_argo_merged_10days_std.rho, 
            ds_argo_merged_10days_std.temperature, 
            ds_argo_merged_10days_std.salinity
            ], dim = 'feature'
    ).stack(nn_feature = ['feature', 'z'])
    train_X_scaled = nn_feature_ds.isel(time = train_indices).values
    val_X_scaled = nn_feature_ds.isel(time = val_indices).values
    test_X_scaled = nn_feature_ds.isel(time = test_indices).values


    train_X_mask = ds_argo_merged_10days.isel(time = train_indices).profile_mask.fillna(0).astype(bool).values
    val_X_mask = ds_argo_merged_10days.isel(time = val_indices).profile_mask.fillna(0).astype(bool).values
    test_X_mask = ds_argo_merged_10days.isel(time = test_indices).profile_mask.fillna(0).astype(bool).values

    train_X_lon = ds_argo_merged_10days.isel(time = train_indices).lon.values
    val_X_lon = ds_argo_merged_10days.isel(time = val_indices).lon.values
    test_X_lon = ds_argo_merged_10days.isel(time = test_indices).lon.values

    offset_days = (
        (
            ds_argo_merged_10days.time_1d -
            ds_argo_merged_10days.time
        )
        .fillna(0).dt.days 
    )

    train_X_days = offset_days.isel(time = train_indices).values
    val_X_days = offset_days.isel(time = val_indices).values
    test_X_days = offset_days.isel(time = test_indices).values

    train_y = t_umo_10days.isel(time = train_indices).dv_dz_times_X
    val_y = t_umo_10days.isel(time = val_indices).dv_dz_times_X
    test_y = t_umo_10days.isel(time = test_indices).dv_dz_times_X

    mean_transport  = train_y.mean('time')

    train_y_scaled = (train_y - mean_transport) / std_transport
    val_y_scaled = (val_y - mean_transport) / std_transport
    test_y_scaled = (test_y - mean_transport) / std_transport

    train_y_scaled = train_y_scaled.fillna(0).values
    val_y_scaled = val_y_scaled.fillna(0).values
    test_y_scaled = test_y_scaled.fillna(0).values

    # dv_dz_obs = xr.open_dataset('../rapid-geostrophic-reconstruction/data/dv_dz_sim_miss_70None.nc')
    # dv_dz_obs = dv_dz_obs.resample(time = time_smoothing).mean()
    dv_dz_moorings = dv_dz_obs.sel(time = ds_argo_merged_10days.time).where(dv_dz_obs.z <= -2000, drop = True)

    mean_dv_dz_moorings = dv_dz_moorings.isel(time = train_indices).mean(['time']).dv_dz_times_X
    std_dv_dz_moorings = dv_dz_moorings.isel(time = train_indices).std(['time']).dv_dz_times_X

    train_dv_dz_moorings_scaled = ((dv_dz_moorings.isel(time = train_indices).dv_dz_times_X - mean_dv_dz_moorings) / std_dv_dz_moorings).fillna(0).values
    val_dv_dz_moorings_scaled = ((dv_dz_moorings.isel(time = val_indices).dv_dz_times_X - mean_dv_dz_moorings) / std_dv_dz_moorings).fillna(0).values
    test_dv_dz_moorings_scaled = ((dv_dz_moorings.isel(time = test_indices).dv_dz_times_X - mean_dv_dz_moorings) / std_dv_dz_moorings).fillna(0).values

    train_dl, val_dl, test_dl, datasets = create_torch_dataset(train_X_scaled, train_y, train_X_mask, train_X_lon, train_dv_dz_moorings_scaled, train_X_days, val_X_scaled, val_y, val_X_mask, val_X_lon, val_dv_dz_moorings_scaled, val_X_days, test_X_scaled, test_y, test_X_mask, test_X_lon, test_dv_dz_moorings_scaled, test_X_days, compartments, args.train_batch_size, args.time_smoothing)


    model = ProfileModelSUSTeR3(
        train_X_scaled.shape[2],
        args.n_compartments, 
        args.n_embedding,
        train_dv_dz_moorings_scaled.shape[1],
        dv_dz_obs.z,
        args.device,
        dv_dz_obs, None
    )

    run_directory = Path(f'../rapid-geostrophic-reconstruction/runs/{args.run_name}')

    writer = SummaryWriter(run_directory)


    trainer = Trainer(model, args, writer)

    best_model = trainer.train(train_dl, val_dl, args.epochs)

    evaluate_model(trainer, test_dl, ds_argo_merged_10days_std, t_umo_10days, test_indices, args.run_name, run_directory)

    torch.save(best_model.state_dict(), run_directory / 'best_model.pt')

def create_torch_dataset(train_X_scaled, train_y, train_X_mask,
                         train_X_lon, train_dv_dz_moorings_scaled, train_X_days, val_X_scaled, val_y, val_X_mask, val_X_lon, val_dv_dz_moorings_scaled, val_X_days, test_X_scaled, test_y, test_X_mask, test_X_lon, test_dv_dz_moorings_scaled, test_X_days, compartments, train_batch_size, time_smoothing):
    train_dataset = ProfileDataset(train_X_scaled, train_y, train_X_mask, train_X_lon, train_dv_dz_moorings_scaled, train_X_days, compartments, time_smoothing)
    val_dataset = ProfileDataset(val_X_scaled, val_y, val_X_mask, val_X_lon, val_dv_dz_moorings_scaled, val_X_days, compartments, time_smoothing)
    test_dataset = ProfileDataset(test_X_scaled, test_y, test_X_mask, test_X_lon, test_dv_dz_moorings_scaled, test_X_days, compartments, time_smoothing)
        

    def merge_profiles(data):
        """
            X: is a list and each entry is a list of compartments in which the profiles are stored with variable length
        """
        X, y, lon, dv_dz, days = zip(*data)
        X = [[torch.tensor(x_compartment).float() for x_compartment in x_sample] for x_sample in X]
        y = torch.tensor(np.array(y)).float()
        lon = [[torch.tensor(lon_compartment).float() for lon_compartment in lon_sample] for lon_sample in lon]
        dv_dz = [torch.tensor(dv_dz_sample).float() for dv_dz_sample in dv_dz]
        days = [[torch.tensor(days_compartment).float() for days_compartment in day_sample] for day_sample in days]

        

        return X, y, lon, dv_dz, days

    dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=merge_profiles, num_workers=4)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=merge_profiles, num_workers=4)
    test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=merge_profiles, num_workers=4)

    return dl, val_dl, test_dl, (train_dataset, val_dataset, test_dataset)

def load_merged_argo_dataset_and_tumo(time_smoothing, lat_bounds = None, suffix = None):
    ds_argo_merged = xr.open_dataset(f'../rapid-geostrophic-reconstruction/data/ds_argo_{time_smoothing}_viking{suffix}.nc')
    t_umo_obs = xr.open_dataset(f'../rapid-geostrophic-reconstruction/data/t_umo_sim_miss{suffix}.nc')
    t_umo_obs['time'] = t_umo_obs.time.dt.floor('D')
    t_umo_obs = t_umo_obs.resample(time = time_smoothing).mean()

    ds_argo_merged = ds_argo_merged.sel(time = t_umo_obs.time.data)#, method = 'nearest')
    ds_argo_merged = ds_argo_merged.isel(z = slice(None, None, -1))

    if lat_bounds is not None:
        ds_argo_merged = ds_argo_merged.where((ds_argo_merged.lat >= lat_bounds[0]) & (ds_argo_merged.lat <= lat_bounds[1]))

    return ds_argo_merged, t_umo_obs


def load_merged_argo_dataset_and_tumo_cycles(suffixe, time_smoothing):
    ds_argo_merged = None
    t_umo_obs = None
    assert len(suffixe) == 2, 'Longer? Think on the t_delta '

    for suffix in suffixe:
        ds_argo_merged_cycle, t_umo_obs_cycle = load_merged_argo_dataset_and_tumo(time_smoothing, suffix= suffix)

        if ds_argo_merged is None:
            ds_argo_merged = ds_argo_merged_cycle
            t_umo_obs = t_umo_obs_cycle
        else:
            t_delta = ds_argo_merged.time.max() - ds_argo_merged_cycle.time.min() + pd.Timedelta(time_smoothing)
            ds_argo_merged_cycle['time'] = ds_argo_merged_cycle['time'] + t_delta 
            t_umo_obs_cycle['time'] = t_umo_obs_cycle['time'] + t_delta
            ds_argo_merged_cycle['time_1d'] = ds_argo_merged_cycle['time_1d'] + t_delta

            ds_argo_merged = xr.concat([ds_argo_merged, ds_argo_merged_cycle], dim = 'time')
            t_umo_obs = xr.concat([t_umo_obs, t_umo_obs_cycle], dim = 'time')

    return ds_argo_merged, t_umo_obs, t_delta


def evaluate_model(trainer, test_dl, ds_argo_merged_10days_std, t_umo_obs, test_indices, run_name, output_dir):

    test_loss, predictions = trainer.evaulate_model_on_dataloader(test_dl)


    transport = xr.DataArray(
    predictions,
    coords={
        'time':ds_argo_merged_10days_std.isel(time = test_indices).time
    }, name='dv_dz_times_X')


    const_prediction_MAE = np.abs(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X - t_umo_obs.sel(time = ds_argo_merged_10days_std.isel(time = test_indices).time, method = "nearest").dv_dz_times_X.mean()).mean().values

    mae_error = mean_absolute_error(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, transport.values)

    print(f'R2: {r2_score(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, transport.values)}')
    print(f'MAE: {mae_error}; const MAE {const_prediction_MAE}', f' improvement percentage:  {(const_prediction_MAE - mae_error) / const_prediction_MAE * 100:.2f}%')
    print(f'MSE: {mean_squared_error(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, transport.values)}')

    ## scatter plot of the predictions
    plt.scatter(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, transport.values)
    # plt.scatter(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, t_merged_argo_2000.sel(time = transport.time, method = 'nearest').values)
    plt.xlabel('Rapid')
    plt.ylabel('NN')
    plt.plot([-35, -5], [-35, -5], 'r--')


    lr = LinearRegression().fit(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values.reshape(-1, 1), transport.values.reshape(-1, 1))
    in_X = np.linspace(-35, -5, 100).reshape(-1, 1)
    plt.plot(in_X, lr.predict(in_X), 'k--')
    # lr = LinearRegression().fit(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values.reshape(-1, 1), t_merged_argo_2000.sel(time = transport.time, method = 'nearest').values.reshape(-1, 1))
    in_X = np.linspace(-35, -5, 100).reshape(-1, 1)
    plt.plot(in_X, lr.predict(in_X), 'g--')
    plt.savefig(f'{output_dir}/scatter_nn_rapid.png')
    plt.close()

    # plot the time series
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1,1,1)

    # calc = t_merged_argo_2000.sel(time = transport.time, method = 'nearest')
    # ax.plot(calc.time, calc.values, label = 'Geostrohphic calculation from Argos', alpha = .5, color = 'black')
    rapid = t_umo_obs.sel(time = transport.time, method = 'nearest')
    ax.plot(transport.time, transport.values, label = 'Neural Network geostrophic Argo reconstruction', linewidth = 2, zorder = 20)
    ax.plot(rapid.time, rapid.dv_dz_times_X.values, label = 'Geostrophic Rapid measurements', linewidth = 2)

    kpi_string = f'R2 {r2_score(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, transport.values)*100:.2f}%; MAE {mae_error:.2f}; MSE {mean_squared_error(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, transport.values):.2f}'

    ax.text(0.05, 0.1, kpi_string, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')

    plt.legend()
    ax.set_ylabel('Transport [Sv]')
    ax.set_xlabel('Time')
    plt.title('AMOC Reconstruction at Rapid Latitude 26.5°N by Argo profiles')
    fig.tight_layout()
    plt.savefig(f'{output_dir}/argo_nn_reconstruction.png')



if __name__ == '__main__':
    argu = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    
    argu.add_argument('--load_std', type=str2bool, default=False)
    argu.add_argument('--device', type=str, default='cpu')

    argu.add_argument('--n_compartments', type=int, default=7)
    argu.add_argument('--n_embedding', type=int, default=64)
    argu.add_argument('--epochs', type=int, default=200)
    argu.add_argument('--run_name', type=str, default=f'run_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    argu.add_argument('--train_batch_size', type=int, default=4)

    argu.add_argument('--time_smoothing', type=str, default='90D')

    argu.add_argument('--loss_method', type=str, default='mse')
    argu.add_argument('--lr', type=float, default=1e-4)
    argu.add_argument('--weight_decay', type=float, default=1e-7)

    argu.add_argument('--scheduler', type=str, default='cosine')



    main(argu.parse_args())