import xarray as xr
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
from copy import deepcopy
import torch
import pandas as pd


class ProfileDataset(Dataset):
    def __init__(self, X, y, y_prev, mask, lon, lat, dv_dz, days, compartments, missing_indices,
                 fs, ac, ws, total_moc, time_smoothing, lat_bounds,
                 using_transport_from_previous_year,
                using_missing_indices,
                use_deep_dvdz, global_indices ):
        self.X = X
        self.y = y
        self.y_prev = y_prev
        self.mask = mask
        self.lon = lon
        self.lat = lat
        self.days = days
        self.compartment = compartments 
        self.dv_dz = dv_dz
        self.missing_indices = missing_indices
        self.using_transport_from_previous_year = using_transport_from_previous_year
        self.using_missing_indices = using_missing_indices
        self.use_deep_dvdz = use_deep_dvdz
        self.global_indices = global_indices  
        self.time_smoothing = time_smoothing      
        self.lat_bounds = lat_bounds

        self.fs = fs
        self.ac = ac
        self.ws = ws
        self.total_moc = total_moc

        self.len_compartments = len(compartments)

        self.X_per_compartment = []
        self.lon_per_compartment = []
        self.lat_per_compartment = []
        self.dv_dz_per_compartment = []
        self.days_per_compartment = []
        self.missing_indices_per_compartment = []

        self.min_longitude = min([comp[1] for comp in compartments])
        self.max_longitude = max([comp[2] for comp in compartments])



        self.lon_scaled = (self.lon - self.min_longitude) / (self.max_longitude - self.min_longitude)
        self.lat_scaled = (self.lat - lat_bounds[0]) / (lat_bounds[1] - lat_bounds[0])  

        self.days_scaled = (self.days + (pd.Timedelta(time_smoothing).days / 2)) / pd.Timedelta(time_smoothing).days

        self.use_fc_input = True
        self.use_ac_input = True
        self.use_ws_input = True
        self.use_ar_input = True
        
        for i in range(len(self)):
            mask = self.mask[i]


            compartment_X_per_sample = []
            compartment_lon_per_sample = []
            compartment_lat_per_sample = []
            compartment_days_per_sample = []

            for j in range(self.len_compartments):

                # get all profile indices that are in the compartment and in the mask 
                profile_indices = np.where((self.lon[i] > self.compartment[j][1]) & (self.lon[i] < self.compartment[j][2]) & mask.any(axis = -1))[0]

                if len(profile_indices) < 1:
                    compartment_X_per_sample.append(np.empty(0))
                    compartment_lon_per_sample.append(np.empty(0))
                    compartment_days_per_sample.append(np.empty(0))
                    continue

                lon_compartments = self.lon_scaled[i][profile_indices]
                sorted_indices_longitude = lon_compartments.argsort()


                # If I have the sequence -5, -15, 5 the argsort will return [1, 0, 2]
                # As we want to have those profiles as last that are closes to the mooring due to the vanishing gradient 
                # For West moorings the smallest value has to be the last one
                if self.compartment[j][3] == 'west':
                    sorted_indices_longitude = sorted_indices_longitude[::-1]
                
                profile_indices = profile_indices[sorted_indices_longitude]

                compartment_X_per_sample.append(self.X[i][ profile_indices])
                compartment_lon_per_sample.append(self.lon_scaled[i][profile_indices])
                compartment_lat_per_sample.append(self.lat_scaled[i][profile_indices])
                compartment_days_per_sample.append(self.days_scaled[i][profile_indices])
            

            self.X_per_compartment.append(compartment_X_per_sample)
            self.lon_per_compartment.append(compartment_lon_per_sample)
            self.lat_per_compartment.append(compartment_lat_per_sample)
            self.dv_dz_per_compartment.append(self.dv_dz[i])
            self.missing_indices_per_compartment.append(self.missing_indices[i])
            self.days_per_compartment.append(compartment_days_per_sample)


    def __len__(self):
        return self.X.shape[0] 

    def __getitem__(self, idx):

        if self.using_transport_from_previous_year:
            y_prev = self.y_prev[idx]
        else:
            y_prev = np.zeros_like(self.y_prev[idx])

        if self.using_missing_indices:
            missing_indices_per_compartment = self.missing_indices_per_compartment[idx] 
        else:
            missing_indices_per_compartment = np.zeros_like(self.missing_indices_per_compartment[idx])
        
        if self.use_deep_dvdz:
            dv_dz = self.dv_dz_per_compartment[idx]
        else:
            dv_dz = np.zeros_like(self.dv_dz_per_compartment[idx])

        
        if self.use_fc_input:
            fs = self.fs[idx]
        else:
            fs = np.zeros_like(self.fs[idx])

        if self.use_ac_input:
            ac = self.ac[idx]
        else:
            ac = np.zeros_like(self.ac[idx])

        if self.use_ws_input:
            ws = self.ws[idx]
        else:
            ws = np.zeros_like(self.ws[idx])

        if self.use_ar_input:
            X_per_comp = self.X_per_compartment[idx]
            lon_per_comp   = self.lon_per_compartment[idx]
            lat_per_comp   = self.lat_per_compartment[idx]
            days_per_comp  = self.days_per_compartment[idx]
        else:
            X_per_comp = [np.zeros_like(xi) for xi in self.X_per_compartment[idx]]
            lon_per_comp = [np.zeros_like(li) for li in self.lon_per_compartment[idx]]
            lat_per_comp = [np.zeros_like(li) for li in self.lat_per_compartment[idx]]
            days_per_comp = [np.zeros_like(di) for di in self.days_per_compartment[idx]]

        return X_per_comp, self.y[idx], y_prev, lon_per_comp, lat_per_comp, dv_dz, days_per_comp, missing_indices_per_compartment, fs, ac, ws, self.total_moc[idx]


        # return self.X_per_compartment[idx], self.y[idx], y_prev, self.lon_per_compartment[idx], self.lat_per_compartment[idx], dv_dz, self.days_per_compartment[idx], missing_indices_per_compartment, self.fs[idx], self.ac[idx], self.ws[idx], self.total_moc[idx]

# def _load_merged_argo_dataset_and_tumo_obs(time_smoothing, lat_bounds = None):
#     ds_argo_merged = xr.open_dataset(dataset_path / f'../../ds_argo_obs.nc')
#     t_umo_obs = xr.open_dataset(dataset_path / ref_folder/ f't_umo_obs.nc')

#     t_umo_obs = t_umo_obs.resample(time = time_smoothing).mean()

#     ds_argo_merged = ds_argo_merged.sel(time = t_umo_obs.time.data)#, method = 'nearest')
#     ds_argo_merged = ds_argo_merged.isel(z = slice(None, None, -1))

#     if lat_bounds is not None:
#         ds_argo_merged = ds_argo_merged.where((ds_argo_merged.lat >= lat_bounds[0]) & (ds_argo_merged.lat <= lat_bounds[1]))

#     return ds_argo_merged, t_umo_obs


def load_merged_argo_dataset_and_tumo(
        time_smoothing, dataset_path, ref_folder, total_moc_path, antilles_current_path, florida_current_path, wind_stress_path, 
        missing_values_in_target = False, lat_bounds = None, suffix = None, deep_argo = False):
    ds_argo_merged = xr.open_dataset(dataset_path / f'ds_{"deep_" if deep_argo else ""}argo_viking_KFS003-{suffix}.nc')
    t_umo_obs = xr.open_dataset(dataset_path / ref_folder / f't_umo_sim{"_miss" if missing_values_in_target else ""}_KFS003-{suffix}.nc')

    ds_pos = xr.open_dataset(dataset_path / ref_folder/ f'ds_pos_sim{"_miss" if missing_values_in_target else ""}_KFS003-{suffix}.nc')
    dv_dz = xr.open_dataset(dataset_path / ref_folder / f'dv_dz_sim{"_miss" if missing_values_in_target else ""}_fillup_KFS003-{suffix}.nc')


    total_moc = xr.open_dataset(total_moc_path / f'cdf_moc_265N_KFS003-{suffix}.nc').load()
    antilles_current = xr.open_dataset(antilles_current_path / f'antilles_current_KFS003-{suffix}.nc').load()
    florida_current = xr.open_dataset(florida_current_path / f'florida_current_KFS003-{suffix}.nc').load()
    wind_stress = xr.open_dataset(wind_stress_path / f'sozotaux_2527N_8010W_KFS003-{suffix}.nc').load()

    # t_umo_obs['time'] = t_umo_obs.time.dt.floor('D')
    # t_umo_obs = t_umo_obs.resample(time = time_smoothing).mean()

    t_umo_obs = t_umo_obs.groupby(t_umo_obs.time.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})
    ds_pos = ds_pos.groupby(ds_pos.time.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})
    dv_dz = dv_dz.groupby(dv_dz.time.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})

    total_moc = total_moc.groupby(total_moc.time_counter.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})
    antilles_current = antilles_current.groupby(antilles_current.time_counter.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})
    florida_current = florida_current.groupby(florida_current.time_counter.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})
    wind_stress = wind_stress.groupby(wind_stress.time_counter.dt.floor(time_smoothing)).mean().rename({'floor': 'time'})

    t_umo_obs = t_umo_obs.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))
    ds_pos = ds_pos.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))
    dv_dz = dv_dz.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))

    total_moc = total_moc.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))
    antilles_current = antilles_current.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))
    florida_current = florida_current.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))
    wind_stress = wind_stress.sel(time = slice(ds_argo_merged.time.min(), ds_argo_merged.time.max()))
    
    ds_argo_merged = ds_argo_merged.sel(time = t_umo_obs.time.data)
    ds_argo_merged = ds_argo_merged.isel(z = slice(None, None, -1))

    if lat_bounds is not None:
        ds_argo_merged = ds_argo_merged.where((ds_argo_merged.lat >= lat_bounds[0]) & (ds_argo_merged.lat <= lat_bounds[1]))

    return ds_argo_merged, t_umo_obs, ds_pos, dv_dz, total_moc, antilles_current, florida_current, wind_stress


def load_merged_argo_dataset_and_tumo_cycles(suffixe, dataset_path, ref_folder, total_moc_path, antilles_current_path, florida_current_path, wind_stress_path,  time_smoothing, deep_argo, missing_values_in_target):
    ds_argo_merged = None
    t_umo_obs = None
    ds_pos_sim = None
    dv_dz_obs = None

    total_moc = None
    antilles_current = None
    florida_current = None
    wind_stress = None
    
    # assert len(suffixe) <= 2, 'Longer? Think on the t_delta '

    for suffix in suffixe:
        ds_argo_merged_cycle, t_umo_obs_cycle, ds_pos_cycle, dv_dz_cycle, total_moc_cycle, antilles_current_cycle, florida_current_cycle, wind_stress_cycle = load_merged_argo_dataset_and_tumo(
            time_smoothing, 
            dataset_path,
            ref_folder,
            total_moc_path,
            antilles_current_path,
            florida_current_path,
            wind_stress_path,
            missing_values_in_target= missing_values_in_target,
            suffix= suffix, 
            deep_argo = deep_argo
        )

        if ds_argo_merged is None:
            if len(suffixe) > 4:
                referenced_to_1800 = (ds_argo_merged_cycle.time.min() - np.datetime64('1800-01-01'))
                print('Moving the time to 1800 with an offset of ', referenced_to_1800)
                print('Test if we can substract this from the test_period_start', np.datetime64('2005-01-01') - referenced_to_1800)
                ds_argo_merged_cycle['time'] = ds_argo_merged_cycle['time'] - referenced_to_1800
                t_umo_obs_cycle['time'] = t_umo_obs_cycle['time'] - referenced_to_1800
                ds_pos_cycle['time'] = ds_pos_cycle['time'] - referenced_to_1800
                dv_dz_cycle['time'] = dv_dz_cycle['time'] - referenced_to_1800
                total_moc_cycle['time'] = total_moc_cycle['time'] - referenced_to_1800
                antilles_current_cycle['time'] = antilles_current_cycle['time'] - referenced_to_1800
                florida_current_cycle['time'] = florida_current_cycle['time'] - referenced_to_1800
                wind_stress_cycle['time'] = wind_stress_cycle['time'] - referenced_to_1800
                ds_argo_merged_cycle['time_1d'] = ds_argo_merged_cycle['time_1d'] - referenced_to_1800
            else:
                referenced_to_1800 = None 
            
            
            ds_argo_merged = ds_argo_merged_cycle
            t_umo_obs = t_umo_obs_cycle
            ds_pos_sim = ds_pos_cycle
            dv_dz_obs = dv_dz_cycle
            total_moc = total_moc_cycle
            antilles_current = antilles_current_cycle
            florida_current = florida_current_cycle
            wind_stress = wind_stress_cycle

            
        else:
            t_delta = ds_argo_merged.time.max() - ds_argo_merged_cycle.time.min() + pd.Timedelta(time_smoothing)
            ds_argo_merged_cycle['time'] = ds_argo_merged_cycle['time'] + t_delta 
            t_umo_obs_cycle['time'] = t_umo_obs_cycle['time'] + t_delta
            ds_pos_cycle['time'] = ds_pos_cycle['time'] + t_delta
            dv_dz_cycle['time'] = dv_dz_cycle['time'] + t_delta
            total_moc_cycle['time'] = total_moc_cycle['time'] + t_delta
            antilles_current_cycle['time'] = antilles_current_cycle['time'] + t_delta
            florida_current_cycle['time'] = florida_current_cycle['time'] + t_delta
            wind_stress_cycle['time'] = wind_stress_cycle['time'] + t_delta
            ds_argo_merged_cycle['time_1d'] = ds_argo_merged_cycle['time_1d'] + t_delta

            ds_argo_merged = xr.concat([ds_argo_merged, ds_argo_merged_cycle], dim = 'time')
            t_umo_obs = xr.concat([t_umo_obs, t_umo_obs_cycle], dim = 'time')

            ds_pos_sim = xr.concat([ds_pos_sim, ds_pos_cycle], dim = 'time')
            dv_dz_obs = xr.concat([dv_dz_obs, dv_dz_cycle], dim = 'time')

            total_moc = xr.concat([total_moc, total_moc_cycle], dim = 'time')
            antilles_current = xr.concat([antilles_current, antilles_current_cycle], dim = 'time')
            florida_current = xr.concat([florida_current, florida_current_cycle], dim = 'time')
            wind_stress = xr.concat([wind_stress, wind_stress_cycle], dim = 'time')

    return ds_argo_merged, t_umo_obs, ds_pos_sim, dv_dz_obs, t_delta if len(suffixe) > 1 else None, total_moc, antilles_current, florida_current, wind_stress, referenced_to_1800


def filter_ds_argo_data(ds_argo_merged_10days, deep_argo = False):

    

    # VIKING contains more detailed information in the upper layers as the Argo dataset
    # Argo only measures each 20 meters while VIKING has measurments at 0, 3, 9, 12 ... meters

    boundaries = np.arange(-10000, 1, 20) + 10
    depth_grouped = ds_argo_merged_10days.groupby_bins('z', boundaries).mean()
    depth_grouped = depth_grouped.where(ds_argo_merged_10days.z.groupby_bins('z', boundaries).count() > 0)


    ds_argo_merged_10days = depth_grouped.assign_coords(z_bins = ('z_bins', np.arange(-10000, 1, 20)[1:])).rename({'z_bins': 'z'}).sel(z = ds_argo_merged_10days.z, method = 'nearest').assign_coords(z = ds_argo_merged_10days.z).transpose('time', 'pos', 'z')

    # We only take the first 2000 meters depth
    if not deep_argo:
        ds_argo_merged_10days = ds_argo_merged_10days.where(ds_argo_merged_10days.z > -2000, drop = True)

    ds_argo_merged_10days = ds_argo_merged_10days.where(ds_argo_merged_10days.salinity > 0)
    # ds_argo_merged_10days = ds_argo_merged_10days.where((ds_argo_merged_10days.z <= -1500) | (ds_argo_merged_10days.z > -100) , drop = True)
    # ds_argo_merged_10days = ds_argo_merged_10days.where((ds_argo_merged_10days.z <= -1500) , drop = True)

    ds_argo_merged_10days = ds_argo_merged_10days.sortby('z')

    return ds_argo_merged_10days


def split_dataset_into_training_validation_testing_profile_datasets(
        random_split_data, test_start_year, test_end_year, validation_years_on_each_side,
        t_umo_10days, ds_argo_merged_10days, add_tmp, add_sal, deep_argo, load_std,
        total_moc_ds,
        florida_current,
        antilles_current,
        wind_stress,
        dv_dz_obs,
        missing_values_in_target,
        ds_pos_sim,
        compartments,
        time_smoothing,
        lat_bounds,
        using_transport_from_previous_year,
        using_missing_indices,
        use_deep_dvdz, 
        backwards_timeshift = None
):

    # Prepare a dataset by 1) splitting the data, 2) scaling the input data 
    train_slice_pos = 'start'

    train_ratio = .6
    val_ratio = .15


    # t_umo_10days = t_umo_10days.sel(time = slice(None, '2020-01-01'))
    # ds_argo_merged_10days = ds_argo_merged_10days.sel(time = slice(None, '2020-01-01'))






    if random_split_data:

        indices = np.arange(t_umo_10days.time.shape[0])
        np.random.shuffle(indices)

        if train_slice_pos == 'start':
            train_indices = indices[:int(train_ratio * indices.shape[0])]
            val_indices = indices[int(train_ratio * indices.shape[0]):int((train_ratio + val_ratio) * indices.shape[0])]
            test_indices = indices[int((train_ratio + val_ratio) * indices.shape[0]):]
        elif train_slice_pos == 'end':
            train_indices = indices[-int(train_ratio * indices.shape[0]):]
            val_indices = indices[-int((train_ratio + val_ratio) * indices.shape[0]): -int(train_ratio * indices.shape[0])]
            test_indices = indices[:-int((train_ratio + val_ratio) * indices.shape[0])]
    else:
        if backwards_timeshift is None:
            test_indices = t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time)))).sel(time = slice(f'{test_start_year}-01-01', f'{test_end_year}-01-01')).ti.values
            valid_indices_1 =  t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time)))).sel(time = slice(f'{test_start_year - validation_years_on_each_side}-01-01', f'{test_start_year}-01-01')).ti.values
            valid_indices_2 =  t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time)))).sel(time = slice(f'{test_end_year}-01-01',f'{test_end_year + validation_years_on_each_side}-01-01')).ti.values
            val_indices = np.concatenate([valid_indices_1, valid_indices_2])
        else:
            test_indices = t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time))))\
            .sel(time = slice(
                    np.datetime64(f'{test_start_year}-01-01') - backwards_timeshift, 
                    np.datetime64(f'{test_end_year}-01-01') - backwards_timeshift)
                ).ti.values
            valid_indices_1 =  t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time))))\
                .sel(time = slice(
                    np.datetime64(f'{test_start_year - validation_years_on_each_side}-01-01') -backwards_timeshift, 
                    np.datetime64(f'{test_start_year}-01-01') - backwards_timeshift)
                ).ti.values
            valid_indices_2 =  t_umo_10days.assign_coords(ti = ('time', np.arange(len(t_umo_10days.time))))\
                .sel(time = slice(
                    np.datetime64(f'{test_end_year}-01-01') - backwards_timeshift,
                    np.datetime64(f'{test_end_year + validation_years_on_each_side}-01-01') - backwards_timeshift)
                ).ti.values
            val_indices = np.concatenate([valid_indices_1, valid_indices_2])





        train_mask = ~np.isin(np.arange(t_umo_10days.time.shape[0]), np.concatenate((val_indices, test_indices)))
        train_indices = np.arange(t_umo_10days.time.shape[0])[train_mask]

        print(len(train_indices), len(val_indices), len(test_indices))



    rho_mean = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).mean(['time', 'pos']).rho
    temperature_mean = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).mean(['time', 'pos']).temperature
    salinity_mean = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).mean(['time', 'pos']).salinity

    if load_std:
        temperature_std = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_temperature_std_argo.nc').temperature
        salinity_std = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_salinity_std_argo.nc').salinity
        rho_std = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_rho_std_argo.nc').rho
        std_transport = xr.open_dataset('../rapid-geostrophic-reconstruction/data/train_transport_std_argo.nc').dv_dz_times_X
    else:
        rho_std = ds_argo_merged_10days.where(ds_argo_merged_10days.profile_mask).isel(time = train_indices).std(['time', 'pos']).rho
        ## Making the changes in the deeper layers not that significant
        # grouped_std = rho_std.groupby_bins('z', [-2000, -800,-200, -100,-50, 0], labels = [-1400,-500,-150,-75,-25]).mean()
        # rho_std = grouped_std.sel(z_bins = rho_std.z, method = 'nearest')


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

    # Detrend the rho data from the training indices
    
    # ds_argo_merged_10days_std = ds_argo_merged_10days_std.isel(z = [2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28])

    ds_argo_merged_10days_std = ds_argo_merged_10days_std.fillna(0)

    feature_list = [ds_argo_merged_10days_std.rho]

    if add_tmp:
        feature_list.append(ds_argo_merged_10days_std.temperature)
    if add_sal:
        feature_list.append(ds_argo_merged_10days_std.salinity)

    nn_feature_ds = xr.concat(
        feature_list, dim = 'feature'
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

    train_X_lat = ds_argo_merged_10days.isel(time = train_indices).lat.values
    val_X_lat = ds_argo_merged_10days.isel(time = val_indices).lat.values
    test_X_lat = ds_argo_merged_10days.isel(time = test_indices).lat.values

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

    train_y_prev = t_umo_10days.isel(
        time = xr.DataArray(train_indices.reshape(-1, 1) - (np.arange(3) + 1).reshape(1,-1))
    ).dv_dz_times_X.values
    val_y_prev = t_umo_10days.isel(
        time = xr.DataArray(val_indices.reshape(-1, 1) - (np.arange(3) + 1).reshape(1,-1))
    ).dv_dz_times_X.values
    test_y_prev = t_umo_10days.isel(
        time = xr.DataArray(test_indices.reshape(-1, 1) - (np.arange(3) + 1).reshape(1,-1))
    ).dv_dz_times_X.values

    total_moc_max = total_moc_ds.sel(depthw = -1000, method='nearest').zomsfatl

    train_total_moc = total_moc_max.isel(time = train_indices)
    val_total_moc = total_moc_max.isel(time = val_indices)
    test_total_moc = total_moc_max.isel(time = test_indices)

    mean_total_moc = total_moc_max.isel(time = train_indices).mean('time')
    std_total_moc = total_moc_max.isel(time = train_indices).std('time')

    train_total_moc_scaled = ((train_total_moc - mean_total_moc) / std_total_moc).values
    val_total_moc_scaled = ((val_total_moc - mean_total_moc) / std_total_moc).values
    test_total_moc_scaled = ((test_total_moc - mean_total_moc) / std_total_moc).values


    florida_current_sv = (florida_current.sobarstf / 1e6)

    train_fs = florida_current_sv.isel(time = train_indices)
    val_fs = florida_current_sv.isel(time = val_indices)
    test_fs = florida_current_sv.isel(time = test_indices)

    mean_fs = florida_current_sv.isel(time = train_indices).mean('time')
    std_fs = florida_current_sv.isel(time = train_indices).std('time')

    train_fs_scaled = ((train_fs - mean_fs) / std_fs).values
    val_fs_scaled = ((val_fs - mean_fs) / std_fs).values
    test_fs_scaled = ((test_fs - mean_fs) / std_fs).values

    antilles_current_sv = antilles_current.__xarray_dataarray_variable__

    train_ac = antilles_current_sv.isel(time = train_indices)
    val_ac = antilles_current_sv.isel(time = val_indices)
    test_ac = antilles_current_sv.isel(time = test_indices)

    mean_ac = antilles_current_sv.isel(time = train_indices).mean('time')
    std_ac = antilles_current_sv.isel(time = train_indices).std('time')

    train_ac_scaled = ((train_ac - mean_ac) / std_ac).values
    val_ac_scaled = ((val_ac - mean_ac) / std_ac).values
    test_ac_scaled = ((test_ac - mean_ac) / std_ac).values

    # windstress_mean = wind_stress.mean(['y', 'x']).sozotaux
    windstress_mean = wind_stress.sozotaux.mean('y').groupby_bins('x', 10).mean()

    train_ws = windstress_mean.isel(time = train_indices)
    val_ws = windstress_mean.isel(time = val_indices)
    test_ws = windstress_mean.isel(time = test_indices)

    mean_ws = windstress_mean.isel(time = train_indices).mean('time')
    std_ws = windstress_mean.isel(time = train_indices).std('time')

    train_ws_scaled = ((train_ws - mean_ws) / std_ws).values
    val_ws_scaled = ((val_ws - mean_ws) / std_ws).values
    test_ws_scaled = ((test_ws - mean_ws) / std_ws).values


    mean_transport  = train_y.mean('time')

    train_y_scaled = (train_y - mean_transport) / std_transport
    val_y_scaled = (val_y - mean_transport) / std_transport
    test_y_scaled = (test_y - mean_transport) / std_transport

    train_y_prev_scaled = (train_y_prev - mean_transport.values) / std_transport.values
    val_y_prev_scaled = (val_y_prev - mean_transport.values) / std_transport.values
    test_y_prev_scaled = (test_y_prev - mean_transport.values) / std_transport.values

    train_y_scaled = train_y_scaled.fillna(0).values
    val_y_scaled = val_y_scaled.fillna(0).values
    test_y_scaled = test_y_scaled.fillna(0).values

    # train_y_prev_scaled = train_y_prev_scaled.fillna(0).values
    # val_y_prev_scaled = val_y_prev_scaled.fillna(0).values
    # test_y_prev_scaled = test_y_prev_scaled.fillna(0).values

    # dv_dz_obs = xr.open_dataset('../rapid-geostrophic-reconstruction/data/dv_dz_sim_miss_70None.nc')
    # dv_dz_obs = dv_dz_obs.resample(time = time_smoothing).mean()
    dv_dz_moorings = dv_dz_obs.sel(time = ds_argo_merged_10days.time).where(dv_dz_obs.z <= -2000, drop = True)

    mean_dv_dz_moorings = dv_dz_moorings.isel(time = train_indices).mean(['time']).dv_dz_times_X
    std_dv_dz_moorings = dv_dz_moorings.isel(time = train_indices).std(['time']).dv_dz_times_X

    train_dv_dz_moorings_scaled = ((dv_dz_moorings.isel(time = train_indices).dv_dz_times_X - mean_dv_dz_moorings) / std_dv_dz_moorings).fillna(0).values
    val_dv_dz_moorings_scaled = ((dv_dz_moorings.isel(time = val_indices).dv_dz_times_X - mean_dv_dz_moorings) / std_dv_dz_moorings).fillna(0).values
    test_dv_dz_moorings_scaled = ((dv_dz_moorings.isel(time = test_indices).dv_dz_times_X - mean_dv_dz_moorings) / std_dv_dz_moorings).fillna(0).values

    if missing_values_in_target:
        scaled_mis_positions = ds_pos_sim.sel(time = ds_argo_merged_10days.time, method='nearest').missing_indices / 10

        train_missing_indices = scaled_mis_positions.isel(time = train_indices).values
        val_missing_indices = scaled_mis_positions.isel(time = val_indices).values
        test_missing_indices = scaled_mis_positions.isel(time = test_indices).values

    else:
        train_missing_indices = np.zeros((train_indices.shape[0], 4))
        val_missing_indices = np.zeros((val_indices.shape[0], 4))
        test_missing_indices = np.zeros((test_indices.shape[0], 4))

    train_dataset = ProfileDataset(train_X_scaled, train_y_scaled, train_y_prev_scaled, train_X_mask, train_X_lon, train_X_lat, train_dv_dz_moorings_scaled, train_X_days, compartments, train_missing_indices, train_fs_scaled, train_ac_scaled, train_ws_scaled, train_total_moc_scaled, time_smoothing, lat_bounds, using_transport_from_previous_year, using_missing_indices, use_deep_dvdz, train_indices)
    val_dataset = ProfileDataset(val_X_scaled, val_y_scaled, val_y_prev_scaled, val_X_mask, val_X_lon, val_X_lat, val_dv_dz_moorings_scaled, val_X_days, compartments, val_missing_indices, val_fs_scaled, val_ac_scaled, val_ws_scaled, val_total_moc_scaled, time_smoothing, lat_bounds, using_transport_from_previous_year, using_missing_indices, use_deep_dvdz, val_indices)
    test_dataset = ProfileDataset(test_X_scaled, test_y_scaled, test_y_prev_scaled, test_X_mask, test_X_lon, test_X_lat, test_dv_dz_moorings_scaled, test_X_days, compartments, test_missing_indices, test_fs_scaled, test_ac_scaled, test_ws_scaled, test_total_moc_scaled, time_smoothing, lat_bounds, using_transport_from_previous_year, using_missing_indices, use_deep_dvdz, test_indices)

    return train_dataset, val_dataset, test_dataset, (mean_total_moc, std_total_moc, mean_transport, std_transport)

def merge_profiles(data):
    """
        X: is a list and each entry is a list of compartments in which the profiles are stored with variable length
    """
    X, y, y_prev, lon, lat, dv_dz, days, missing_indices, fs, ac, ws, total_moc = zip(*data)
    X = [[torch.tensor(x_compartment).float() for x_compartment in x_sample] for x_sample in X]
    y = torch.tensor(np.array(y)).float()
    y_prev = torch.tensor(np.array(y_prev)).float()
    lon = [[torch.tensor(lon_compartment).float() for lon_compartment in lon_sample] for lon_sample in lon]
    lat = [[torch.tensor(lat_compartment).float() for lat_compartment in lat_sample] for lat_sample in lat]
    dv_dz = [torch.tensor(dv_dz_sample).float() for dv_dz_sample in dv_dz]
    missing_indices = [torch.tensor(missing_indices_sample).float() for missing_indices_sample in missing_indices]
    days = [[torch.tensor(days_compartment).float() for days_compartment in day_sample] for day_sample in days]
    fs = torch.tensor(np.array(fs)).float()
    ac = torch.tensor(np.array(ac)).float()
    ws = torch.tensor(np.array(ws)).float()
    total_moc = torch.tensor(np.array(total_moc)).float()

    return X, y, y_prev, lon, lat, dv_dz, days,missing_indices, fs, ac, ws, total_moc
    

def merge_profiles_max_profiles(data):
    """
        X: is a list and each entry is a list of compartments in which the profiles are stored with variable length
    """
    x, y, y_prev, lon, lat, dv_dz, days, missing_indices, fs, ac, ws, total_moc = zip(*data)
    x = [[torch.tensor(x_compartment).float() for x_compartment in x_sample] for x_sample in x]
    y = torch.tensor(np.array(y)).float()
    y_prev = torch.tensor(np.array(y_prev)).float()
    lon = [[torch.tensor(lon_compartment).float() for lon_compartment in lon_sample] for lon_sample in lon]
    lat = [[torch.tensor(lat_compartment).float() for lat_compartment in lat_sample] for lat_sample in lat]
    dv_dz = [torch.tensor(dv_dz_sample).float() for dv_dz_sample in dv_dz]
    missing_indices = [torch.tensor(missing_indices_sample).float() for missing_indices_sample in missing_indices]
    days = [[torch.tensor(days_compartment).float() for days_compartment in day_sample] for day_sample in days]
    fs = torch.tensor(np.array(fs)).float()
    ac = torch.tensor(np.array(ac)).float()
    ws = torch.tensor(np.array(ws)).float()
    total_moc = torch.tensor(np.array(total_moc)).float()

    n_batch = len(x)
    x = [torch.cat(x_batch, dim = 0) for x_batch in x]
    max_profiles = max([len(x_i) for x_i in x])

    x_padded = torch.zeros(n_batch, max_profiles, x[0].shape[-1])
    mask_padded = torch.zeros(n_batch, max_profiles)

    for i, x_i in enumerate(x):
        x_padded[i, :len(x_i)] = x_i
        mask_padded[i, :len(x_i)] = 1

    x = x_padded # shape (batch, max_profiles, n_features)

    lon = [torch.cat(lon_batch, dim = 0) for lon_batch in lon]
    lon_pad = torch.zeros(n_batch, max_profiles)
    for i, lon_i in enumerate(lon):
        lon_pad[i, :len(lon_i)] = lon_i
    lon = lon_pad # shape (batch, max_profiles)

    lat = [torch.cat(lat_batch, dim = 0) for lat_batch in lat]
    lat_pad = torch.zeros(n_batch, max_profiles)
    for i, lat_i in enumerate(lat):
        lat_pad[i, :len(lat_i)] = lat_i
    lat = lat_pad # shape (batch, max_profiles)

    days = [torch.cat(days_batch, dim = 0) for days_batch in days]
    days_pad = torch.zeros(n_batch, max_profiles)
    for i, days_i in enumerate(days):
        days_pad[i, :len(days_i)] = days_i
    days = days_pad # shape (batch, max_profiles)

    # y_prev = torch.stack(y_prev, dim = 0) # shape (batch, 3)

    missing_indices = torch.stack(missing_indices, dim = 0) # shape (batch, 4)

    return x, mask_padded, y, y_prev, lon, lat, dv_dz, days,missing_indices, fs, ac, ws, total_moc



