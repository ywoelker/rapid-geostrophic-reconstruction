

import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from gsw import CT_from_pt, rho
from dask.distributed import Client
from xarray import open_mfdataset, open_dataset
from sklearn.neighbors import NearestNeighbors


def connect_to_dask(dask_url):
    client = Client(dask_url)
    return client


def find_grid_files(grid_type, years, nest, model, delta_t, folder = 'output'): 
    experiment_path = Path('/gxfs_work/geomar/smomw355/model_data/ocean-only/')
    nest_part = '1_' if nest else '' 

    if len(grid_type) < 2:
        grid_type = f'grid_{grid_type}'

    grid_files = (experiment_path / model / 'nemo' / folder).glob(f'{nest_part}{model}_{delta_t}_*_*_{grid_type}.nc')
    grid_files = list(grid_files)
    
    result_files = []
    for file in grid_files:
        start_year = int(str(file).split('/')[-1].split('_')[- (3 + grid_type.count('_'))][:4])
        
        if (years[0] is None or start_year >= years[0]) and \
           (years[1] is None or start_year < years[1]):
            result_files.append(file)
    
    return sorted(result_files)

def load_dataset_from_grid_type(grid_type, years, nest, model, delta_t, chunks = {"time_counter" : 2,"y": 2, "x": None}, **kwargs):
    files_list = find_grid_files(grid_type, years, nest, model, delta_t, **kwargs)
    grid = open_mfdataset(
        files_list, # type: ignore
        chunks= chunks,
        combine='by_coords'
    )
    return grid

def load_masks(nest, model):

    if str.startswith(model, 'VIKING20X.L46-KFS003'):
        # All the mask for the long runs VIKING20X.L46-KFS003-2nd, ... are placed in the original experiment
        model = 'VIKING20X.L46-KFS003' 
    
    nest_part = '1_' if nest else ''
    
    mask_mesh = open_dataset(
        f'/gxfs_work/geomar/smomw355/model_data/ocean-only/{model}/nemo/suppl/{nest_part}mesh_mask.nc',
        decode_cf=False,
        chunks={'y': 512, 'x':512},
    )
    mask_glo = open_dataset(
        f'/gxfs_work/geomar/smomw355/model_data/ocean-only/{model}/nemo/suppl/{nest_part}new_maskglo.nc',
        decode_cf=False,
        chunks={'Y':512, 'X':512}
    )

    mask_mesh = mask_mesh.squeeze()
    mask_glo = mask_glo.rename({'X':'x', 'Y':'y'}).squeeze() # Rename coordinates to have the same name as the data file
    
    return mask_mesh, mask_glo








"""
Back when we only had the original Argo in 10 day splits and used this code to transform them to 90days sections
"""
# max_profiles_per_time_smooting_unit = obs_argo_array.profile_mask.resample(time=time_smoothing).sum().sum('pos').max().values[()]

# def group_profiles(dataset):
    
#     reference_time = dataset.time.dt.round(time_smoothing).isel(time = 0).values[()]

#     smoothed = dataset.stack(pos_2 = ['time', 'pos'])
#     smoothed['time_1d'] = smoothed.time_1d.astype('datetime64[ns]')
#     smoothed = smoothed.where(smoothed.profile_mask > 0 , drop = True).drop_vars(['time', 'pos', 'pos_2']).rename({'pos_2': 'pos'}).expand_dims({'time': [reference_time]})
#     smoothed = smoothed.pad(pos = (0, int(max_profiles_per_time_smooting_unit)- smoothed.pos.shape[0])).assign_coords({'pos': np.arange(int(max_profiles_per_time_smooting_unit))})

#     # smoothed['time_1d'] = smoothed.time_1d.fillna(smoothed.time_1d.isel(pos = 0))

#     return smoothed

# grouped_array_of_argos = obs_argo_array.resample(time=time_smoothing).apply(group_profiles).assign_coords({'time': obs_argo_array.time.resample(time=time_smoothing).mean()}).isel(time = (obs_argo_array.profile_mask.sum('pos') > phi).resample(time = time_smoothing).all().values)





def create_time_mapping(time_mapping_file, time_smoothing, time_min, time_max, argo_time_slice, grouped_array_of_argos, T):
    if time_mapping_file.exists():

        with open(time_mapping_file, 'rb') as handle:
            time_mapping_dict = pickle.load( handle)
            print('Loaded')

    else:

        single_frequency = True

        if single_frequency:
            list_of_fine_days_timesteps = np.unique(T.time_counter.dt.round(time_smoothing))

            list_of_argo_three_month_timesteps = grouped_array_of_argos.time.sel(time = argo_time_slice)
            list_of_argo_three_month_timesteps = list_of_argo_three_month_timesteps.where((list_of_argo_three_month_timesteps + pd.Timedelta(time_smoothing) / 2 <= time_max) & (list_of_argo_three_month_timesteps - pd.Timedelta(time_smoothing) / 2 >= time_min), drop = True)


            random_indices = np.random.choice(np.arange(len(list_of_argo_three_month_timesteps)), len(list_of_fine_days_timesteps), replace = True)


            time_mapping_dict = dict(zip(list_of_fine_days_timesteps, list_of_argo_three_month_timesteps.values[random_indices]))
        else:
            list_of_model_three_month_timesteps = np.unique(T.time_counter.dt.round('90D'))

            list_of_fine_days_timesteps = np.unique(T.time_counter.dt.round('10D'))

            list_of_argo_timesteps = grouped_array_of_argos.time.sel(time = argo_time_slice)
            list_of_argo_timesteps = list_of_argo_timesteps.where((list_of_argo_timesteps + pd.Timedelta('90D') / 2 <= time_max) & (list_of_argo_timesteps - pd.Timedelta('90D') / 2 >= time_min), drop = True)

            coarse_list_of_argo_timesteps = list_of_argo_timesteps.groupby(list_of_argo_timesteps.time.dt.floor('90D')).mean()


            random_indices = np.random.choice(np.arange(len(list_of_argo_timesteps)), len(list_of_model_three_month_timesteps), replace = True)

            time_mapping_dict = dict(zip(list_of_model_three_month_timesteps, list_of_argo_timesteps.values[random_indices]))

            ten_days_model = T.time_counter.dt.round('10D').groupby(T.time_counter.dt.round('10D')).first()
            rounded_ninety_days_model = ten_days_model.dt.round('90D')

            time_mapping_dict_fine = {}

            for fine_time_model, coarse_time_model in zip(ten_days_model, rounded_ninety_days_model):

                coarse_time_obs = time_mapping_dict[coarse_time_model.values[()]]


                delta_model = fine_time_model - coarse_time_model

                fine_time_obs = coarse_time_obs + delta_model

                assert fine_time_obs.values[()] in grouped_array_of_argos.time.values, f"{fine_time_obs.values[()]} not in list_of argo timesteps"

                time_mapping_dict_fine[fine_time_model.values[()]] = fine_time_obs.values[()]

                    
            time_mapping_dict = time_mapping_dict_fine

        with open(time_mapping_file, 'wb') as handle:
            pickle.dump(time_mapping_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saved')

    return time_mapping_dict





def create_argo_frame(time_mapping_dict, time_smoothing, grouped_array_of_argos, T):
    argo_frame = None 

    # max_model_time = T.time_counter.max().values[()]



    for timestep in list(time_mapping_dict.keys()):
        if type(timestep) == xr.DataArray:
            timestep = timestep.values[()]

        if timestep + (pd.to_timedelta(time_smoothing) / 2) > T.time_counter.max().values[()] or\
            timestep - (pd.to_timedelta(time_smoothing) / 2) < T.time_counter.min().values[()]:
            continue
        # df = grouped_array_of_argos.sel(time = time_mapping_dict[timestep]).isel(z = 0).to_dataframe()
        # df['days_offset'] = (df.time_1d.where(df.profile_mask > 0) - df.time.where(df.profile_mask > 0)).dt.days

        df = grouped_array_of_argos.sel(time = slice(
            time_mapping_dict[timestep] - (pd.to_timedelta(time_smoothing) / 2),
            time_mapping_dict[timestep] + (pd.to_timedelta(time_smoothing) / 2)
        )).stack(pos_2 = ['time', 'pos']).isel(z = 0).to_dataframe().reset_index(drop = True)

        df['days_offset'] = (df.time_1d.where(df.profile_mask > 0) - time_mapping_dict[timestep]).dt.days

        df['time_sim'] = timestep
        df['time_1d_sim'] = df.time_sim + pd.to_timedelta(df.days_offset, unit = 'D')

        df = df.rename(columns = {
            'time_1d': 'time_1d_argo',
            'time': 'time_argo'
            })

        df.drop(columns = ['time_argo'], inplace = True)

        df = df[~pd.isna(df.profile_mask)]

        if df is not None:
            argo_frame = pd.concat([argo_frame, df])
        else:
            argo_frame = df


    return argo_frame#.groupby('time_sim').count()['temperature'].plot()

def add_model_xy_to_argo_frame(model, argo_frame):
    mask_mesh, _ = load_masks(True, model)

    c_mask_mesh = mask_mesh.assign_coords({'x': np.arange(mask_mesh.x.shape[0]), 'y': np.arange(mask_mesh.y.shape[0])})

    lon_min = argo_frame.lon.min() - 2
    lon_max = argo_frame.lon.max() + 2
    lat_min = argo_frame.lat.min() - 2
    lat_max = argo_frame.lat.max() + 2

    year_min = argo_frame.time_sim.dt.year.min()
    year_max = argo_frame.time_sim.dt.year.max() + 1


    cutted_c_mask = c_mask_mesh.where(
        ((lat_min <= mask_mesh.nav_lat) & (mask_mesh.nav_lat <= lat_max) & (lon_min <= mask_mesh.nav_lon) & (mask_mesh.nav_lon <= lon_max)).compute()
        , drop = True)

    lonlat_model = np.vstack((cutted_c_mask.where(cutted_c_mask.tmaskutil > 0).nav_lon.stack({'list': ['x', 'y']}).values, cutted_c_mask.where(cutted_c_mask.tmaskutil > 0).nav_lat.stack({'list': ['x', 'y']}).values)).T


    # lonlat_argo = np.vstack((obs_argo_array.stack(f=('time', 'pos')).lon.values, obs_argo_array.stack(f=('time', 'pos')).lat.values)).T.astype(float)

    lonlat_argo = argo_frame[['lon', 'lat']].values # here are only those entries that are no NaNs
    # print(lonlat_model.shape, lonlat_argo.shape)
    # print(lonlat_model)
    # print(lonlat_argo)


    non_nan_model_mask = ~np.isnan(lonlat_model).any(axis = -1) 

    neigh = NearestNeighbors(n_neighbors=1, metric='haversine')
    neigh.fit(np.radians(lonlat_model[non_nan_model_mask]))

    distances, indices = neigh.kneighbors(np.radians(lonlat_argo))
    argo_indices_without_nans = indices.squeeze()


    non_nan_model_indices = np.argwhere(non_nan_model_mask).squeeze()
    argo_indices = non_nan_model_indices[argo_indices_without_nans]

    argo_indices, argo_indices_without_nans

    XY_indices = np.array([*cutted_c_mask.where(cutted_c_mask.tmaskutil > 0).stack({'list': ['x', 'y']}).list.isel(list = xr.DataArray(argo_indices)).values])


    argo_frame = argo_frame.assign(model_x = XY_indices[:, 0], model_y =  XY_indices[:, 1])
    argo_frame = argo_frame.assign(
        model_lon = cutted_c_mask.sel(x = xr.DataArray(XY_indices[:, 0]), y = xr.DataArray(XY_indices[:, 1])).nav_lon.values,
        model_lat = cutted_c_mask.sel(x = xr.DataArray(XY_indices[:, 0]), y = xr.DataArray(XY_indices[:, 1])).nav_lat.values,
    )

    argo_frame = argo_frame.assign(
        error = (argo_frame.lon - argo_frame.model_lon) ** 2 + (argo_frame.lat - argo_frame.model_lat) ** 2,
    )

    argo_frame = argo_frame.reset_index()
    argo_frame.time_1d_sim = argo_frame.time_1d_sim + np.timedelta64(12, 'h')

    return argo_frame

def cut_modeloutput_to_argos(T, argo_frame, time_smoothing):
    modified_T = T.sel(
        y = xr.DataArray(argo_frame.model_y.values),
        x = xr.DataArray(argo_frame.model_x.values),
        # deptht = slice(None, T_with_stress.deptht[DEPTH_DIM - 1]),
        time_counter = xr.DataArray(argo_frame.time_1d_sim.values),
    )


    modified_T = modified_T.assign(
        votemper_depth_max = modified_T.votemper.where(-modified_T.deptht >= xr.DataArray(argo_frame.depth.values), 0.0),
        vosaline_depth_max = modified_T.vosaline.where(-modified_T.deptht >= xr.DataArray(argo_frame.depth.values), 0.0),
        depth_max = argo_frame.depth.astype(np.float32)
    )




    max_argos_per_ts = modified_T.time_counter.groupby('time_counter').count().max().values[()]

    profile_index = np.arange(max_argos_per_ts)
    time_index = T.time_counter.values


    pandas_multiindex = pd.MultiIndex.from_product([time_index, profile_index], names = ['time_counter', 'profiles'])



    modified_T = modified_T.assign_coords(time_counter_1d = modified_T.time_counter)
    modified_T = modified_T.assign_coords(time_counter_smooth = modified_T.time_counter.dt.round(time_smoothing))
    modified_T


    time_counter_array = modified_T.time_counter_smooth.values
    pos = []

    indices = {}

    for i in range(0, time_counter_array.shape[0]):

        ts = time_counter_array[i] 


        if ts in indices:
            pos.append(indices[ts])
            indices[ts] += 1
        else:
            indices[ts] = 1
            pos.append(0)

    pos = np.array(pos)

    modified_T = modified_T.assign_coords({'profiles': ('dim_0', pos)})


    pandas_multiindex = pd.MultiIndex.from_tuples([(a,b) for a,b in zip(modified_T.time_counter_smooth.values, modified_T.profiles.values)], names = ['time_counter', 'profiles'])

    dataset = modified_T.assign(dim_0 = pandas_multiindex).unstack('dim_0')

    return dataset

def load_dataset(dataset, deep_argo):
    if deep_argo:
        dataset.votemper.load()
        print('--> Loaded temp')
        dataset.vosaline.load()
        print('--> Loaded salt')
    else:
        dataset.votemper_depth_max.load()
        print('--> Loaded temp')
        dataset.vosaline_depth_max.load()
        print('--> Loaded salt')



def create_dataset(dataset, deep_argo):
    viking_argo_ds = xr.Dataset(
        {
            'temperature': (('z', 'time', 'pos'), dataset.votemper.values if deep_argo else dataset.votemper_depth_max.values ),
            'salinity': (('z', 'time', 'pos'),  dataset.vosaline.values if deep_argo else dataset.vosaline_depth_max.values),
            'profile_mask': (('time', 'pos'), ~np.isnan(dataset.votemper.isel(deptht = 0)).values if deep_argo else ~np.isnan(dataset.votemper_depth_max.isel(deptht = 0)).values),
        },
        coords = {
            'time': dataset.time_counter.values,
            'pos': dataset.profiles.values,
            'z': -dataset.deptht.values,
            'pressure': ('z',dataset.deptht.values),
            'lon': (('time', 'pos'), dataset.nav_lon.values),
            'lat': (('time', 'pos'), dataset.nav_lat.values),
            'time_1d': (('time','pos'), dataset.time_counter_1d.values),
        }
    )
    viking_argo_ds.profile_mask.plot()



    viking_argo_ds['ct']  = CT_from_pt(viking_argo_ds.salinity, viking_argo_ds.temperature)
    viking_argo_ds['rho'] = rho(viking_argo_ds.salinity, viking_argo_ds.temperature, viking_argo_ds.pressure)

    viking_argo_ds = viking_argo_ds.transpose('time', 'pos', 'z')
    viking_argo_ds = viking_argo_ds.sortby('z')
    viking_argo_ds.isel(time = 0, pos = 0).salinity.plot(y = 'z')


    return viking_argo_ds




print('Finished!')



import argparse


def main():

    parser = argparse.ArgumentParser(description='Create the dataset for the paper draft')
    
    parser.add_argument('--smoothing_days', type=int, default=10, help='The number of days to smooth the Argo data')
    parser.add_argument('--cycle_number', type=int, default=1, help='The cycle number of the model')
    parser.add_argument('--dask_url', type=str, default='tcp://10.0.4.100:8786', help='The url of the dask scheduler. None if no dask is used')
    parser.add_argument('--deep_argo', type=bool, default=True, help='Whether to use the deep argo data or not')



    args = parser.parse_args()

    time_smoothing = f'{args.smoothing_days}D'

    dataset_path = Path(f'../rapid-geostrophic-reconstruction/datasets/smoothing_{args.smoothing_days}_days/argo_after_2012')


    if args.cycle_number == 1:
        suffix = '1st_7024' 
        model = 'VIKING20X.L46-KFS003'
        model_years_selector = (1970, 2024)
    elif args.cycle_number == 2:
        suffix = '2nd_5824'
        model = 'VIKING20X.L46-KFS003-2nd'
        model_years_selector = (1958, 2024)
    elif args.cycle_number == 3:
        suffix = '3rd_5824'
        model = 'VIKING20X.L46-KFS003-3rd'
        model_years_selector = (1958, 2024)
    elif args.cycle_number == 4:
        suffix = '4th_5824'
        model = 'VIKING20X.L46-KFS003-4th'
        model_years_selector = (1958, 2024)
    elif args.cycle_number == 5:
        suffix = '5th_5824'
        model = 'VIKING20X.L46-KFS003-5th'
        model_years_selector = (1958, 2024)
    elif args.cycle_number == 6:
        suffix = '6th_5824'
        model = 'VIKING20X.L46-KFS003-6th'
        model_years_selector = (1958, 2024)
    else:
        assert False, 'Cycle number not recognized'

    argo_time_slice = slice('01-01-2012', None)

    dataset_sample_name = 'paperdraft'


    if not (dataset_path / dataset_sample_name).exists():
        (dataset_path / dataset_sample_name).mkdir(parents=True)


    time_mapping_file = dataset_path / dataset_sample_name / f'time_mapping_dict_{suffix}.pickle'


    if args.dask_url is not None:
        client = connect_to_dask(args.dask_url)
        print('Connected to dask')


    ## Create a dictionary that maps from the simulation time data to the observation time within th goal to keep time windows of the size `time_smoothing` as one block

    # Load the data
    T = load_dataset_from_grid_type('T', model_years_selector, True, model, '1d', chunks = {"time_counter" : 1,"y": None, "x": None})

    print('Loaded model data')

    grouped_array_of_argos = xr.open_dataset(dataset_path / '../../ds_argo_obs_daily.nc')
    mooring_data = xr.open_dataset(dataset_path / "../../ts_gridded.nc")

    time_max = min(grouped_array_of_argos.time.max(), mooring_data.time.max())
    time_min = max(grouped_array_of_argos.time.min(), mooring_data.time.min())

    grouped_array_of_argos = grouped_array_of_argos.sel(time = slice(time_min, time_max))

    time_mapping_dict = create_time_mapping(time_mapping_file, time_smoothing, time_min, time_max, argo_time_slice, grouped_array_of_argos, T)

    first_nan_index = np.isnan(grouped_array_of_argos.temperature).argmax('z') 

    depth = grouped_array_of_argos.z.isel(z = (first_nan_index - 1)).where(first_nan_index > 0).rename('depth') 

    grouped_array_of_argos = grouped_array_of_argos.assign_coords({'depth': depth, 'z': grouped_array_of_argos.z})


    argo_frame = create_argo_frame(time_mapping_dict, time_smoothing, grouped_array_of_argos, T)

    argo_frame = add_model_xy_to_argo_frame(model, argo_frame)

    print('Created argoframe')

    dataset = cut_modeloutput_to_argos(T, argo_frame, time_smoothing)

    print('Cut model output')

    load_dataset(dataset, args.deep_argo)

    viking_argo_ds = create_dataset(dataset, args.deep_argo)

    
    viking_argo_ds.to_netcdf(dataset_path / dataset_sample_name / f'ds_{"deep_" if args.deep_argo else ""}argo_viking_KFS003-{suffix}.nc')


if __name__ == '__main__':
    main()

