from argparse import ArgumentParser

from xarray import open_dataset, merge, Dataset, where, concat
import json
import numpy as np
import pandas as pd
import gsw
from pathlib import Path
import pickle as pckl


def load_ds_pos_obs(ts_gridded):
    ds_tsg = ts_gridded
    ds_tsg = ds_tsg.rename({"depth": "z"})
    ds_tsg = ds_tsg.assign_coords(z=-abs(ds_tsg.pressure))
    ds_tsg = ds_tsg.set_coords(["pressure", ])
    ds_tsg.z.attrs["units"] = "m"
    ds_tsg.z.attrs["long_name"] = "vertical position"
    ds_tsg

    _TG = ds_tsg[["TG_west", "TG_wb3", "TG_marwest", "TG_mareast", "TG_east"]].to_array("pos")
    _TG = _TG.assign_coords(pos=range(5))
    _TG = _TG.assign_coords(name=(("pos", ), ["west", "wb3", "marwest", "mareast", "east"]))
    _TG = _TG.assign_coords(lon=(("pos", ), [-76.74, -76.50, -50.57, -41.21, -16.23,]))
    _TG = _TG.assign_coords(lat=(("pos", ), [26.52, 26.50, 24.52, 24.52, 26.99, ]))
    _TG = _TG.rename("temperature")
    _TG.attrs["long_name"] = "Temperature"
    _TG.attrs["units"] = "degC"

    _SG = ds_tsg[["SG_west", "SG_wb3", "SG_marwest", "SG_mareast", "SG_east"]].to_array("pos")
    _SG = _SG.assign_coords(pos=range(5))
    _SG = _SG.assign_coords(name=(("pos", ), ["west", "wb3", "marwest", "mareast", "east"]))
    _SG = _SG.assign_coords(lon=(("pos", ), [-76.74, -76.50, -50.57, -41.21, -16.23,]))
    _SG = _SG.assign_coords(lat=(("pos", ), [26.52, 26.50, 24.52, 24.52, 26.99, ]))
    _SG = _SG.rename("salinity")
    _SG.attrs["long_name"] = "Salinity"
    _SG.attrs["units"] = "psu"

    ds_pos_obs = merge([_TG, _SG])

    ds_pos_obs = ds_pos_obs.assign_coords(pos=ds_pos_obs.name)

    return ds_pos_obs

def load_ds_pos_model(mooring_group_sim, ds_pos_obs):
    ds_model_groups = mooring_group_sim
    ds_model_groups = ds_model_groups.rename({"western_mooring": "field"})
    
    ds_pos_sim = Dataset(
        {
            "temperature": ds_model_groups.field.sel(variable="temp", drop=True).rename({
                "time_counter": "time",
                "deptht": "z",
                "mooring": "pos",
                "nav_lon": "lon",
                "nav_lat": "lat",
            }),
            "salinity": ds_model_groups.field.sel(variable="salt", drop=True).rename({
                "time_counter": "time",
                "deptht": "z",
                "mooring": "pos",
                "nav_lon": "lon",
                "nav_lat": "lat",
            }),
        },
    )
    ds_pos_sim = ds_pos_sim.assign_coords(z=-abs(ds_pos_sim.z))
    ds_pos_sim = ds_pos_sim.interp(
        z=ds_pos_obs.z,
        method="slinear",
        kwargs={"fill_value": "extrapolate"},
    )
    ds_pos_sim = ds_pos_sim.drop_vars(["x", "y", "i_moorings", "moorings", "time_centered", "pos"])
    ds_pos_sim = ds_pos_sim.assign_coords(
        name=(("pos", ), ['west', 'east', 'marwest', 'mareast']),
        pos=(("pos", ), ['west', 'east', 'marwest', 'mareast']),
    )
    ds_pos_sim = ds_pos_sim.assign_coords(
        lon=ds_pos_sim.lon.mean("z"),
        lat=ds_pos_sim.lat.mean("z"),
    )
    ds_pos_sim = merge([v.transpose(*ds_pos_obs.dims) for k, v in ds_pos_sim.data_vars.items()])

    ds_pos_sim = ds_pos_sim.assign_coords(pos=ds_pos_sim.name)


    return ds_pos_sim


def create_ds_pos_sim_miss(ds_pos_sim, time_mapping_dict, time_smooth, ds_pos_obs):
    t_sim_mapped_for_obs_masking = []

    for t_sim in ds_pos_sim.time:
        t_sim_smooth = t_sim.dt.round(time_smooth)
        day_offset = (t_sim - t_sim_smooth).dt.days
        t_mapped_obs = time_mapping_dict[t_sim_smooth.values[()]]
        t_mapped_obs_offset = t_mapped_obs + pd.Timedelta(day_offset.values[()], 'D')
        t_sim_mapped_for_obs_masking.append(t_mapped_obs_offset)
        
    t_sim_mapped_for_obs_masking = np.array(t_sim_mapped_for_obs_masking)

    ds_pos_sim_miss = ds_pos_sim.where(
        ds_pos_obs.sel(pos = ds_pos_sim.pos, z = ds_pos_sim.z).sel(time = t_sim_mapped_for_obs_masking).temperature.notnull().assign_coords(time = ds_pos_sim.time)
    )

    ds_pos_sim_miss = ds_pos_sim_miss.assign_coords(
        lon = ds_pos_obs.lon, 
        lat = ds_pos_obs.lat,
    )

    missing_indices = ds_pos_obs.sel(pos = ds_pos_sim.pos, z = ds_pos_sim.z).sel(time = t_sim_mapped_for_obs_masking).temperature.notnull().assign_coords(time = ds_pos_sim.time).argmax('z')

    missing_indices = where(missing_indices > 0, missing_indices, ds_pos_obs.z.size)
    ds_pos_sim_miss['missing_indices'] = missing_indices

    ds_pos_sim_miss = ds_pos_sim_miss.sortby("lon")
    return ds_pos_sim_miss


def add_ct_and_rho(ds_pos):

    ds_pos["ct"] = gsw.conversions.CT_from_pt(ds_pos.salinity, ds_pos.temperature)
    ds_pos["ct"].attrs["long_name"] = "Conservative Temperature"
    ds_pos["ct"].attrs["units"] = "degC"
    
    ds_pos["rho"] = gsw.density.rho(ds_pos.salinity, ds_pos.ct, ds_pos.pressure)
    ds_pos["rho"].attrs["long_name"] = "Density"
    ds_pos["rho"].attrs["units"] = "kg/m3"
    
    return ds_pos


def calc_t_umo_geostrophic(
    ds_pos=None,
    zref=-4800,
):

    pos_pairs_depth_range = [
        (("west", "east"), (None, -3700)),
        (("west", "marwest"), (-3700, None)),
        (("mareast", "east"), (-3700, None)),
    ]
    
    rho_diff = concat(
        [
            (
                ds_pos.sel(pos=ds_pos.name == n1).squeeze(drop=True).sel(z=slice(z0, z1)).rho
                - ds_pos.sel(pos=ds_pos.name == n0).squeeze(drop=True).sel(z=slice(z0, z1)).rho
            ).assign_coords(box="_".join((n0, n1)))
            for ((n0, n1), (z0, z1)) in pos_pairs_depth_range
        ],
        dim="box"
    )
    rho_diff

    dv_dz = (- rho_diff * g / f / rho0).sum("box").where(
        (~(- rho_diff * g / f / rho0).isnull()).sum("box") > 0
    ).rename("dv_dz_times_X")

    

    _dv_dz = dv_dz.isel(z=slice(-40, None))
    dv_dz_fillup = where(
        ~dv_dz.isnull(),
        dv_dz,
        (_dv_dz.isel(z=0) + _dv_dz.diff("z").ffill("z").cumsum("z")).interp(z=dv_dz.z)
    )

    
    transp_umo = (-(
        ((dv_dz_fillup.isel(z=slice(None, None, -1)).cumsum("z") * 20.0))
        - ((dv_dz_fillup.isel(z=slice(None, None, -1)).cumsum("z") * 20.0)).sel(z=zref, method="nearest")
    ).cumsum("z") * 20.0 / 1e6).sel(z=slice(None, -1300)).sel(z=-1000, method="nearest")
    
    return  transp_umo, dv_dz, dv_dz_fillup







if __name__ == '__main__':

    args = ArgumentParser()

    args.add_argument('--mooring_group_sim', type=str, help='File with simulation moorings')
    args.add_argument('--ts_gridded', type= str, help='File with gridded temperature and salinity from rapid')

    args.add_argument('--time_mapping_dict', type=str, help='File with time mapping dictionary saved as json')
    args.add_argument('--cycle_name', type=str, help='The simulation cycle name for the olutput file')

    args.add_argument('--reference_level', type=int, help='refernece level for transport calculation', default=-4800)

    args.add_argument('--time_smooth', type=str, help='Time smoothing for the model data', default='90D')


    args = args.parse_args()

    mooring_group_sim = open_dataset(args.mooring_group_sim)
    ts_gridded = open_dataset(args.ts_gridded)


    if str(args.time_mapping_dict).endswith('.json'):
        time_mapping_dict = json.load(open(args.time_mapping_dict))
    elif str(args.time_mapping_dict).endswith('.pickle'):
        time_mapping_dict = pckl.load(open(args.time_mapping_dict, 'rb'))

    ds_pos_obs = load_ds_pos_obs(ts_gridded)
    ds_pos_sim = load_ds_pos_model(mooring_group_sim, ds_pos_obs)
    ds_pos_sim_miss = create_ds_pos_sim_miss(ds_pos_sim, time_mapping_dict, args.time_smooth, ds_pos_obs)

    # add rho
    ds_pos_sim_miss = add_ct_and_rho(ds_pos_sim_miss)
    ds_pos_sim = add_ct_and_rho(ds_pos_sim)
    ds_pos_obs = add_ct_and_rho(ds_pos_obs)

    # constants
    g = 9.81
    f = 2 * 7.292116E-5 * np.sin(np.deg2rad(26.5))
    rho0 = 1025.0



    output_path = Path(args.time_mapping_dict).parent / f'{abs(args.reference_level)}m_ref/'
    output_path.mkdir(exist_ok=True)

    
    t_umo_obs, dv_dz_obs, dv_dz_obs_fillup = calc_t_umo_geostrophic(ds_pos_obs, zref=args.reference_level)

    t_umo_sim_miss, dv_dz_sim_miss, dv_dz_sim_miss_fillup = calc_t_umo_geostrophic(ds_pos_sim_miss, zref = args.reference_level)

    t_umo_sim, dv_dz_sim, dv_dz_sim_fillup = calc_t_umo_geostrophic(ds_pos_sim, zref = args.reference_level)


    t_umo_obs.to_netcdf(output_path / f't_umo_obs.nc')
    dv_dz_obs.to_netcdf(output_path / f'dv_dz_obs.nc')
    dv_dz_obs_fillup.to_netcdf(output_path / f'dv_dz_obs_fillup.nc')
    ds_pos_obs.to_netcdf(output_path / 'ds_pos_obs.nc')

    t_umo_sim_miss.to_netcdf(output_path / f't_umo_sim_miss_{args.cycle_name}.nc')
    dv_dz_sim_miss.to_netcdf(output_path / f'dv_dz_sim_miss_{args.cycle_name}.nc')
    dv_dz_sim_miss_fillup.to_netcdf(output_path / f'dv_dz_sim_miss_fillup_{args.cycle_name}.nc')
    ds_pos_sim_miss.to_netcdf(output_path / f'ds_pos_sim_miss_{args.cycle_name}.nc')

    t_umo_sim.to_netcdf(output_path / f't_umo_sim_{args.cycle_name}.nc')
    dv_dz_sim.to_netcdf(output_path / f'dv_dz_sim_{args.cycle_name}.nc')
    dv_dz_sim_fillup.to_netcdf(output_path / f'dv_dz_sim_fillup_{args.cycle_name}.nc')
    ds_pos_sim.to_netcdf(output_path / f'ds_pos_sim_{args.cycle_name}.nc')

