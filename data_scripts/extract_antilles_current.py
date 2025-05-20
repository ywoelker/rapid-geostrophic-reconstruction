import xarray as xr
import numpy as np
from pathlib import Path
import gsw
from xarray import open_mfdataset, open_dataset
from matplotlib import pyplot as plt

from dask.distributed import Client

import numpy as np
import xarray as xr
from pathlib import Path




import argparse



# Utilities 
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
        chunks={'y': 100, 'x':100, 'z': -1},
    )

    if model == 'VIKING20X.L46-KFS003' and not nest:
        nest_part = 'ori_'

    mask_glo = open_dataset(
        f'/gxfs_work/geomar/smomw355/model_data/ocean-only/{model}/nemo/suppl/{nest_part}new_maskglo.nc',
        decode_cf=False,
        chunks={'Y':100, 'X':100, 'z': -1}
    )

    mask_mesh = mask_mesh.squeeze()
    mask_glo = mask_glo.rename({'X':'x', 'Y':'y'}).squeeze() # Rename coordinates to have the same name as the data file
    
    return mask_mesh, mask_glo




def main():

    parser = argparse.ArgumentParser(description='Create the dataset for the paper draft')
    
    parser.add_argument('--cycle_number', type=int, default=1, help='The cycle number of the model')
    parser.add_argument('--dask_url', type=str, default='tcp://10.0.4.100:8786', help='The url of the dask scheduler. None if no dask is used')



    args = parser.parse_args()

    cycle_number = args.cycle_number


    if cycle_number == 1:
        suffix = '1st_7024' 
        model = 'VIKING20X.L46-KFS003'
        model_years_selector = (1970, 2024)
    elif cycle_number == 2:
        suffix = '2nd_5824'
        model = 'VIKING20X.L46-KFS003-2nd'
        model_years_selector = (1958, 2024)
    elif cycle_number == 3:
        suffix = '3rd_5824'
        model = 'VIKING20X.L46-KFS003-3rd'
        model_years_selector = (1958, 2024)
    elif cycle_number == 4:
        suffix = '4th_5824'
        model = 'VIKING20X.L46-KFS003-4th'
        model_years_selector = (1958, 2024)
    elif cycle_number == 5:
        suffix = '5th_5824'
        model = 'VIKING20X.L46-KFS003-5th'
        model_years_selector = (1958, 2024)
    elif cycle_number == 6:
        suffix = '6th_5824'
        model = 'VIKING20X.L46-KFS003-6th'
        model_years_selector = (1958, 2024)
    else:
        assert False, 'Cycle number not recognized'

    from dask.distributed import Client
    # client = Client(n_workers=6, threads_per_worker=1, memory_limit='8GB')

    client = Client(args.dask_url)


    V_grid = load_dataset_from_grid_type('V', model_years_selector, nest = True, model = model, delta_t = '1d', chunks={"time_counter" : 12,"y": None, "x": None})



    mask_mesh, mask_glo = load_masks(nest = True, model = model)





    V_grid_loaded = V_grid.isel(y = 1274).isel(x = slice(400,2000)).vomecrty
    dx = mask_mesh.e1v.isel(y = 1274).isel(x = slice(400,2000))
    dz = mask_mesh.e3v_0.isel(y = 1274).isel(x = slice(400,2000)).rename({'z':'depthv'})
    volume_transport = V_grid_loaded * dx * dz



    transport = (
        volume_transport.set_index(x = 'nav_lon').sel(x = slice(-77,-76.5), depthv = slice(0, 1000)).sum(['depthv', 'x']) / 1e6
    )



    transport.to_netcdf(f'../rapid-geostrophic-reconstruction/datasets/antilles_current/antilles_current_KFS003-{suffix}.nc')


if __name__ == '__main__':
    main()