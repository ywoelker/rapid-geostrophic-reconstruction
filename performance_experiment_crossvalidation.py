import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from amoc_reconstruction.reconstruction.model import init_assignment_module
from amoc_reconstruction.reconstruction.model import ProfileModelSUSTeR5_fast, ProfileModelSUSTeR6, ProfileModelSUSTeR6_wind
from amoc_reconstruction.train import train, make_predictions
from amoc_reconstruction.reconstruction.dataset import merge_profiles_max_profiles
from amoc_reconstruction.reconstruction.dataset import load_merged_argo_dataset_and_tumo_cycles, filter_ds_argo_data, split_dataset_into_training_validation_testing_profile_datasets
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch
### PARAMETERS



def run_experiment():
    smoothing_days = 90

    time_smoothing = f'{smoothing_days}D'
    lat_bounds = (25, 30)



    target_reference_level = 4800

    missing_values_in_target = False


    using_missing_indices = False
    using_transport_from_previous_year = False
    use_deep_dvdz = True
    deep_argo = False
    geostrophic_prediction = False


    train_batch_sizes = {
        '10D': 32,
        '30D': 16,
        '90D': 8,
        '365D': 4,
    }


    n_compartments_for_smoothing = {
        '10D': 49,
        '30D': 11,
        '90D': 7,
        '365D': 42,
    }

    train_batch_size = train_batch_sizes[time_smoothing]

    add_tmp = False
    add_sal = False






    random_split_data = False

    assert target_reference_level in [2000, 4800], "Only 2000 and 4800 are supported as reference levels"
    version = 'v5'
    n_compartments = n_compartments_for_smoothing[time_smoothing]
    n_embedding = 12

    distance_assignment_module  = 'distance'

    experiment_path = Path(f'../rapid-geostrophic-reconstruction/figs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    dataset_path = Path(f'../rapid-geostrophic-reconstruction/datasets/smoothing_{smoothing_days}_days/argo_after_2012/base_sample/')

    total_moc_path = Path('../rapid-geostrophic-reconstruction/datasets/moc_total/')
    antilles_current_path = Path('../rapid-geostrophic-reconstruction/datasets/antilles_current/')
    florida_current_path = Path('../rapid-geostrophic-reconstruction/datasets/florida_current/')
    wind_stress_path = Path('../rapid-geostrophic-reconstruction/datasets/windstress/')

    ref_folder = f'{abs(target_reference_level)}m_ref'

    compartments = [
            ['west_p', -76.74, -70, 'west'],
            ['mar_west_p', -60, -47, 'east'],
            ['mar_east_p', -47, -40, 'west'],
            ['east_p', -30, -13.5, 'east']
        ]


    device =  'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print('Using GPU')


    assert version in ['v2', 'v3', 'v4', 'v5']
    assert distance_assignment_module in ['distance', 'fixed']


    ### Experiment 20012025
    import numpy as np

    n_iters = 3
    random_seeds = np.random.randint(0, 1000, n_iters)




    ds_argo_merged, t_umo_obs, ds_pos_sim, dv_dz_obs, t_delta, total_moc, antilles_current, florida_current, wind_stress = load_merged_argo_dataset_and_tumo_cycles(
        ['1st_70None', '2nd_NoneNone'],
        dataset_path,
        ref_folder,
        total_moc_path,
        antilles_current_path,
        florida_current_path,
        wind_stress_path,
        time_smoothing,
        deep_argo,
        missing_values_in_target
    )


    ds_argo_merged = filter_ds_argo_data(ds_argo_merged, deep_argo)


    test_year_count = 20

    year_min = ds_argo_merged.time.dt.year.min()
    year_max = ds_argo_merged.time.dt.year.max()
    stride_test_window = 10
    validation_years_on_each_side = 5

    start_years = np.arange(year_min + validation_years_on_each_side +1, year_max - test_year_count - 1, stride_test_window)

    for test_start_year in start_years:
         


        train_dataset, val_dataset, test_dataset, (total_moc_mean, total_moc_std, geostrophic_moc_mean, geostorphic_moc_std) = split_dataset_into_training_validation_testing_profile_datasets(
            False, test_start_year, test_start_year + test_year_count, validation_years_on_each_side, t_umo_obs, ds_argo_merged, add_tmp, add_sal, deep_argo, False, total_moc, florida_current, antilles_current, wind_stress, dv_dz_obs, missing_values_in_target, ds_pos_sim, compartments, time_smoothing, lat_bounds, using_transport_from_previous_year, using_missing_indices, use_deep_dvdz
        )


        dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=merge_profiles_max_profiles, num_workers=8)
        val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=merge_profiles_max_profiles, num_workers=4)


        n_features = train_dataset.X.shape[2]


        for version in ['v6-wo-argo', 'v5-std-embedmean', 'v6-aux']:      
            r2_scores = []
            mae_scores = []
            mse_scores = []

            if version == 'v5-full':
                argo_mean_in_gnn = True
                argo_mean_in_embedding = False
                n_compartments = 3
            elif version == 'v5-std':
                argo_mean_in_gnn = False
                argo_mean_in_embedding = False
                n_compartments = 13
            elif version == 'v5-std-embedmean':
                argo_mean_in_gnn = False
                argo_mean_in_embedding = True
                n_compartments = 27
            elif version == 'v6':
                use_argo_mean = True
                use_argo_std = False
                use_auxiliry = True
            elif version == 'v6-wo-argo':
                use_argo_mean = False
                use_argo_std = False
                use_auxiliry = True
            elif version == 'v6-aux':
                use_argo_mean = True
                use_argo_std = False
                use_auxiliry = False


            for i in range(n_iters):

                torch.random.manual_seed(random_seeds[i])
                np.random.seed(random_seeds[i])

                node_assigner = init_assignment_module(distance_assignment_module, train_dataset, n_compartments, device)
                if version.startswith('v5'):
                    model = ProfileModelSUSTeR5_fast(
                        n_features, n_compartments,n_embedding, 
                        train_dataset.dv_dz.shape[1] , dv_dz_obs.z, device, profile_embedder=None, node_assigner= node_assigner,
                        argo_mean_in_embedding=argo_mean_in_embedding, argo_mean_in_gnn_input=argo_mean_in_gnn).to(device)
                elif version.startswith('v6'):
                    model = ProfileModelSUSTeR6(
                        n_features, n_compartments,n_embedding, 
                        train_dataset.dv_dz.shape[1] , dv_dz_obs.z, device, profile_embedder=None, node_assigner= node_assigner,
                        use_argo_mean=use_argo_mean, use_argo_std = use_argo_std, use_auxiliry = use_auxiliry).to(device)
                else:
                    raise ValueError('Unknown version')





                best_model = train(model, dl, val_dl, device, verbose = False, geostrophic_target= geostrophic_prediction)

                if geostrophic_prediction:
                    test_predictions, (test_hidden_spaces, test_embedding_spaces, test_gt_transport, test_inner_values) = make_predictions(test_dataset, best_model, ds_argo_merged.isel(time = test_dataset.global_indices).time, geostrophic_moc_mean, geostorphic_moc_std,device = device, geostrophic_target=geostrophic_prediction)
                else:
                    test_predictions, (test_hidden_spaces, test_embedding_spaces, test_gt_transport, test_inner_values) = make_predictions(test_dataset, best_model, ds_argo_merged.isel(time = test_dataset.global_indices).time, total_moc_mean, total_moc_std,device = device)



                target_variable = test_gt_transport


                mae_error = mean_absolute_error(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values)
                
                r2_scores.append(r2_score(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values))
                mae_scores.append(mae_error)
                mse_scores.append(mean_squared_error(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values))
                
                print(f'\t R2 {r2_scores[-1]*100:.2f}%; MAE {mae_scores[-1]:.2f}; MSE {mse_scores[-1]:.2f}')


            print(f'{version} {test_start_year}-{test_start_year+test_year_count} -' +
                f'R2 mean {np.mean(r2_scores)*100:.2f}% std {np.std(r2_scores)*100:.2f}%; ' +
                f'MAE mean {np.mean(mae_scores):.2f} std {np.std(mae_scores):.2f}; ' +
                f'MSE mean {np.mean(mse_scores):.2f} std {np.std(mse_scores):.2f}; ')

        del(dl)
        del(val_dl)

if __name__ == '__main__':
    run_experiment()