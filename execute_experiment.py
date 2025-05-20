import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from amoc_reconstruction.reconstruction.model import init_assignment_module
from amoc_reconstruction.reconstruction.model import ProfileModelSUSTeR5_fast, ProfileModelSUSTeR6, ProfileModelSUSTeR7
from amoc_reconstruction.train import train, make_predictions
from amoc_reconstruction.reconstruction.dataset import merge_profiles_max_profiles
from amoc_reconstruction.reconstruction.dataset import load_merged_argo_dataset_and_tumo_cycles, filter_ds_argo_data, split_dataset_into_training_validation_testing_profile_datasets, ProfileDataset
from amoc_reconstruction.utils.plots import prediction_plot 
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import xarray as xr

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_arguments():

    parser = argparse.ArgumentParser(description='Execute experiment')
    
    parser.add_argument('--input_feat', type=str, default='ArFcAcWs', help = 'Input features ordering does not matter [Ar-Argo, Fc-Florida Current, Ac-Antilles Current, Ws-Wind Stress]')

    parser.add_argument('--input_smoothing', type=int, default=10, help = 'Input smoothing window size')
    parser.add_argument('--output_smoothing', type=int, default=None, help = 'Output smoothing window size (default = input_smoothing)')
    parser.add_argument('--depth_information', type=str, default = 'mooring', choices=['none', 'deep', 'mooring'])
    parser.add_argument('--moc_type', choices=['total', 'geostrophic'], default='total')

    parser.add_argument('--input_cycles', type=str, default='1,2', help='A comma sparated list of the input cycles. For all cycles \'all\' is possible.') 
    parser.add_argument('--test_cycle', default = None, type=int, help='Cycle to be used for testing. This attribute or test_period has to be set but not both.')
    parser.add_argument('--test_period', default='2004,2024', type=str, help='Period to be used for testing as a string of an inclusive lower year and an explusive upper bound (e.g. 2005,2020). This attribute or test_cycle has to be set but not both.')   
    parser.add_argument('--validation_length', type=int, default=5, help='Length of the validation period in years for each side of the test period.')
    parser.add_argument('--n_iters', type=int, default=11, help='Number of iterations to run the experiment.')

    parser.add_argument('--paperdraft_index', type=str, default=None, help='Index of the paperdraft to be used for the experiment')

    parser.add_argument('--n_compartments', type = int, default = None, help = 'Number of compartments to be used in the model. Default is 27 for 10D, 30D and 90D and 100 for 365D and 1825D.')
    parser.add_argument('--n_embedding', type=int, default=None, help='Number of embedding dimensions to be used in the model. Default is 12 for 10D, 30D and 90D and 8 for 365D and 1825D.')
    parser.add_argument('--verbose_training', type=str2bool, default=False, help='Verbose training output')
    return parser.parse_args()

def compute_skill(ground_truth, prediction):

    time_selected_truth = ground_truth.sel(time = prediction.time, method = "nearest").values
    prediction = prediction.values
    return {
        'MAE': mean_absolute_error(time_selected_truth, prediction),
        'R2': r2_score(time_selected_truth, prediction),
        'RMSE': root_mean_squared_error(time_selected_truth, prediction),
        'MAPE': mean_absolute_percentage_error(time_selected_truth, prediction)
    }


paper_draft_path = Path('../rapid-geostrophic-reconstruction/paperdraft.json')
def is_paperdraft_experiment_started(paper_draft_id):
    if paper_draft_id is None:
        return False

    if paper_draft_path.exists():
        with open('../rapid-geostrophic-reconstruction/paperdraft.json', 'r') as f:
            paper_draft_status = json.load(f)
    else:
        return False

    return paper_draft_id in paper_draft_status

def log_paperdraft_experiment_started(experiment_path, paper_draft_id):
    if paper_draft_id is None:
        return

    if paper_draft_path.exists():
        with open('../rapid-geostrophic-reconstruction/paperdraft.json', 'r') as f:
            paper_draft_status = json.load(f)
    else:
        paper_draft_status = {}

    paper_draft_status[paper_draft_id] = {
        'status': 'running',
        'experiment_path': str(experiment_path), 
        'start_time': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    with open('../rapid-geostrophic-reconstruction/paperdraft.json', 'w') as f:
        json.dump(paper_draft_status, f, indent=4)

def log_paperdraft_experiment_finished(paper_draft_id):
    if paper_draft_id is None:
        return

    with open('../rapid-geostrophic-reconstruction/paperdraft.json', 'r') as f:
        paper_draft_status = json.load(f)

    paper_draft_status[paper_draft_id]['status'] = 'finished'
    paper_draft_status[paper_draft_id]['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open('../rapid-geostrophic-reconstruction/paperdraft.json', 'w') as f:
        json.dump(paper_draft_status, f, indent=4)


"""




"""
def main(args):
    smoothing_days = args.input_smoothing
    time_smoothing = f'{smoothing_days}D'

    lat_bounds = (25, 30)
    target_reference_level = 4800

    if args.depth_information == 'mooring':
        missing_values_in_target = True
        using_missing_indices = True
    else:
        missing_values_in_target = False
        using_missing_indices = False

    if args.moc_type == 'total':
        missing_values_in_target = False
        using_missing_indices = False
        using_transport_from_previous_year = False
        geostrophic_prediction = False
    elif args.moc_type == 'geostrophic':
        
        using_transport_from_previous_year = False
        geostrophic_prediction = True
    else:
        raise NotImplementedError(f'MOC type {args.moc_type} is not supported')

    if args.depth_information == 'none':
        use_deep_dvdz = False
        deep_argo = False
    elif args.depth_information == 'deep':
        use_deep_dvdz = False
        deep_argo = True
    elif args.depth_information == 'mooring':
        use_deep_dvdz = True
        deep_argo = False
    else:
        raise NotImplementedError(f'Depth information {args.depth_information} is not supported')
    



    train_batch_sizes = {
        '10D': 32,
        '30D': 16,
        '90D': 16,
        '365D': 6,
        '1825D': 4
    }


    n_compartments_for_smoothing = {
        '10D': 27,
        '30D': 27,
        '90D': 27,
        '365D': 100 ,
        '1825D': 100,
    }

    train_batch_size = train_batch_sizes[time_smoothing]

    add_tmp = False
    add_sal = False


    if args.test_period is not None:
        test_start_year, test_end_year = map(int, args.test_period.split(','))
    validation_years_on_each_side = args.validation_length
        
    assert target_reference_level in [2000, 4800], "Only 2000 and 4800 are supported as reference levels"
    
    
    if args.n_compartments is not None:
        n_compartments = args.n_compartments
    else:
        n_compartments = n_compartments_for_smoothing[time_smoothing]
    n_embedding = 12 if args.input_smoothing < 365 else 8

    import uuid

    experiment_path = Path(f'../rapid-geostrophic-reconstruction/figs/experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:4]}')
    dataset_path = Path(f'../rapid-geostrophic-reconstruction/datasets/smoothing_{smoothing_days}_days/argo_after_2012/paperdraft/') # TODO: Change when changing cycles

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

    lr = 1e-3 if args.input_smoothing < 365 else 3e-3
    wd = 1e-6 if args.input_smoothing < 365 else 1e-5

    if is_paperdraft_experiment_started(args.paperdraft_index):
        print(f'Experiment {args.paperdraft_index} is already running')
        return

    ### Experiment 20012025
    import numpy as np

    n_iters = args.n_iters
    random_seeds = np.random.randint(0, 1000, n_iters)



    cycle_suffixe = ['1st_7024', '2nd_5824', '3rd_5824', '4th_5824', '5th_5824', '6th_5824']
    # cycle_suffixe = ['1st_7020', '2nd_5820', '3rd_5820', '4th_5820', '5th_5820', '6th_5820']
    # cycle_suffixe = ['1st_70None', '2nd_NoneNone']

    if args.input_cycles == 'all':
        selected_cycles = cycle_suffixe
    else:
        selected_cycles = [cycle_suffixe[i-1] for i in map(int, args.input_cycles.split(','))]

    ds_argo_merged, t_umo_obs, ds_pos_sim, dv_dz_obs, t_deltas, total_moc, antilles_current, florida_current, wind_stress, time_backwardshift = load_merged_argo_dataset_and_tumo_cycles(
        selected_cycles,
        dataset_path,
        ref_folder,
        total_moc_path,
        antilles_current_path,
        florida_current_path,
        wind_stress_path,
        time_smoothing,
        True, # This has to be true in every case to load the right files becuase for the paperdraft there were only deepargos extracted
        missing_values_in_target
    )


    ds_argo_merged = filter_ds_argo_data(ds_argo_merged, deep_argo)


    if args.output_smoothing is not None and args.output_smoothing > args.input_smoothing:

        rolling_factor = args.output_smoothing // args.input_smoothing

        total_moc = total_moc.rolling(time=rolling_factor, min_periods=1, center=True).mean()
        t_umo_obs = t_umo_obs.rolling(time=rolling_factor, min_periods=1, center=True).mean()


    def get_years_from_cycle_suffix(cycle_suffix):
        start_year = int(cycle_suffix[-4:-2]) + 1900
        end_year = int(cycle_suffix[-2:]) + 2000

        return start_year, end_year

    if args.test_cycle is not None:


        if args.test_cycle >= 3 and args.test_cycle <= 6:
            start_year_cycle, end_year_cycle = get_years_from_cycle_suffix(cycle_suffixe[args.test_cycle - 1])

            start_date = np.datetime64(f'{start_year_cycle}-01-01') + t_deltas[args.test_cycle -2] + np.timedelta64(smoothing_days, 'D')
            end_date = start_date + np.timedelta64( (end_year_cycle - start_year_cycle) * 365, 'D') 

            test_start_year = start_date.values.astype('datetime64[Y]').astype(int) + 1970
            test_end_year = end_date.values.astype('datetime64[Y]').astype(int) + 1970

            time_backwardshift = None # we can set this to None here because we used it implicit in the t_deltas such that test_start_year and test_end_year are also considering that the first cycle starts now at 1800

        else:
            raise NotImplementedError(f'Test cycle {args.test_cycle} is not supported')
    


    train_dataset, val_dataset, test_dataset, (total_moc_mean, total_moc_std, geostrophic_moc_mean, geostorphic_moc_std) = split_dataset_into_training_validation_testing_profile_datasets(
        False, test_start_year, test_end_year, validation_years_on_each_side, t_umo_obs, ds_argo_merged, add_tmp, add_sal, deep_argo, False, total_moc, florida_current, antilles_current, wind_stress, dv_dz_obs, missing_values_in_target, ds_pos_sim, compartments, time_smoothing, lat_bounds, using_transport_from_previous_year, using_missing_indices, use_deep_dvdz, time_backwardshift
    )

    def input_setting_dataset(dataset):
        dataset.use_ac_input = 'Ac' in args.input_feat
        dataset.use_fc_input = 'Fc' in args.input_feat
        dataset.use_ws_input = 'Ws' in args.input_feat
        dataset.use_ar_input = 'Ar' in args.input_feat

    input_setting_dataset(train_dataset)
    input_setting_dataset(val_dataset)
    input_setting_dataset(test_dataset)


    dl = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=merge_profiles_max_profiles, num_workers=8)
    val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=merge_profiles_max_profiles, num_workers=4)
    # test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=merge_profiles_max_profiles, num_workers=4)

    log_paperdraft_experiment_started(experiment_path, args.paperdraft_index)


    n_features = train_dataset.X.shape[2]

    r2_scores = []
    mae_scores = []
    mse_scores = []
    mape_scores = []

    test_predictions_members = []
    test_ground_truth_members = []

    member_metrics = []

    for i in range(n_iters):

        torch.random.manual_seed(random_seeds[i])
        np.random.seed(random_seeds[i])

        node_assigner = init_assignment_module('distance', train_dataset, n_compartments, device)
        model = ProfileModelSUSTeR5_fast(
            n_features, n_compartments,n_embedding, 
            train_dataset.dv_dz.shape[1] , dv_dz_obs.z, device, profile_embedder=None, node_assigner= node_assigner,
            argo_mean_in_embedding=True, argo_mean_in_gnn_input=False,assignment_threshold=.1, embedding_dropout=.34
        ).to(device)

    
        best_model = train(model, dl, val_dl, device, verbose = args.verbose_training, geostrophic_target= geostrophic_prediction, lr=lr, wd = wd, num_epochs = 80)

        if geostrophic_prediction:
            test_predictions, (test_hidden_spaces, test_embedding_spaces, test_gt_transport, test_inner_values) = make_predictions(test_dataset, best_model, ds_argo_merged.isel(time = test_dataset.global_indices).time, geostrophic_moc_mean, geostorphic_moc_std,device = device, geostrophic_target=True)
        else:
            test_predictions, (test_hidden_spaces, test_embedding_spaces, test_gt_transport, test_inner_values) = make_predictions(test_dataset, best_model, ds_argo_merged.isel(time = test_dataset.global_indices).time, total_moc_mean, total_moc_std,device = device)



        target_variable = test_gt_transport


        mae_error = mean_absolute_error(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values)
        
        r2_scores.append(r2_score(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values))
        mae_scores.append(mae_error)
        mse_scores.append(root_mean_squared_error(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values))
        mape_scores.append(mean_absolute_percentage_error(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values))
        
        print(f'\t R2 {r2_scores[-1]*100:.2f}%; MAE {mae_scores[-1]:.2f}; RMSE {mse_scores[-1]:.2f}; MAPE {mape_scores[-1]*100:.2f}%')


        # Save the model
        member_experiment_path = experiment_path / f'member_{i:02d}'
        member_experiment_path.mkdir(parents=True, exist_ok=True)
        model_path = member_experiment_path / f'model_{i}.pt'

        torch.save(best_model.state_dict(), model_path)

        # Save the predictions
        test_predictions_path = member_experiment_path / 'test_predictions.nc'
        test_prediction_ds = xr.Dataset(
            {'test_predictions': test_predictions, 
            'test_gt_transport': test_gt_transport} ,

            attrs={
                    'MAE': mae_scores[-1],
                    'R2': r2_scores[-1],
                    'RMSE': mse_scores[-1],
                    'MAPE': mape_scores[-1]
                } 
        )
        test_prediction_ds.to_netcdf(test_predictions_path)

        test_predictions_members.append(test_predictions)
        test_ground_truth_members.append(test_gt_transport)

        # save the plots
        prediction_plot(member_experiment_path, target_variable, test_predictions, None)


        # Save the test dataset for reference
        test_dataset_path = member_experiment_path / 'test_dataset.pt'
        torch.save(test_dataset, test_dataset_path)


        # Execute the importance experiment
        permutation_skill = {}
        for feature in ['Ar', 'Fc', 'Ac', 'Ws']:

            if feature not in args.input_feat:
                continue

            skill_records = []

            for _ in range(10):
                rand_indices = np.arange(len(test_dataset))
                np.random.shuffle(rand_indices)

                argo_indices = np.arange(len(test_dataset))
                fc_indices = np.arange(len(test_dataset))
                ac_indices = np.arange(len(test_dataset))
                ws_indices = np.arange(len(test_dataset))

                if feature == 'Ar':
                    argo_indices = rand_indices
                elif feature == 'Fc':
                    fc_indices = rand_indices
                elif feature == 'Ac':
                    ac_indices = rand_indices
                elif feature == 'Ws':
                    ws_indices = rand_indices


                test_dataset_wo_feature = ProfileDataset(
                    test_dataset.X[argo_indices], 
                    test_dataset.y[argo_indices], 
                    test_dataset.y_prev, 
                    test_dataset.mask[argo_indices], 
                    test_dataset.lon[argo_indices], 
                    test_dataset.lat[argo_indices], 
                    test_dataset.dv_dz[argo_indices], 
                    test_dataset.days[argo_indices], 
                    compartments, 
                    test_dataset.missing_indices[argo_indices], 
                    test_dataset.fs[fc_indices], 
                    test_dataset.ac[ac_indices], 
                    test_dataset.ws[ws_indices], 
                    test_dataset.total_moc,
                    test_dataset.time_smoothing, 
                    test_dataset.lat_bounds, 
                    test_dataset.using_transport_from_previous_year, 
                    test_dataset.using_missing_indices, 
                    test_dataset.use_deep_dvdz, 
                    test_dataset.global_indices)
                
                if geostrophic_prediction:
                    test_predictions, (test_hidden_spaces, test_embedding_spaces, test_gt_transport, test_inner_values) = make_predictions(test_dataset_wo_feature, best_model, ds_argo_merged.isel(time = test_dataset.global_indices).time, geostrophic_moc_mean, geostorphic_moc_std,device = device, geostrophic_target=True)
                else:
                    test_predictions, (test_hidden_spaces, test_embedding_spaces, test_gt_transport, test_inner_values) = make_predictions(test_dataset_wo_feature, best_model, ds_argo_merged.isel(time = test_dataset.global_indices).time, total_moc_mean, total_moc_std,device = device)
                
                target_variable = test_gt_transport

                skill_json = compute_skill(target_variable, test_predictions)
                skill_records.append(skill_json)

            permutation_skill[feature] = {
                'mean': {
                    'MAE': np.mean([s['MAE'] for s in skill_records]),
                    'R2': np.mean([s['R2'] for s in skill_records]),
                    'RMSE': np.mean([s['RMSE'] for s in skill_records]),
                    'MAPE': np.mean([s['MAPE'] for s in skill_records])
                },
                'std': {
                    'MAE': np.std([s['MAE'] for s in skill_records]),
                    'R2': np.std([s['R2'] for s in skill_records]),
                    'RMSE': np.std([s['RMSE'] for s in skill_records]),
                    'MAPE': np.std([s['MAPE'] for s in skill_records])
                }
            }


        # Execute the averaging experiment
        smoothing_skill = {}
        for smoothing_days in [10, 30, 90, 365, 1825]:
            
            relevant_smoothing = args.input_smoothing if args.output_smoothing is None else args.output_smoothing
            if smoothing_days <= relevant_smoothing:
                continue

            smoothed_ds = test_prediction_ds.resample(time=f'{smoothing_days}D').mean()
            
            skill = compute_skill(smoothed_ds.test_gt_transport, smoothed_ds.test_predictions)
            smoothing_skill[f'{smoothing_days}D'] = {
                'mean': skill
            }

            prediction_plot(member_experiment_path, smoothed_ds.test_gt_transport, smoothed_ds.test_predictions, f'smoothed_{smoothing_days}D')

            



        # save the metrics.json
        metrics = {
            'original': {
                'mean' : {'MAE': mae_scores[-1], 'R2': r2_scores[-1], 'RMSE': mse_scores[-1], 'MAPE': mape_scores[-1]},
            },
            'smoothed': smoothing_skill,
            'permutation': permutation_skill
        }

        metrics_path = member_experiment_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        member_metrics.append(metrics)
        


    # save all testresults 

    ensemble_predictions = xr.Dataset(
        {
            'test_predictions': xr.concat(test_predictions_members, dim='member'),
            'test_gt_transport': xr.concat(test_ground_truth_members, dim='member'),
            'mae': xr.DataArray(mae_scores, dims=['member']),
            'r2': xr.DataArray(r2_scores, dims=['member']),
            'rmse': xr.DataArray(mse_scores, dims=['member']),
            'mape': xr.DataArray(mape_scores, dims=['member'])
        }
    )

    ensemble_predictions_path = experiment_path / 'ensemble_predictions.nc'
    ensemble_predictions.to_netcdf(ensemble_predictions_path)

    # Save the ensemble metrics.json

    ensemble_metrics = {
        'original': {
            'mean': {
                'MAE': np.mean(mae_scores),
                'R2': np.mean(r2_scores),
                'RMSE': np.mean(mse_scores),
                'MAPE': np.mean(mape_scores)
            },
            'std': {
                'MAE': np.std(mae_scores),
                'R2': np.std(r2_scores),
                'RMSE': np.std(mse_scores),
                'MAPE': np.std(mape_scores)
            }
        },
    }

    ensemble_metrics['smoothed'] = {}
    for smoothing in member_metrics[0]['smoothed'].keys():

        ensemble_metrics['smoothed'][smoothing] = {
            'mean': {
                'MAE': np.mean([s['smoothed'][smoothing]['mean']['MAE'] for s in member_metrics]),
                'R2': np.mean([s['smoothed'][smoothing]['mean']['R2'] for s in member_metrics]),
                'RMSE': np.mean([s['smoothed'][smoothing]['mean']['RMSE'] for s in member_metrics]),
                'MAPE': np.mean([s['smoothed'][smoothing]['mean']['MAPE'] for s in member_metrics])
            },
            'std': {
                'MAE': np.std([s['smoothed'][smoothing]['mean']['MAE'] for s in member_metrics]),
                'R2': np.std([s['smoothed'][smoothing]['mean']['R2'] for s in member_metrics]),
                'RMSE': np.std([s['smoothed'][smoothing]['mean']['RMSE'] for s in member_metrics]),
                'MAPE': np.std([s['smoothed'][smoothing]['mean']['MAPE'] for s in member_metrics])
            }
        }

    ensemble_metrics['permutation'] = {}
    for feature in ['Ar', 'Fc', 'Ac', 'Ws']:
        if feature not in args.input_feat:
            continue

        feature_metrics = {
            'mean': {
                'MAE': np.mean([s['permutation'][feature]['mean']['MAE'] for s in member_metrics]),
                'R2': np.mean([s['permutation'][feature]['mean']['R2'] for s in member_metrics]),
                'RMSE': np.mean([s['permutation'][feature]['mean']['RMSE'] for s in member_metrics]),
                'MAPE': np.mean([s['permutation'][feature]['mean']['MAPE'] for s in member_metrics])
            },
            'std': {
                'MAE': np.std([s['permutation'][feature]['mean']['MAE'] for s in member_metrics]),
                'R2': np.std([s['permutation'][feature]['mean']['R2'] for s in member_metrics]),
                'RMSE': np.std([s['permutation'][feature]['mean']['RMSE'] for s in member_metrics]),
                'MAPE': np.std([s['permutation'][feature]['mean']['MAPE'] for s in member_metrics])
            }
        }

        ensemble_metrics['permutation'][feature] = feature_metrics

    ensemble_metrics_path = experiment_path / 'ensemble_metrics.json'
    with open(ensemble_metrics_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=4)

        
    # Save configuration


    configuration = {
        'input_feat': args.input_feat,
        'input_smoothing': args.input_smoothing,
        'output_smoothing': args.output_smoothing,
        'depth_information': args.depth_information,
        'moc_type': args.moc_type,
        'input_cycles': args.input_cycles,
        'test_cycle': args.test_cycle,
        'test_period': args.test_period,
        'validation_length': args.validation_length,
        'n_iters': args.n_iters,
        'lr': lr,
        'embedding_dim': n_embedding,
        'weight_decay': wd,
        'seeds': [int(rs) for rs in random_seeds],
        'test_start_year': int(test_start_year),
        'test_end_year': int(test_end_year),
        'backward_shift': time_backwardshift.values.astype(int).astype(str) if time_backwardshift is not None else None,
        't_deltas': [td.values.astype(int).astype(str) for td in t_deltas],
    }

    print(configuration)


    configuration_path = experiment_path / 'configuration.json'
    with open(configuration_path, 'w') as f:
        json.dump(configuration, f, indent=4)




    log_paperdraft_experiment_finished(args.paperdraft_index)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
    