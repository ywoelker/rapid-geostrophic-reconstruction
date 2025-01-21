import matplotlib.pyplot as plt


def time_window_plot():
    ds_argo_merged_obs, t_umo_obs_obs = load_merged_argo_dataset_and_tumo_obs(time_smoothing)#, lat_bounds = lat_bounds)


    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(2,1,1)

    train1 = t_umo_10days.isel(time = train_indices[train_indices <= 101]).dv_dz_times_X
    ax.plot(train1.time, train1.values, c = 'b', label = 'Train')
    train2 = t_umo_10days.isel(time = train_indices[train_indices > 101]).dv_dz_times_X
    ax.plot(train2.time, train2.values, c = 'b')


    val1 = t_umo_10days.isel(time = val_indices[val_indices < 150]).dv_dz_times_X
    val2 = t_umo_10days.isel(time = val_indices[val_indices > 150]).dv_dz_times_X

    ax.plot(val1.time, val1.values, c = 'g', label ='Validation')
    ax.plot(val2.time, val2.values, c = 'g')

    test = t_umo_10days.isel(time = test_indices).dv_dz_times_X
    ax.plot(test.time, test.values, c = 'r', label = 'Test')

    ax.set_xlim(t_umo_10days.time.min().data, t_umo_10days.time.max().data)
    ax.set_ylim(t_umo_10days.dv_dz_times_X.min().data, t_umo_10days.dv_dz_times_X.max().data)
    ax.set_xticklabels([])
    ax.set_title('VIKING20X (1st and 2nd cycle)')
    ax.set_ylabel('Transport [Sv]')
    plt.legend()
    ax = fig.add_subplot(2,1,2)
    ax.set_xlim(t_umo_10days.time.min().data, t_umo_10days.time.max().data)
    ax.set_ylim(t_umo_10days.dv_dz_times_X.min().data, t_umo_10days.dv_dz_times_X.max().data)

    ax.set_title('RAPID')
    ax.plot(t_umo_obs_obs.dv_dz_times_X.time, t_umo_obs_obs.dv_dz_times_X.values, c = 'r')
    ax.set_ylabel('Transport [Sv]')
    fig.tight_layout()
    fig.savefig('../rapid-geostrophic-reconstruction/train_val_test_split.png')


def gt_to_prediction_scatter():
    import matplotlib.pyplot as plt
    plt.scatter(target_variable.sel(time = test_predictions.time, method = "nearest").values, test_predictions.values)
    # plt.scatter(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values, t_merged_argo_2000.sel(time = transport.time, method = 'nearest').values)
    plt.xlabel('Calculated AMOC')
    plt.ylabel('Reconstructed AMOC')
    min_transport = min(test_predictions.min(), target_variable.min()).values[()]
    max_transport = max(test_predictions.max(), target_variable.max()).values[()]
    plt.plot([min_transport, max_transport], [min_transport, max_transport], 'r--')

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression().fit(target_variable.sel(time = test_predictions.time, method = "nearest").values.reshape(-1, 1), test_predictions.values.reshape(-1, 1))
    # in_X = np.linspace(-35, -5, 100).reshape(-1, 1)
    # plt.plot(in_X, lr.predict(in_X), 'k--')
    # lr = LinearRegression().fit(t_umo_obs.sel(time = transport.time, method = "nearest").dv_dz_times_X.values.reshape(-1, 1), t_merged_argo_2000.sel(time = transport.time, method = 'nearest').values.reshape(-1, 1))
    in_X = np.linspace(min_transport, max_transport, 100).reshape(-1, 1)
    plt.plot(in_X, lr.predict(in_X), 'g--')

    plt.savefig(experiment_path / f'AMOC_reconstruction.png')


def prediction_plot():
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1,1,1)

    # calc = t_merged_argo_2000.sel(time = transport.time, method = 'nearest')
    # ax.plot(calc.time, calc.values, label = 'Geostrohphic calculation from Argos', alpha = .5, color = 'black')
    # rapid = t_umo_obs.sel(time = test_predictions.time, method = 'nearest')
    rapid = target_variable.sel(time = test_predictions.time, method = 'nearest')

    test_predictions_t = test_predictions#.resample(time = '365D').mean()
    rapid_t = rapid #t_umo_90days.sel(time = test_predictions_t.time, method = 'nearest')#
    # rapid_t = rapid.resample(time = '365D').mean()


    ax.plot(test_predictions_t.time, test_predictions_t.values, label = 'Reconstruction (only argo)', linewidth = 2, zorder = 20)
    ax.plot(rapid_t.time, rapid_t.values, label = 'Geostrophic Rapid measurements', linewidth = 2)

    plt.legend()
    ax.set_ylabel('Transport [Sv]')
    ax.set_xlabel('Time')

    # plt.text(0.01, 0.9, f'R2 {r2_score(t_umo_obs.sel(time = test_predictions.time, method = "nearest").dv_dz_times_X.values, test_predictions.values)*100:.2f}%; MAE {mae_error:.2f}; MSE {mean_squared_error(t_umo_obs.sel(time = test_predictions.time, method = "nearest").dv_dz_times_X.values, test_predictions.values):.2f}', transform=ax.transAxes)
    plt.text(0.01, 0.9,f'R2 {r2_score(rapid_t.values, test_predictions_t.values)*100:.2f}%; MAE {mean_absolute_error(rapid_t.values, test_predictions_t.values):.2f}; MSE {mean_squared_error(rapid_t.values, test_predictions_t.values):.2f}', transform=ax.transAxes)
    plt.title('AMOC Reconstruction at Rapid Latitude 26.5°N by Argo profiles')
    fig.tight_layout()
    plt.savefig(experiment_path / f'argo_nn_reconstruction.png')