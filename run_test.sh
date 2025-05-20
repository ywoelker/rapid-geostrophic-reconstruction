# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-1st_70None.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_10_days/all_argo_samples/base_sample/time_mapping_dict.pickle --cycle_name KFS003-1st_70None --reference_level -4800 --time_smooth 10D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-1st_70None.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_10_days/all_argo_samples/base_sample/time_mapping_dict.pickle --cycle_name KFS003-1st_70None --reference_level -2000 --time_smooth 10D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-2nd_NoneNone.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_10_days/all_argo_samples/base_sample/time_mapping_dict_2nd_NoneNone.pickle --cycle_name KFS003-2nd_NoneNone --reference_level -4800 --time_smooth 10D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-2nd_NoneNone.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_10_days/all_argo_samples/base_sample/time_mapping_dict_2nd_NoneNone.pickle --cycle_name KFS003-2nd_NoneNone --reference_level -2000 --time_smooth 10D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-1st_70None.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_365_days/argo_after_2012/base_sample/time_mapping_dict_1st_70None.pickle --cycle_name KFS003-1st_70None --reference_level -4800 --time_smooth 365D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-1st_70None.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_365_days/argo_after_2012/base_sample/time_mapping_dict_1st_70None.pickle --cycle_name KFS003-1st_70None --reference_level -2000 --time_smooth 365D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-2nd_NoneNone.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_365_days/argo_after_2012/base_sample/time_mapping_dict_2nd_NoneNone.pickle --cycle_name KFS003-2nd_NoneNone --reference_level -4800 --time_smooth 365D;

# python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_groups_KFS003-2nd_NoneNone.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_365_days/argo_after_2012/base_sample/time_mapping_dict_2nd_NoneNone.pickle --cycle_name KFS003-2nd_NoneNone --reference_level -2000 --time_smooth 365D;



## make a for loop of the strings ['1st_7020', '2nd_5820', '3rd_5820', '4th_5820', '5th_5820', '6th_5820']
# for cycle in '1st_7020' '2nd_5820' '3rd_5820' '4th_5820' '5th_5820' '6th_5820'; do
for cycle in '1st_7024' '2nd_5824' '3rd_5824' '4th_5824' '5th_5824' '6th_5824'; do
    
    ## make a for loop of the strings ['90D', '365D', '1825D']
    for time_smooth in '90' '365' '1825'; do
    # for time_smooth in '10' '30'; do

        for reference_level in -4800 -2000; do

            echo ${cycle} ${time_smooth} ${reference_level};
            python data_scripts/calc_transport_from_moorings.py --mooring_group_sim datasets/mooring_data/mooring_groups_${cycle}.nc --ts_gridded datasets/ts_gridded.nc --time_mapping_dict datasets/smoothing_${time_smooth}_days/argo_after_2012/paperdraft/time_mapping_dict_${cycle}.pickle --cycle_name KFS003-${cycle} --reference_level ${reference_level} --time_smooth ${time_smooth}D;

        done

    done

done
