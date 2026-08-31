import dataset_reading
from usefull_functions import run_pca

sep_pret = dataset_reading.load_sep_pret_v2()
sep_pret_reduced = dataset_reading.load_sep_pret_v2_reduced()
# sep_pret_sample_03 = dataset_reading.load_sep_pret_v2_sample_03()
# sep_pret_sample_50 = dataset_reading.load_sep_pret_v2_sample_50()

# sep_pret_removed = sep_pret.loc[~sep_pret.index.isin(sep_pret_reduced.index)]


#%%

sep_pret_reduced_pca = sep_pret_reduced[[ 'S_class', 'fl_goes_xray', 'lasco_linear_speed', 'lasco_cme_width', 
                                         'fl_lat', 
                                         'fl_lon', 
                                         'fl_total_time', 
                                         'daily_sn', 'AR_Hale_int', 'AR_Mcintosh_int']]

_,_,_ = run_pca(sep_pret_reduced_pca, correlation_circle=True)
_,_,_ = run_pca(sep_pret_reduced_pca, correlation_circle=True, n_components=3)