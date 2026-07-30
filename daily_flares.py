import dataset_reading

noaa_flares = dataset_reading.load_noaa_flares_c1()

noaa_flares_daily_counts = (
    noaa_flares.set_index('time_start')
      .resample('D')
      .agg(
          count=('xray_flux', 'size'),
          xray_average=('xray_flux', 'mean'),
          xray_max=('xray_flux', 'max')
      )
      .reset_index()
      .rename(columns={'time_start': 'date'})
)

#%%

import matplotlib.pyplot as plt

plt.plot( noaa_flares_daily_counts['date'], noaa_flares_daily_counts['xray_max'],)
plt.plot( noaa_flares_daily_counts['date'], noaa_flares_daily_counts['xray_average'])

