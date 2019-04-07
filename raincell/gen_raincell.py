import json
import traceback
from tempfile import TemporaryDirectory
import os
import numpy as np
from curw.rainfall.wrf.extraction.observation_utils import CurwObservationException
from mpl_toolkits.basemap import Basemap, cm
from netCDF4 import Dataset
import pandas as pd
from datetime import datetime,timedelta
from curwmysqladapter import MySQLAdapter
import decimal


def get_max_min_lat_lon(basin_points_file):
    points = np.genfromtxt(basin_points_file, delimiter=',')
    # points = [[id, longitude, latitude],[],[]]
    kel_lon_min = np.min(points, 0)[1]
    kel_lat_min = np.min(points, 0)[2]
    kel_lon_max = np.max(points, 0)[1]
    kel_lat_max = np.max(points, 0)[2]
    print('[kel_lon_min, kel_lat_min, kel_lon_max, kel_lat_max] : ', [kel_lon_min, kel_lat_min, kel_lon_max, kel_lat_max])
    print(points[0][1])
    print(points[0][2])


def get_observed_precip(obs_stations, start_dt, end_dt, duration_days, adapter, forecast_source='wrf0', ):
    def _validate_ts(_s, _ts_sum, _opts):
        print('len(_ts_sum):', len(_ts_sum))
        print('duration_days[0] * 24 + 1:',duration_days[0] * 24 + 1)
        if len(_ts_sum) == duration_days[0] * 24 + 1:
            return

        f_station = {'station': obs_stations[_s][3],
                     'variable': 'Precipitation',
                     'unit': 'mm',
                     'type': 'Forecast-0-d',
                     'source': forecast_source,
                     }
        f_ts = np.array(adapter.retrieve_timeseries(f_station, _opts)[0]['timeseries'])

        if len(f_ts) != duration_days[0] * 24 + 1:
            raise CurwObservationException('%s Forecast time-series validation failed' % _s)

        for j in range(duration_days[0] * 24 + 1):
            d = start_dt + timedelta(hours=j)
            d_str = d.strftime('%Y-%m-%d %H:00')
            if j < len(_ts_sum.index.values):
                if _ts_sum.index[j] != d_str:
                    _ts_sum.loc[d_str] = f_ts[j, 1]
                    _ts_sum.sort_index(inplace=True)
            else:
                _ts_sum.loc[d_str] = f_ts[j, 1]

        if len(_ts_sum) == duration_days[0] * 24 + 1:
            return
        else:
            raise CurwObservationException('time series validation failed')

    obs = {}
    opts = {
        'from': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        'to': end_dt.strftime('%Y-%m-%d %H:%M:%S'),
    }

    for s in obs_stations.keys():
        print('obs_stations[s][2]: ', obs_stations[s][2])
        station = {'station': s,
                   'variable': 'Precipitation',
                   'unit': 'mm',
                   'type': 'Observed',
                   'source': 'WeatherStation',
                   'name': obs_stations[s][2]
                   }
        print('station : ', s)
        row_ts = adapter.retrieve_timeseries(station, opts)
        if len(row_ts) == 0:
            print('No data for {} station from {} to {} .'.format(s, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')))
        else:
            ts = np.array(row_ts[0]['timeseries'])
            print('ts length:', len(ts))
            if len(ts) != 0 :
                ts_df = pd.DataFrame(data=ts, columns=['ts', 'precip'], index=ts[0:])
                ts_sum = ts_df.groupby(by=[ts_df.ts.map(lambda x: x.strftime('%Y-%m-%d %H:00'))]).sum()
                _validate_ts(s, ts_sum, opts)

        obs[s] = ts_sum
    print('get_observed_precip|success')
    return obs


def get_forecast_precipitation(forecast_stations, start_dt, end_dt, adapter, forecast_source='wrf0', ):
    forecast = {}
    opts = {
        'from': start_dt,
        'to': end_dt,
    }

    for s in forecast_stations:
        station = {'station': s,
                   'variable': 'Precipitation',
                   'unit': 'mm',
                   'type': 'Forecast-0-d',
                   'source': forecast_source
                   }
        print('station : ', s)
        row_ts = adapter.retrieve_timeseries(station, opts)
        if len(row_ts) == 0:
            print('No data for {} station from {} to {} .'.format(s, start_dt, end_dt))
        else:
            ts = np.array(row_ts[0]['timeseries'])
            print('ts length:', len(ts))
            if len(ts) != 0 :
                ts_df = pd.DataFrame(data=ts, columns=['ts', 'precip'], index=ts[0:])
                ts_sum = ts_df.groupby(by=[ts_df.ts.map(lambda x: x.strftime('%Y-%m-%d %H:00'))]).sum()
                forecast[s] = ts_sum
    print('get_forecast_precip|success')
    return forecast


def get_forecast_stations_from_point_file(basin_points_file):
    forecast_stations_list = []
    points = np.genfromtxt(basin_points_file, delimiter=',')
    for point in points:
        forecast_stations_list.append('wrf0_{}_{}'.format(point[1], point[2]))
    return forecast_stations_list


def get_forecast_stations_from_net_cdf(net_cdf_file, min_lat, min_lon, max_lat, max_lon):
    nc_fid = Dataset(net_cdf_file, 'r')
    lats = nc_fid.variables['XLAT'][0, :, 0]
    lons = nc_fid.variables['XLONG'][0, 0, :]

    lon_min_idx = np.argmax(lons >= min_lon) - 1
    lat_min_idx = np.argmax(lats >= min_lat) - 1
    lon_max_idx = np.argmax(lons >= max_lon)
    lat_max_idx = np.argmax(lats >= max_lat)

    lats = lats[lat_min_idx:lat_max_idx]
    lons = lons[lon_min_idx:lon_max_idx]

    stations = []
    for lat in lats:
        for lon in lons:
            station_name = '%s_%.6f_%.6f' % ('wrf0', lon, lat)
            stations.append(station_name)
    return stations


if __name__ == "__main__":
    try:
        config = json.loads(open('/home/hasitha/PycharmProjects/Flo2d/configs/config.json').read())
        kelani_basin_points_file = '/home/hasitha/PycharmProjects/Flo2d/configs/kelani_basin_points_250m.txt'
        reference_net_cdf = '/home/hasitha/PycharmProjects/Flo2d/configs/wrf_wrfout_d03_2019-03-31_18_00_00_rf'
        output_dir = '/mnt/disks/wrf-mod'
        output_file = 'RAINCELL.DAT'
        res_mins = '60'
        data_hours = '120'

        if 'output_dir' in config:
            output_dir = config['output_dir']
        if 'kelani_basin_points_file' in config:
            kelani_basin_points_file = config['kelani_basin_points_file']
        if 'forecast_db_config' in config:
            forecast_db_config = config['forecast_db_config']
        if 'res_mins' in config:
            res_mins = config['res_mins']
        if 'data_hours' in config:
            data_hours = config['data_hours']

        #output_file.write("%d %d %s %s\n" % (res_mins, data_hours, start_ts, end_ts))

        points = np.genfromtxt(kelani_basin_points_file, delimiter=',')

        kel_lon_min = np.min(points, 0)[1]
        kel_lat_min = np.min(points, 0)[2]
        kel_lon_max = np.max(points, 0)[1]
        kel_lat_max = np.max(points, 0)[2]

        # forecast_adapter = MySQLAdapter(host=forecast_db_config['host'],
        #                                 user=forecast_db_config['user'],
        #                                 password=forecast_db_config['password'],
        #                                 db=forecast_db_config['db'])
        get_forecast_stations_from_net_cdf(reference_net_cdf, kel_lat_min, kel_lon_max, kel_lat_max, kel_lon_max)
        #forecast_stations = get_forecast_stations_from_point_file(kelani_basin_points_file)
        #get_forecast_precipitation(forecast_stations, '2019-03-31 00:00:00', '2019-03-31 12:00:00', forecast_adapter)
        #forecast_adapter.close()
    except Exception as e:
        print('JSON config data loading error.')
        traceback.print_exc()

