import json
import traceback
from tempfile import TemporaryDirectory
import numpy as np
import os
from curw.rainfall.wrf.extraction.observation_utils import CurwObservationException
from mpl_toolkits.basemap import Basemap, cm
from shapely.geometry import Polygon, Point
from scipy.spatial import Voronoi
from netCDF4 import Dataset
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
from curwmysqladapter import MySQLAdapter
from decimal import Decimal
from curw.rainfall.wrf.extraction import spatial_utils
import csv


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


def get_observed_precip(obs_stations, start_dt, end_dt, duration_days, adapter, forecast_source='wrf0'):
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
        # print('station : ', s)
        row_ts = adapter.retrieve_timeseries(station, opts)
        if len(row_ts) == 0:
            print('No data for {} station from {} to {} .'.format(s, start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')))
        else:
            ts = np.array(row_ts[0]['timeseries'])
            #print('ts length:', len(ts))
            if len(ts) != 0 :
                ts_df = pd.DataFrame(data=ts, columns=['ts', 'precip'], index=ts[0:])
                ts_sum = ts_df.groupby(by=[ts_df.ts.map(lambda x: x.strftime('%Y-%m-%d %H:00'))]).sum()
                _validate_ts(s, ts_sum, opts)

        obs[s] = ts_sum
    print('get_observed_precip|success')
    return obs


def get_forecast_precipitation(forecast_stations, start_dt, end_dt, adapter, forecast_source='wrf_v3_A', ):
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
        #print('station : ', s)
        row_ts = adapter.retrieve_timeseries(station, opts)
        if len(row_ts) == 0:
            print('No data for {} station from {} to {} .'.format(s, start_dt, end_dt))
        else:
            ts = np.array(row_ts[0]['timeseries'])
            #print('ts length:', len(ts))
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
    init_lats = nc_fid.variables['XLAT'][:][0]
    lats = []
    for lat_row in init_lats:
        lats.append(lat_row[0])
    lons = nc_fid.variables['XLONG'][:][0][0]

    lon_min_idx = np.argmax(lons >= min_lon) -1
    lat_min_idx = np.argmax(lats >= min_lat) -1
    lon_max_idx = np.argmax(lons >= max_lon)
    lat_max_idx = np.argmax(lats >= max_lat)

    # lat_inds = np.where((lats >= min_lat) & (lats <= max_lat))
    # lon_inds = np.where((lons >= min_lon) & (lons <= max_lon))
    #
    # lon_min_idx = np.argmin(lon_inds)
    # lat_min_idx = np.argmin(lat_inds)
    # lon_max_idx = np.argmax(lon_inds)
    # lat_max_idx = np.argmax(lat_inds)

    lats = lats[lat_min_idx:lat_max_idx]
    lons = lons[lon_min_idx:lon_max_idx]

    print('get_forecast_stations_from_net_cdf : ', [lats[0], lats[-1], lons[0], lons[-1]])

    width = len(lons)
    height = len(lats)

    stations = []
    station_points = {}
    csv_file_name = '/home/hasitha/PycharmProjects/Flo2d/output/wrf_points.csv'
    line1 = ['wrf_point','latitude','longitude']
    count = 1
    with open(csv_file_name, 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(line1)
        for y in range(height):
            for x in range(width):
                lat = lats[y]
                lon = lons[x]
                station_name = '%s_%.6f_%.6f' % ('wrf_v3_A', lon, lat)
                stations.append(station_name)
                station_points[station_name] = [lon, lat, 'WRF', station_name]
                index = 'wrf_point%s' % (count)
                lat_val = '%.6f' % (lat)
                lon_val = '%.6f' % (lon)
                writer.writerow([index, lat_val, lon_val])
                count = count+1
    csvFile.close()
    return stations, station_points


def get_two_element_average(prcp, return_diff=True):
    avg_prcp = (prcp[1:] + prcp[:-1]) * 0.5
    if return_diff:
        return avg_prcp - np.insert(avg_prcp[:-1], 0, [0], axis=0)
    else:
        return avg_prcp


def is_inside_geo_df(geo_df, lon, lat, polygon_attr='geometry', return_attr='id'):
    point = Point(lon, lat)
    for i, poly in enumerate(geo_df[polygon_attr]):
        if point.within(poly):
            return geo_df[return_attr][i]
    return None


def _voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of inputs vertices, with 'points at infinity' appended to the
        end.

    from: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D inputs")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def get_voronoi_polygons(points_dict, shape_file, shape_attribute=None, output_shape_file=None, add_total_area=True):
    """
    :param points_dict: dict of points {'id' --> [lon, lat]}
    :param shape_file: shape file path of the area
    :param shape_attribute: attribute list of the interested region [key, value]
    :param output_shape_file: if not none, a shape file will be created with the output
    :param add_total_area: if true, total area shape will also be added to output
    :return:
    geo_dataframe with voronoi polygons with columns ['id', 'lon', 'lat','area', 'geometry'] with last row being the area of the
    shape file
    """
    if shape_attribute is None:
        shape_attribute = ['OBJECTID', 1]

    shape_df = gpd.GeoDataFrame.from_file(shape_file)
    shape_polygon_idx = shape_df.index[shape_df[shape_attribute[0]] == shape_attribute[1]][0]
    shape_polygon = shape_df['geometry'][shape_polygon_idx]

    ids = [p if type(p) == str else np.asscalar(p) for p in points_dict.keys()]
    points = np.array(list(points_dict.values()))[:, :2]

    vor = Voronoi(points)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    data = []
    for i, region in enumerate(regions):
        polygon = Polygon([tuple(x) for x in vertices[region]])
        if polygon.intersects(shape_polygon):
            intersection = polygon.intersection(shape_polygon)
            data.append({'id': ids[i], 'lon': vor.points[i][0], 'lat': vor.points[i][1], 'area': intersection.area,
                         'geometry': intersection
                         })
    if add_total_area:
        data.append({'id': '__total_area__', 'lon': shape_polygon.centroid.x, 'lat': shape_polygon.centroid.y,
                     'area': shape_polygon.area, 'geometry': shape_polygon})

    df = gpd.GeoDataFrame(data, columns=['id', 'lon', 'lat', 'area', 'geometry'], crs=shape_df.crs)

    if output_shape_file is not None:
        df.to_file(output_shape_file)

    return df


if __name__ == "__main__":
    try:
        config = json.loads(open('/home/hasitha/PycharmProjects/Flo2d/configs/config.json').read())
        kelani_basin_points_file = '/home/hasitha/PycharmProjects/Flo2d/configs/kelani_basin_points_250m.txt'
        reference_net_cdf = '/home/hasitha/PycharmProjects/Flo2d/configs/wrf_wrfout_d03_2019-03-31_18_00_00_rf'
        # reference_net_cdf = '/home/hasitha/PycharmProjects/Flo2d/configs/wrf_wrfout_d03_2018-01-01_18_00_00_rf'
        output_dir = '/mnt/disks/wrf-mod'
        output_file = 'RAINCELL.DAT'
        res_mins = '60'
        data_hours = '120'
        run_date = datetime.now().strftime("%Y-%m-%d")
        run_time = datetime.now().strftime("%H:00:00")
        tag = ''
        backward = '2'
        forward = '3'

        if 'output_dir' in config:
            output_dir = config['output_dir']
        if 'kelani_basin_points_file' in config:
            kelani_basin_points_file = config['kelani_basin_points_file']
        if 'kelani_lower_basin_shp_file' in config:
            kelani_lower_basin_shp_file = config['kelani_lower_basin_shp_file']
        if 'forecast_db_config' in config:
            forecast_db_config = config['forecast_db_config']
        if 'observed_db_config' in config:
            observed_db_config = config['observed_db_config']
        if 'obs_stations' in config:
            obs_stations = config['obs_stations']
        if 'res_mins' in config:
            res_mins = config['res_mins']
        if 'forward' in config:
            forward = config['forward']
        if 'backward' in config:
            backward = config['backward']
        if 'run_date' in config:
            run_date = config['run_date']
        if 'run_time' in config:
            run_time = config['run_time']
        if 'rain_cell_file' in config:
            rain_cell_file = config['rain_cell_file']

        print('[run_date, run_time] : ', [run_date, run_time])
        start_ts_lk = datetime.strptime('%s %s' % (run_date, run_time), '%Y-%m-%d %H:%M:%S')
        start_ts_lk = start_ts_lk.strftime('%Y-%m-%d_%H:00')  # '2018-05-24_08:00'
        duration_days = (int(backward), int(forward))
        obs_start = datetime.strptime(start_ts_lk, '%Y-%m-%d_%H:%M') - timedelta(days=duration_days[0])
        obs_end = datetime.strptime(start_ts_lk, '%Y-%m-%d_%H:%M')
        forecast_end = datetime.strptime(start_ts_lk, '%Y-%m-%d_%H:%M') + timedelta(days=duration_days[1])
        print([obs_start, obs_end, forecast_end])

        obs_start = '2019-03-18 00:00:00'
        obs_end = '2019-03-20 00:00:00'
        forecast_end = '2019-03-22 23:00:00'

        points = np.genfromtxt(kelani_basin_points_file, delimiter=',')

        kel_lon_min = np.min(points, 0)[1]
        kel_lat_min = np.min(points, 0)[2]
        kel_lon_max = np.max(points, 0)[1]
        kel_lat_max = np.max(points, 0)[2]

        print('[kel_lon_min, kel_lat_min, kel_lon_max, kel_lat_max] : ', [kel_lon_min, kel_lat_min, kel_lon_max, kel_lat_max])

        forecast_adapter = MySQLAdapter(host=forecast_db_config['host'],
                                        user=forecast_db_config['user'],
                                        password=forecast_db_config['password'],
                                        db=forecast_db_config['db'])
        # #min_lat, min_lon, max_lat, max_lon
        forecast_stations, station_points = get_forecast_stations_from_net_cdf(reference_net_cdf,
                                                                               kel_lat_min,
                                                                               kel_lon_min,
                                                                               kel_lat_max,
                                                                               kel_lon_max)
        print('forecast_stations length : ', len(forecast_stations))

        forecast_precipitations = get_forecast_precipitation(forecast_stations, obs_end, forecast_end, forecast_adapter)
        # print('forecast_precipitations : ', forecast_precipitations)
        forecast_adapter.close()

        observed_adapter = MySQLAdapter(host=observed_db_config['host'],
                                        user=observed_db_config['user'],
                                        password=observed_db_config['password'],
                                        db=observed_db_config['db'])

        # observed_precipitations = get_observed_precip(obs_stations, obs_start, obs_end, duration_days, observed_adapter, forecast_source='wrf0')
        observed_precipitations = get_observed_precip(obs_stations, datetime.strptime(obs_start, '%Y-%m-%d %H:%M:%S'),
                                                     datetime.strptime(obs_end, '%Y-%m-%d %H:%M:%S'), duration_days,
                                                     observed_adapter, forecast_source='wrf0')
        print('observed_precipitations : ', observed_precipitations)
        observed_adapter.close()

        thess_poly = get_voronoi_polygons(obs_stations, kelani_lower_basin_shp_file, add_total_area=False)

        fcst_thess_poly = get_voronoi_polygons(station_points, kelani_lower_basin_shp_file, add_total_area=False)
        #print(fcst_thess_poly)
        fcst_point_thess_idx = []
        for point in points:
            fcst_point_thess_idx.append(is_inside_geo_df(fcst_thess_poly, lon=point[1], lat=point[2]))
            pass
        print('fcst_point_thess_idx : ', fcst_point_thess_idx)

        output_dir = os.path.join(output_dir, run_date + '_' + run_time)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            output_file_path = os.path.join(output_dir, 'RAINCELL.DAT')

            # update points array with the thessian polygon idx
            point_thess_idx = []
            for point in points:
                point_thess_idx.append(is_inside_geo_df(thess_poly, lon=point[1], lat=point[2]))
                pass
            print('point_thess_idx : ', point_thess_idx)
            with open(output_file_path, 'w') as output_file:
                res_mins = int(res_mins)
                data_hours = int(sum(duration_days) * 24 * 60 / res_mins)
                output_file.write("%d %d %s %s\n" % (res_mins, data_hours, obs_start, forecast_end))

                print('range 1 : ', int(24 * 60 * duration_days[0] / res_mins) + 1)
                print('range 1 : ', int(24 * 60 * duration_days[1] / res_mins) - 1)

                for t in range(int(24 * 60 * duration_days[0] / res_mins) + 1):
                    for i, point in enumerate(points):
                        rf = float(observed_precipitations[point_thess_idx[i]].values[t]) if point_thess_idx[i] is not None else 0
                        output_file.write('%d %.1f\n' % (point[0], rf))

                for t in range(int(24 * 60 * duration_days[1] / res_mins) - 1):
                    for point in points:
                        rf = float(forecast_precipitations[fcst_point_thess_idx[i]].values[t]) if fcst_point_thess_idx[
                                                                                                 i] is not None else 0
                        output_file.write('%d %.1f\n' % (point[0], rf))

    except Exception as e:
        print('JSON config data loading error.')
        traceback.print_exc()

