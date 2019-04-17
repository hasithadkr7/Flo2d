import getopt
import sys
import traceback
from datetime import datetime
import json


def usage():
    usage_text = """
Usage: ./gen_rain.py [-d YYYY-MM-DD] [-t HH:MM:SS] [-h]

-h  --help          Show usage
-d  --run_date          Date in YYYY-MM-DD. Default is current date.
-t  --run_time          Time in HH:MM:SS. Default is current time.
    --ts-start-date    Start date of timeseries which need to run the forecast in YYYY-MM-DD format. Default is same as -d(date).
    --ts-start-time    Start time of timeseries which need to run the forecast in HH:MM:SS format. Default is same as -t(date).
"""
    print(usage_text)


if __name__ == "__main__":
    try:
        config = json.loads(open('/home/hasitha/PycharmProjects/Flo2d/configs/config.json').read())
        output_dir = '/home/hasitha/PycharmProjects/Flo2d/output'
        output_file = 'RAIN.DAT'
        run_date = datetime.now().strftime("%Y-%m-%d")
        run_time = datetime.now().strftime("%H:00:00")
        backward = '2'
        forward = '3'
        ts_start_date = ''
        ts_start_time = ''
        ts_end_date = ''
        ts_end_time = ''

        if 'output_dir' in config:
            output_dir = config['output_dir']
        if 'rain_file' in config:
            rain_file = config['rain_file']

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hd:t:f:b:", [
                "help", "run_date=", "run_time=", "forward=", "backward=", "ts-start-date=", "ts-start-time=", "ts-end-date=", "ts-end-time="
            ])
        except getopt.GetoptError:
            usage()
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit(0)
            elif opt in ("-d", "--run_date"):
                run_date = arg
            elif opt in ("-t", "--run_time"):
                run_time = arg
            elif opt in ("-f", "--forward"):
                forward = arg
            elif opt in ("-b", "--backward"):
                backward = arg
            elif opt in ("--ts-start-date"):
                ts_start_date = arg
            elif opt in ("--ts-start-time"):
                ts_start_time = arg
            elif opt in ("--ts-end-date"):
                ts_end_date = arg
            elif opt in ("--ts-end-time"):
                ts_end_time = arg
        print('[run_date, run_time, forward, backward, ts_start_date, ts_start_time, ts_end_date, ts_end_time] : ',
              [run_date, run_time, forward, backward, ts_start_date, ts_start_time, ts_end_date, ts_end_time])
        ts_start_date_time = datetime.strptime('%s %s' % (ts_start_date, ts_start_time), '%Y-%m-%d %H:%M:%S')
        ts_end_date_time = datetime.strptime('%s %s' % (ts_end_date, ts_end_time), '%Y-%m-%d %H:%M:%S')
        print('ts_start_date_time : ', ts_start_date_time)
        print('ts_end_date_time : ', ts_end_date_time)
    except Exception as e:
        print('JSON config data loading error.')
        traceback.print_exc()

