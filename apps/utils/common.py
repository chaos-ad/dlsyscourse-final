import yaml
import logging
import logging.config
import dateutil
import datetime
import numpy as np
import pandas as pd
import tabulate

#############################################################################

def setup_logging(config_file='/app/conf/logging.yml'):
    with open(config_file, 'rt') as file:
        try:
            config = yaml.safe_load(file.read())
            logging.config.dictConfig(config)
        except Exception as e:
            print(e)
            print('Error in Logging Configuration. Using default configs')
            logging.basicConfig(level=logging.INFO)

#############################################################################

logger = logging.getLogger(__name__)

#############################################################################

def parse_date(date):
    return dateutil.parser.parse(date).date() if isinstance(date, str) else date

def today():
    return parse_date(datetime.datetime.now().date())

def days_ago(days, date=today()):
    return parse_date(date) - datetime.timedelta(days=days)

def days_after(days, date=today()):
    return parse_date(date) + datetime.timedelta(days=days)

def next_day(date=today()):
    return days_after(1, date=date)

def prev_day(date=today()):
    return days_ago(1, date=date)

def tomorrow():
    return next_day()

def yesterday():
    return prev_day()

def date_range(date_from, date_to):
    date_from = parse_date(date_from)
    date_to = parse_date(date_to)
    while date_from < date_to:
        yield date_from
        date_from = next_day(date_from)

def safe_cast_to_int(sequence, check=True):
    intermediate = pd.to_numeric(sequence.replace({np.nan: None, '': None}), downcast="integer")
    if check:
        assert np.all(np.modf(intermediate.dropna().to_numpy())[0] == 0)
    return intermediate.round().astype('Int64')

def tabulate_formats():
    return tabulate.tabulate_formats + ['none']

def tabulate_df(df, format='psql', index=True):
    if format and format.lower() != "none":
        if format not in tabulate_formats():
            raise Exception(f"Invalid argument {format=}")
        print(tabulate.tabulate(df, headers=df.columns, tablefmt=format, showindex=index))
    return df

#############################################################################

