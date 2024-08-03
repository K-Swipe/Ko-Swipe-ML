import pandas as pd
from scipy import stats

value_counts = new_train['VISIT_AREA_NM'].value_counts().reset_index()
value_counts.columns = ['VISIT_AREA_NM', 'count']

def mode_coordinates(group):
    lat_mode = stats.mode(group['X_COORD'])[0]
    lon_mode = stats.mode(group['Y_COORD'])[0]
    return pd.Series([lat_mode, lon_mode], index=['X_COORD', 'Y_COORD'])

coordinates = new_train.groupby('VISIT_AREA_NM').apply(mode_coordinates).reset_index()
coordinates