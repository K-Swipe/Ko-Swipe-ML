import pandas as pd

places = pd.read_csv(r"C:\workspace\Ko-Swipe-ML\data\preprocessing\followup_places.csv")
places = places['VISIT_AREA_NM']

places = places.unique()
print(places)