import pandas as pd

"""2022, 2023 Data concat"""
path2022 = "./data/raw/train_2022"
visit_area_info2022 = pd.read_csv(path2022 + "/tn_visit_area_info_방문지정보_B.csv") 
travel2022 = pd.read_csv(path2022 + "/tn_travel_여행_B.csv")  
traveller_master2022 = pd.read_csv(path2022 + "/tn_traveller_master_여행객 Master_B.csv") 

path2023 = "./data/raw/train_2023"
visit_area_info2023 = pd.read_csv(path2023 + "/tn_visit_area_info_방문지정보_F.csv")  
travel2023 = pd.read_csv(path2023 + "/tn_travel_여행_F.csv")  
traveller_master2023 = pd.read_csv(path2023 + "/tn_traveller_master_여행객 Master_F.csv")

path_val_2022 = "./data/raw/validation_2022"
val_visit_area_info2022 = pd.read_csv(path_val_2022 + "/tn_visit_area_info_방문지정보_B.csv") 
val_travel2022 = pd.read_csv(path_val_2022 + "/tn_travel_여행_B.csv") 
val_traveller_master2022 = pd.read_csv(path_val_2022 + "/tn_traveller_master_여행객 Master_B.csv") 

path_val_2023 = "./data/raw/validation_2023"
val_visit_area_info2023 = pd.read_csv(path_val_2023 + "/tn_visit_area_info_방문지정보_F.csv")  
val_travel2023 = pd.read_csv(path_val_2023 + "/tn_travel_여행_F.csv") 
val_traveller_master2023 = pd.read_csv(path_val_2023 + "/tn_traveller_master_여행객 Master_F.csv")

"""Data concat"""
visi_area_info = pd.concat([visit_area_info2022, visit_area_info2023, val_visit_area_info2022, val_visit_area_info2023], ignore_index=True)
travel = pd.concat([travel2022, travel2023, val_travel2022, val_travel2023], ignore_index=True)
traveller_master = pd.concat([traveller_master2022, traveller_master2023, val_traveller_master2022, val_traveller_master2023], ignore_index=True)

visi_area_info.to_csv("./data/preprocessing/visit_area_info.csv", index=False)
travel.to_csv("./data/preprocessing/travel.csv", index=False)
traveller_master.to_csv("./data/preprocessing/traveller_master.csv", index=False)