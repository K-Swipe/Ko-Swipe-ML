import warnings
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

def load_data(paths):
    busan_spot_info = pd.read_csv(paths['busan_spot_info'], encoding="cp949")
    travel = pd.read_csv(paths['travel'])
    traveller_master = pd.read_csv(paths['traveller_master'])
    return busan_spot_info, travel, traveller_master

def preprocess_travel_data(travel):
    travel["TRAVEL_MISSION_PRIORITY"] = travel["TRAVEL_MISSION_CHECK"].apply(lambda x: int(x.split(";")[0]))
    travel = travel[["TRAVEL_ID", "TRAVELER_ID", "TRAVEL_MISSION_PRIORITY", "MVMN_NM"]]
    exclude_priorities = [8, 10, 25, 27, 28]
    return travel[~travel["TRAVEL_MISSION_PRIORITY"].isin(exclude_priorities)]

def preprocess_traveller_master_data(traveller_master):
    traveller_master["TRAVEL_MOTIVE_1"] = traveller_master["TRAVEL_MOTIVE_1"].replace({4: 1, 6: 2, 8: 7})
    traveller_master = traveller_master[traveller_master["TRAVEL_MOTIVE_1"] != 10]

    travel_style_columns = [
        "TRAVEL_STYL_1", "TRAVEL_STYL_2", "TRAVEL_STYL_3", "TRAVEL_STYL_4",
        "TRAVEL_STYL_5", "TRAVEL_STYL_6", "TRAVEL_STYL_7", "TRAVEL_STYL_8",
    ]
    traveller_master["TRAVEL_STYL"] = traveller_master[travel_style_columns].mode(axis=1)[0]
    traveller_master["TRAVEL_STYL"] = traveller_master["TRAVEL_STYL"].replace({1: 1, 2: 1, 3: 1, 4: 2, 5: 3, 6: 3, 7: 3})

    traveller_master["AGE_GRP"] = traveller_master["AGE_GRP"].apply(lambda x: 40 if x >= 40 else x)
    return traveller_master[
        [
            "TRAVELER_ID", "GENDER", "AGE_GRP", "INCOME",
            "TRAVEL_STYL", "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM"
        ]
    ]

def preprocess_busan_spot_info(busan_spot_info):
    return busan_spot_info[
        [
            "TRAVEL_ID", "VISIT_AREA_NM", "SIDO", "GUNGU", "VISIT_AREA_TYPE_CD",
            "DGSTFN", "REVISIT_INTENTION", "RCMDTN_INTENTION", "RESIDENCE_TIME_MIN", "REVISIT_YN"
        ]
    ].reset_index(drop=True)

def merge_data(travel, traveller_master, busan_spot_info):
    df = pd.merge(travel, traveller_master, on="TRAVELER_ID", how="inner")
    df = pd.merge(busan_spot_info, df, on="TRAVEL_ID", how="left")
    df["RESIDENCE_TIME_MIN"] = df["RESIDENCE_TIME_MIN"].replace(0, df["RESIDENCE_TIME_MIN"].median())
    df["REVISIT_YN"] = df["REVISIT_YN"].replace({"N": 0, "Y": 1})
    df = df.dropna(subset=["TRAVEL_STYL", "TRAVEL_MOTIVE_1"]).reset_index(drop=True)
    return df

def create_train_df(df):
    df_copy = df.copy()
    train_df = pd.DataFrame(columns=df.columns)

    for area in tqdm(df["VISIT_AREA_NM"].unique()):
        area_visitors_df = df_copy[df_copy["VISIT_AREA_NM"] == area]
        if area_visitors_df.empty:
            continue

        random_visitor_df = area_visitors_df.sample(n=1, random_state=42)
        travel_id = random_visitor_df["TRAVEL_ID"].values[0]
        visitor_trips_df = df_copy[df_copy["TRAVEL_ID"] == travel_id]

        df_copy = df_copy[~df_copy["TRAVEL_ID"].isin(visitor_trips_df["TRAVEL_ID"])]
        train_df = pd.concat([train_df, visitor_trips_df], ignore_index=True)

    while len(df_copy) / len(df) > 0.15:
        random_visitor = df_copy.sample(n=1, random_state=42)
        travel_id = random_visitor["TRAVEL_ID"].values[0]
        visitor_trips = df_copy[df_copy["TRAVEL_ID"] == travel_id]

        df_copy = df_copy[~df_copy["TRAVEL_ID"].isin(visitor_trips["TRAVEL_ID"])]
        train_df = pd.concat([train_df, visitor_trips], ignore_index=True)

    return train_df, df_copy

def add_mean_features(train_df):
    mean_features = [
        "RESIDENCE_TIME_MIN", "RCMDTN_INTENTION", "REVISIT_YN", 
        "TRAVEL_COMPANIONS_NUM", "REVISIT_INTENTION"
    ]
    new_train = pd.DataFrame(columns=train_df.columns.tolist() + [f"{col}_mean" for col in mean_features])

    for area in tqdm(train_df["VISIT_AREA_NM"].unique()):
        df2 = train_df[train_df["VISIT_AREA_NM"] == area].copy()
        for col in mean_features:
            mean_val = df2[col].mean()
            df2[f"{col}_mean"] = mean_val
        new_train = pd.concat([new_train, df2], axis=0)

    new_train.sort_values(by=["TRAVEL_ID"], axis=0, inplace=True)
    return new_train

def save_data(new_train, df_copy, path):
    new_train.to_csv(path + "/관광지 추천시스템 Trainset_B.csv", index=False)
    df_copy.to_csv(path + "/관광지 추천시스템 Testset_B.csv", index=False)

def filter_train_data(Train):
    count = Train["VISIT_AREA_NM"].value_counts()
    five_places = count[count >= 5].index.tolist()
    Train = Train[Train["VISIT_AREA_NM"].isin(five_places)].reset_index(drop=True)
    return Train

def drop_unnecessary_features(df):
    drop_columns = [
        "TRAVELER_ID", "REVISIT_INTENTION", "RCMDTN_INTENTION", 
        "RESIDENCE_TIME_MIN", "REVISIT_YN"
    ]
    df.drop(columns=drop_columns, inplace=True)
    return df

def change_dtype(Train, test):
    Train["VISIT_AREA_TYPE_CD"] = Train["VISIT_AREA_TYPE_CD"].astype("string")
    test["VISIT_AREA_TYPE_CD"] = test["VISIT_AREA_TYPE_CD"].astype("string")
    return Train, test

def train_model(x_train, y_train, category_features, path):
    model = CatBoostRegressor(
        n_estimators=1150, max_depth=10, subsample=0.95, colsample_bylevel=0.95,
        cat_features=category_features, random_state=42, verbose=100
    )
    model.fit(x_train, y_train)
    joblib.dump(model, path + "/catboost_model_B.pkl")
    return model

def preprocess_user_info(user_info):
    new_user_info = pd.DataFrame(
        columns=[
            "TRAVEL_ID", "TRAVEL_MISSION_PRIORITY", "MVMN_NM", "GENDER",
            "AGE_GRP", "INCOME", "TRAVEL_STYL", "TRAVEL_MOTIVE_1",
            "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM", "sido_gungu_list"
        ]
    )

    for travel_id in tqdm(user_info["TRAVEL_ID"].unique()):
        user_info_filtered = user_info[user_info["TRAVEL_ID"] == travel_id]
        user_locations = user_info_filtered[["SIDO", "GUNGU"]].apply(lambda x: f"{x['SIDO']}+{x['GUNGU']}", axis=1).unique()
        user_data = user_info_filtered.drop(["SIDO", "GUNGU"], axis=1).iloc[0]
        user_data["sido_gungu_list"] = str(list(user_locations))
        new_user_info = pd.concat([new_user_info, pd.DataFrame([user_data])], axis=0)

    new_user_info.reset_index(drop=True, inplace=True)
    return new_user_info

def preprocess_places_list(places_list_str):
    places_list = places_list_str.strip('[]').replace("'", "").split(", ")
    return places_list

def generate_final_df(info, new_user_info, places_list):
    final_df = pd.DataFrame(columns=final_columns)
    for place in places_list:
        sido, gungu = place.split("+")
        info_df = info[(info["SIDO"] == sido) & (info["GUNGU"] == gungu)].drop(["SIDO"], axis=1).reset_index(drop=True)
        user_data = new_user_info.drop(["sido_gungu_list"], axis=1).values.tolist()[0]
        user_data = [sido] + user_data
        user_df = pd.DataFrame([user_data] * len(info_df), columns=user_columns)
        df = pd.concat([user_df, info_df], axis=1)[features]
        df["VISIT_AREA_TYPE_CD"] = df["VISIT_AREA_TYPE_CD"].astype("string")
        final_df = pd.concat([final_df, df], axis=0)

    final_df.reset_index(drop=True, inplace=True)
    final_df.drop_duplicates(["VISIT_AREA_NM"], inplace=True)
    return final_df

def recommend_places(model, final_df):
    final_df = final_df.copy()
    final_df[final_df.select_dtypes(include=['float']).columns] = final_df.select_dtypes(include=['float']).astype(int)
    y_pred = model.predict(final_df)
    y_pred_df = pd.DataFrame(y_pred, columns=["y_pred"])
    sorted_df = pd.concat([final_df, y_pred_df], axis=1).sort_values(by="y_pred", ascending=False).iloc[:20]
    return sorted_df["VISIT_AREA_NM"].tolist()

def main(info, new_user_info, model):
    result = []
    places_list_str = new_user_info["sido_gungu_list"].values[0]
    places_list = preprocess_places_list(places_list_str)
    final_df = generate_final_df(info, new_user_info, places_list)
    visiting_candidates = recommend_places(model, final_df)

    if final_df.empty:
        result.append([])
    else:
        rec = final_df.iloc[0][user_columns].tolist()
        rec.append(visiting_candidates)
        result.append(rec)

    return result

if __name__ == "__main__":
    paths = {
        'busan_spot_info': "./data/preprocessing/busan_spot_info - final.csv",
        'travel': "./data/preprocessing/travel.csv",
        'traveller_master': "./data/preprocessing/traveller_master.csv",
        'path_to_save': r"C:\workspace\Ko-Swipe-ML\data\preprocessing"
    }
    
    busan_spot_info, travel, traveller_master = load_data(paths)
    travel = preprocess_travel_data(travel)
    traveller_master = preprocess_traveller_master_data(traveller_master)
    busan_spot_info = preprocess_busan_spot_info(busan_spot_info)
    
    df = merge_data(travel, traveller_master, busan_spot_info)
    train_df, df_copy = create_train_df(df)
    
    new_train = add_mean_features(train_df)
    save_data(new_train, df_copy, paths['path_to_save'])
    
    Train = pd.read_csv(paths['path_to_save'] + "/관광지 추천시스템 Trainset_B.csv")
    test = pd.read_csv(paths['path_to_save'] + "/관광지 추천시스템 Testset_B.csv")
    
    Train = filter_train_data(Train)
    Train = drop_unnecessary_features(Train)
    test = drop_unnecessary_features(test)
    
    Train, test = change_dtype(Train, test)
    
    x_train = Train.drop(["DGSTFN", "TRAVEL_ID"], axis=1)
    y_train = Train["DGSTFN"]
    
    category_features = [
        "VISIT_AREA_NM", "SIDO", "GUNGU", "VISIT_AREA_TYPE_CD", 
        "TRAVEL_MISSION_PRIORITY", "AGE_GRP", "GENDER", "MVMN_NM"
    ]
    
    model = train_model(x_train, y_train, category_features, paths['path_to_save'])
    
    user_info = pd.read_csv(paths['path_to_save'] + "/관광지 추천시스템 Testset_B- 유저 정보.csv")
    info = pd.read_csv(paths['path_to_save'] + "/관광지 추천시스템 Testset_B- 여행지 정보.csv")
    
    result = main(info, user_info, model)
    print(result)
