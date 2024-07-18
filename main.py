
import os

import joblib
import pandas as pd

# Define column lists
final_columns = [
    "VISIT_AREA_NM", "SIDO", "GUNGU", "VISIT_AREA_TYPE_CD", "TRAVEL_MISSION_PRIORITY", "GENDER", "AGE_GRP", "INCOME",
    "TRAVEL_STYL_1", "TRAVEL_STYL_2", "TRAVEL_STYL_3", "TRAVEL_STYL_4", "TRAVEL_STYL_5", "TRAVEL_STYL_6", 
    "TRAVEL_STYL_7", "TRAVEL_STYL_8", "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM", 
    "RESIDENCE_TIME_MIN_mean", "RCMDTN_INTENTION_mean", "REVISIT_YN_mean", "TRAVEL_COMPANIONS_NUM_mean", 
    "REVISIT_INTENTION_mean"
]

user_columns = [
    "SIDO", "TRAVEL_MISSION_PRIORITY", "GENDER", "AGE_GRP", "INCOME", "TRAVEL_STYL_1", "TRAVEL_STYL_2",
    "TRAVEL_STYL_3", "TRAVEL_STYL_4", "TRAVEL_STYL_5", "TRAVEL_STYL_6", "TRAVEL_STYL_7", "TRAVEL_STYL_8",
    "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM"
]

features = final_columns

def convert_float_to_int(df):
    """type conversion from float to int"""
    float_cols = df.select_dtypes(include=['float']).columns
    
    for col in float_cols:
        df[col] = df[col].astype(int)
    
    return df

def preprocess_places_list(places_list_str):
    """
    Preprocesses the places list string into a list of places.

    Parameters:
    places_list_str (str): String representation of the places list.

    Returns:
    list: A list of places.
    """
    places_list = places_list_str.replace("[", "").replace("]", "").replace("'", "").replace(", ", ",")
    return list(map(str, places_list.split(",")))

def generate_final_df(info, new_user_info, places_list):
    """
    Generates the final DataFrame based on user information and places list.

    Parameters:
    info (DataFrame): DataFrame containing area information.
    new_user_info (DataFrame): DataFrame containing new user information.
    places_list (list): List of places.

    Returns:
    DataFrame: The final DataFrame containing combined user and area information.
    """
    final_df = pd.DataFrame(columns=final_columns)
    
    for place in places_list:
        sido, gungu = map(str, place.split("+"))
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
    """
    Recommends places based on the model's predictions.

    Parameters:
    model: The predictive model.
    final_df (DataFrame): The final DataFrame containing combined user and area information.

    Returns:
    list: List of recommended places.
    """
    final_df = convert_float_to_int(final_df)
    y_pred = model.predict(final_df)
    y_pred_df = pd.DataFrame(y_pred, columns=["y_pred"])
    sorted_df = pd.concat([final_df, y_pred_df], axis=1).sort_values(by="y_pred", ascending=False).iloc[:10]
    return list(sorted_df["VISIT_AREA_NM"])


def generate_user_info_df(final_df):
    """
    Generates a DataFrame containing user information.

    Parameters:
    final_df (DataFrame): The final DataFrame containing combined user and area information.

    Returns:
    DataFrame: DataFrame containing user information.
    """
    return final_df[user_columns]

def main(info, new_user_info, model):
    """
    Main function to generate recommendations and user information.

    Parameters:
    info (DataFrame): DataFrame containing area information.
    new_user_info (DataFrame): DataFrame containing new user information.
    model: The predictive model.

    Returns:
    list: A list containing user information and recommended places.
    """
    result = []
    places_list_str = new_user_info["sido_gungu_list"].values[0]
    places_list = preprocess_places_list(places_list_str)
    final_df = generate_final_df(info, new_user_info, places_list)
    
    visiting_candidates = recommend_places(model, final_df)
    user_info_df = generate_user_info_df(final_df)
    
    if len(user_info_df) == 0:
        result.append([])
    else:
        rec = user_info_df.iloc[0].to_list()
        rec.append(visiting_candidates)
        result.append(rec)
    
    return result


if __name__ == "__main__":
    PATH = r"C:\workspace\Ko-Swipe-ML\data\all"
    
    info = pd.read_csv(os.path.join(PATH, '관광지 추천시스템 Testset_B- 여행지 정보.csv'))
    recommend_model = joblib.load(os.path.join(PATH, 'base_reco.pkl'))
    test_data = pd.read_pickle("test_data.pkl")
    result = main(info, test_data, recommend_model)
    print(result)
    # [['부산', 22, '대중교통 등', '여', 20, 4, 7, 7, 3, 6, 4, 5, 7, 3, 7, 4, 0, ['부산시립미술관', '일광해수욕장', '벡스코 제2전시장', '청사포 다릿돌 전망대', '스카이라인루지 부산', '신세계 센텀시티몰', '뮤지엄원', '더베이101', '수영만요트경기장', '송정해수욕장']]]
