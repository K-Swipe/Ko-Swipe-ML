from pathlib import Path

class Config:
    base_path = Path(__file__).parents[2] 
    data_path = base_path / "data" / "preprocessing"
    information_path = data_path / "관광지 추천시스템 Testset_B- 여행지 정보.csv"
    places_path = data_path / "coordinates-final.csv"
    model_path = data_path / "catboost_model_B.pkl"
    followup_places_path = data_path / "followup_places.csv"
    
    # Inference
    # final_columns = ["VISIT_AREA_NM", "SIDO", "GUNGU", "VISIT_AREA_TYPE_CD", "TRAVEL_MISSION_PRIORITY", "MVMN_NM","GENDER",
    #                  "AGE_GRP", "INCOME","TRAVEL_STYL", "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM","RESIDENCE_TIME_MIN_mean", "RCMDTN_INTENTION_mean", "REVISIT_YN_mean", "TRAVEL_COMPANIONS_NUM_mean", "REVISIT_INTENTION_mean"]

    # user_columns = ["SIDO", "TRAVEL_MISSION_PRIORITY", "MVMN_NM", "GENDER", "AGE_GRP", "INCOME", "TRAVEL_STYL",
    #                 "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM"]
    
    final_columns = ["VISIT_AREA_NM", "SIDO", "GUNGU", "VISIT_AREA_TYPE_CD", "TRAVEL_MISSION_PRIORITY", "MVMN_NM","GENDER",
                     "AGE_GRP","TRAVEL_STYL", "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM","RESIDENCE_TIME_MIN_mean", "RCMDTN_INTENTION_mean", "REVISIT_YN_mean", "TRAVEL_COMPANIONS_NUM_mean", "REVISIT_INTENTION_mean"]

    user_columns = ["SIDO", "TRAVEL_MISSION_PRIORITY", "MVMN_NM", "GENDER", "AGE_GRP", "TRAVEL_STYL",
                    "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM"]
    
    features = final_columns

cfg = Config()


if __name__ == "__main__":
    print(cfg.final_columns)
    print(cfg.base_path)
    print(cfg.model_path)