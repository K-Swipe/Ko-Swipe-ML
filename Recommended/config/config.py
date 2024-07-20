from pathlib import Path

class Config:
    base_path = Path(__file__).parents[2] 
    data_path = base_path / "data" / "all"
    information_path = data_path / "관광지 추천시스템 Testset_B- 여행지 정보.csv"
    
    model_path = data_path / "base_reco.pkl"
    
    # Inference
    final_columns = ["VISIT_AREA_NM", "SIDO", "GUNGU", "VISIT_AREA_TYPE_CD", "TRAVEL_MISSION_PRIORITY", "GENDER", "AGE_GRP", "INCOME",
                     "TRAVEL_STYL_1", "TRAVEL_STYL_2", "TRAVEL_STYL_3", "TRAVEL_STYL_4", "TRAVEL_STYL_5", "TRAVEL_STYL_6", 
                     "TRAVEL_STYL_7", "TRAVEL_STYL_8", "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM", 
                     "RESIDENCE_TIME_MIN_mean", "RCMDTN_INTENTION_mean", "REVISIT_YN_mean", "TRAVEL_COMPANIONS_NUM_mean", "REVISIT_INTENTION_mean"]

    user_columns = ["SIDO", "TRAVEL_MISSION_PRIORITY", "GENDER", "AGE_GRP", "INCOME", "TRAVEL_STYL_1", "TRAVEL_STYL_2",
                    "TRAVEL_STYL_3", "TRAVEL_STYL_4", "TRAVEL_STYL_5", "TRAVEL_STYL_6", "TRAVEL_STYL_7", "TRAVEL_STYL_8",
                    "TRAVEL_MOTIVE_1", "TRAVEL_NUM", "TRAVEL_COMPANIONS_NUM"]

    features = final_columns

    
    


cfg = Config()


if __name__ == "__main__":
    print(cfg.final_columns)
    print(cfg.base_path)
    print(cfg.model_path)