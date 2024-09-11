"""가장 많이 방문한 지역을 추출하는데 필요한 데이터를 만드는 코드입니다."""
import pandas as pd


train = pd.read_csv(r"C:\workspace\Ko-Swipe-ML\data\preprocessing\관광지 추천시스템 Trainset.csv")
test = pd.read_csv(r"C:\workspace\Ko-Swipe-ML\data\preprocessing\관광지 추천시스템 Testset.csv")

train = train[["TRAVEL_ID", "VISIT_AREA_NM"]]
test = test[["TRAVEL_ID", "VISIT_AREA_NM"]]

df = pd.concat([train, test]).reset_index(drop=True)

if __name__ == "__main__":
    df.to_csv(r"C:\workspace\Ko-Swipe-ML\data\preprocessing\followup_places.csv", index=False)
