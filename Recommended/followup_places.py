import pandas as pd
from config.config import cfg


def find_popular_followup_places(df, keyword="해운대", top_n=5):
    """
    Find the most popular places visited by those who visited a specific place.

    Parameters:
    df (DataFrame): The dataframe containing travel data.
    keyword (str): The place name to filter visitors
    top_n (int): Number of top popular places to return (default is 5).
    """
    # TODO: raise an error if keyword is not found in the dataframe 
    
    visitors_to_keyword_place = df[df["VISIT_AREA_NM"].str.contains(keyword)]["TRAVEL_ID"].unique()
    other_visits_by_keyword_visitors = df[df["TRAVEL_ID"].isin(visitors_to_keyword_place)]
    other_visits_excluding_keyword = other_visits_by_keyword_visitors[
        ~other_visits_by_keyword_visitors["VISIT_AREA_NM"].str.contains(keyword)
    ]
    popular_places = other_visits_excluding_keyword["VISIT_AREA_NM"].value_counts().reset_index()
    popular_places.columns = ["VISIT_AREA_NM", "visit_count"]
    return popular_places["VISIT_AREA_NM"][:top_n].tolist()


if __name__ == "__main__":
    df = pd.read_csv(cfg.followup_places_path)
    popular_places_after_haeundae = find_popular_followup_places(df, keyword="해동용궁사", top_n=5)
    print(popular_places_after_haeundae)
