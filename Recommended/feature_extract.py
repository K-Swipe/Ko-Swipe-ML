import pandas as pd


class FeatureExtract:
    def __init__(self): ...

    def visit_area_information(visit_area_info):
        """
        방문지 중 유의미한 정보만 추출(1~8까지는 관광지, 나머지는 무의미한 정보) / 지번주소가 없는 데이터 제거
        """
        valid_types = range(1, 9)
        visit_area_info = visit_area_info[visit_area_info["VISIT_AREA_TYPE_CD"].isin(valid_types)]
        visit_area_info = visit_area_info.dropna(subset=["LOTNO_ADDR"])
        visit_area_info = visit_area_info.reset_index(drop=True)
        return visit_area_info

    def address_info(visit_area_info):
        """주소에서 시도와 군구 정보 추출"""
        sido = []
        gungu = []
        for i in range(len(visit_area_info["LOTNO_ADDR"])):
            sido.append(visit_area_info["LOTNO_ADDR"][i].split(" ")[0])
            gungu.append(visit_area_info["LOTNO_ADDR"][i].split(" ")[1])
        visit_area_info["SIDO"] = sido
        visit_area_info["GUNGU"] = gungu
        return visit_area_info

    def mission_check(travel):
        """여행 목적 체크박스 정보 추출"""
        travel_list = []
        for i in range(len(travel)):
            value = int(travel["TRAVEL_MISSION_CHECK"][i].split(";")[0])
            travel_list.append(value)
        return travel_list
