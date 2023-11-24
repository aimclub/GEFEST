import json

# import numpy as np
# from shapely.geometry import shape


def parse_arctic_geojson(result_path, border_path, root_path):
    """Function for parsing data for breakwaters case.

    :param result_path: path to result data
    :param border_path:  path to border info data
    :param root_path:  root path
    :return:
    """
    with open(
        f"{root_path}/{result_path}", "r"
    ) as file:
        res_list = json.load(file)

    water = [i for i in res_list["features"] if i["properties"]["type"] == "water"]
    water_coord = [p["geometry"]["coordinates"] for p in water]
    allow_water = [
        i
        for i in water_coord[0][0]
        if (i[0] > 74.8) and (i[1] < 67.942) and (i[1] > 67.915)
    ]

    return allow_water
