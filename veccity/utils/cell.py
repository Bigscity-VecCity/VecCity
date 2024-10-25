import sys
sys.path.append('..')
import math
from veccity.utils import tool_funcs

class CellSpace:

    def __init__(self, lon_unit, lat_unit, \
                lon_min, lat_min, lon_max, lat_max):
        
        assert lon_unit > 0 and lat_unit > 0

        self.lon_unit = lon_unit
        self.lat_unit = lat_unit
        
        # whole space, not for cells
        self.lon_min = lon_min
        self.lat_min = lat_min
        self.lon_max = lon_max
        self.lat_max = lat_max

        self.lon_size = math.ceil((lon_max - lon_min) / lon_unit)
        self.lat_size = math.ceil((lat_max - lat_min) / lat_unit)


    def get_mbr(self, i_lon, i_lat):
        return self.lon_min + self.lon_unit * i_lon, \
                self.lat_min + self.lat_unit * i_lat, \
                self.lon_min + self.lon_unit * i_lon + self.lon_unit, \
                self.lat_min + self.lat_unit * i_lat + self.lat_unit


    def get_cell_id(self, i_lon, i_lat):
        return int(i_lon * self.lat_size + i_lat)
    
    # return (i_lon, i_lat)
    def get_cell_idx(self, cell_id):
        return int(cell_id / self.lat_size), int(cell_id % self.lat_size)


    def get_cellid_range(self):
        return 0, (self.lon_size - 1) * self.lat_size + (self.lat_size - 1)
    

    def get_midpoint_dist(self, cell_id_1, cell_id_2):
        lon_lo_1, lat_lo_1, lon_hi_1, lat_hi_1 = self.get_mbr(*self.get_cell_idx(cell_id_1))
        lon_lo_2, lat_lo_2, lon_hi_2, lat_hi_2 = self.get_mbr(*self.get_cell_idx(cell_id_2))

        lon_mid_1 = (lon_lo_1 + lon_hi_1) / 2
        lat_mid_1 = (lat_lo_1 + lat_hi_1) / 2
        lon_mid_2 = (lon_lo_2 + lon_hi_2) / 2
        lat_mid_2 = (lat_lo_2 + lat_hi_2) / 2

        return tool_funcs.haversine(lon_mid_1, lat_mid_1, lon_mid_2, lat_mid_2)


    def get_cell_id_by_point(self, lon, lat):
        assert self.lon_min <= lon <= self.lon_max \
                and self.lat_min <= lat <= self.lat_max
        
        i_lon = (lon - self.lon_min) // self.lon_unit
        i_lat = (lat - self.lat_min) // self.lat_unit
        return self.get_cell_id(i_lon, i_lat)


    def neighbour_ids(self, i_lon, i_lat):
        # 8 neighbours and self
        lon_r = [i_lon - 1, i_lon, i_lon + 1] 
        lat_r = [i_lat - 1, i_lat, i_lat + 1]
        lons = [l for l in lon_r for _ in range(3)]
        lats = lat_r * 3
        neighbours = zip(lons, lats)
        # remove illegals
        neighbours = filter(lambda cell: cell[0] >= 0 \
                                        and cell[0] < self.lon_size \
                                        and cell[1] >= 0 \
                                        and cell[1] < self.lat_size \
                                        , neighbours)
        return list(neighbours)


    def all_neighbour_cell_pairs_permutated(self):
        # (self, self) are included
        # if (1, 2) in the result, no (2, 1)

        all_cell_pairs = []
        for i_lon in range(self.lon_size):
            for i_lat in range(self.lat_size):
                n_ids = self.neighbour_ids(i_lon, i_lat)
                all_cell_pairs += list(zip([(i_lon, i_lat)] * len(n_ids), n_ids))
        
        all_cell_pairs = list(filter(lambda x: (x[1][1] - x[0][1]) + (x[1][0] - x[0][0]) >= 0 
                                                and not (x[1][1] > x[0][1] and x[1][0] < x[0][0]), \
                                all_cell_pairs))

        return all_cell_pairs