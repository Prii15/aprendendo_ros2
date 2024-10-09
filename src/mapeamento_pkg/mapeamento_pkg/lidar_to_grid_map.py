"""
LIDAR to 2D grid map example

author: Erno Horvath, Csaba Hajdu based on Atsushi Sakai's scripts
"""

import math
from collections import deque
import matplotlib.pyplot as plt
import numpy as np


class LidarToGridMap:
    EXTEND_AREA = 1.0

    def __init__(self):
        pass

    @staticmethod
    def bresenham(start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1

        is_steep = abs(dy) > abs(dx)

        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2

        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True

        dx = x2 - x1
        dy = y2 - y1

        error = int(dx / 2.0)
        ystep = 1 if y1 < y2 else -1

        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = (y, x) if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += ystep
                error += dx

        if swapped:
            points.reverse()
        return np.array(points)

    @staticmethod
    def calc_grid_map_config(ox, oy, xyreso):
        minx = round(min(ox) - LidarToGridMap.EXTEND_AREA / 2.0)
        miny = round(min(oy) - LidarToGridMap.EXTEND_AREA / 2.0)
        maxx = round(max(ox) + LidarToGridMap.EXTEND_AREA / 2.0)
        maxy = round(max(oy) + LidarToGridMap.EXTEND_AREA / 2.0)
        xw = int(round((maxx - minx) / xyreso))
        yw = int(round((maxy - miny) / xyreso))

        return minx, miny, maxx, maxy, xw, yw

    @staticmethod
    def init_flood_fill(centerpoint, obstacle_points, minx, maxx, miny, maxy, xyreso):
        w, h = maxx - minx, maxy - miny
        grid = np.ones((w, h)) * 0.5
        for (x, y) in obstacle_points:
            ix = int(round((x - minx) / xyreso))
            iy = int(round((y - miny) / xyreso))
            grid[ix][iy] = 0.0

        center_x = int(round((centerpoint[0] - minx) / xyreso))
        center_y = int(round((centerpoint[1] - miny) / xyreso))

        return grid, center_x, center_y

    @staticmethod
    def flood_fill(grid, x, y):
        stack = deque([(x, y)])
        while stack:
            n = stack.pop()
            if grid[n[0]][n[1]] == 0.5:
                grid[n[0]][n[1]] = 1.0
                if n[0] > 0:
                    stack.append((n[0] - 1, n[1]))
                if n[0] < grid.shape[0] - 1:
                    stack.append((n[0] + 1, n[1]))
                if n[1] > 0:
                    stack.append((n[0], n[1] - 1))
                if n[1] < grid.shape[1] - 1:
                    stack.append((n[0], n[1] + 1))

    @staticmethod
    def generate_ray_casting_grid_map(ox, oy, xyreso, show_animation=False):
        minx, miny, maxx, maxy, xw, yw = LidarToGridMap.calc_grid_map_config(
            ox, oy, xyreso)

        pmap = np.ones((xw, yw)) * 0.5

        for (x, y) in zip(ox, oy):
            ix = int(round((x - minx) / xyreso))
            iy = int(round((y - miny) / xyreso))
            pmap[ix][iy] = 0.0

        for (x, y) in zip(ox, oy):
            ix = int(round((x - minx) / xyreso))
            iy = int(round((y - miny) / xyreso))

            laser_beams = LidarToGridMap.bresenham(
                (int((0.0 - minx) / xyreso), int((0.0 - miny) / xyreso)), (ix, iy))

            for laser_beam in laser_beams:
                pmap[laser_beam[0]][laser_beam[1]] = 1.0

        centerpoint = (0.0, 0.0)
        grid, center_x, center_y = LidarToGridMap.init_flood_fill(
            centerpoint, zip(ox, oy), minx, maxx, miny, maxy, xyreso)
        LidarToGridMap.flood_fill(grid, center_x, center_y)

        for x in range(xw):
            for y in range(yw):
                if grid[x][y] == 1.0:
                    pmap[x][y] = 1.0

        if show_animation:
            plt.imshow(pmap, cmap="PiYG_r")
            plt.pause(1.0)

        return pmap, minx, maxx, miny, maxy, xyreso
