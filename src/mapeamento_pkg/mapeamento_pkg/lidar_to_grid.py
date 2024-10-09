# Victor Mello Ayres 11.121.224-7
# Pricila Vazquez 11.121.322-9
# Nityananda Saraswati 11.120.xxx-x

import math
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, radians, pi

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3

from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations

#import lidar_to_grid_map as lg
from collections import deque

EXTEND_AREA = 1.0

class mapa(Node):


    # Construtor do nó
    def __init__(self):
        super().__init__('mapa')
        self.get_logger().debug ('Definido o nome do nó para "mapa"')

        qos_profile = QoSProfile(depth=10, reliability = QoSReliabilityPolicy.BEST_EFFORT)

        self.get_logger().debug ('Definindo o subscriber do laser: "/scan"')
        self.laser = None
        self.angulus = []
        self.distantiae = []
        self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)

        self.get_logger().debug ('Definindo o subscriber do laser: "/odom"')
        self.pose = None
        self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)

        self.get_logger().debug ('Definindo o publisher de controle do robo: "/cmd_Vel"')
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info ('Definindo buffer, listener e timer para acessar as TFs.')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.on_timer)

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges
        self.distantiae = list(self.laser)
        for i in range(len(self.laser)):
            self.angulus.append(msg.angle_min + i * msg.angle_increment) 

    def listener_callback_odom(self, msg):
        self.pose = msg.pose.pose

        # função de callback do timer
    def on_timer(self):
        try:
            self.tf_right = self.tf_buffer.lookup_transform(
                "right_center_wheel",
                "right_leg_base",
                rclpy.time.Time())

            _, _, self.right_yaw = tf_transformations.euler_from_quaternion(
                [self.tf_right.transform.rotation.x, self.tf_right.transform.rotation.y, 
                 self.tf_right.transform.rotation.z, self.tf_right.transform.rotation.w])

            self.get_logger().info(
                f'yaw right_leg_base to right_center_wheel: {self.right_yaw}')

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform right_leg_base to right_center_wheel: {ex}')

        try:
            self.tf_left = self.tf_buffer.lookup_transform(
                "left_center_wheel",
                "left_leg_base",
                rclpy.time.Time())

            _, _, self.left_yaw = tf_transformations.euler_from_quaternion(
                [self.tf_left.transform.rotation.x, self.tf_left.transform.rotation.y,
                 self.tf_left.transform.rotation.z, self.tf_left.transform.rotation.w])

            self.get_logger().info(
                f'yaw left_leg_base to left_center_wheel: {self.left_yaw}')

        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform left_leg_base to left_center_wheel: {ex}')

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
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        points = np.array(points)
        return points

    @staticmethod
    def calc_grid_map_config(ox, oy, xy_resolution):
        """
        Calculates the size, and the maximum distances according to the the
        measurement center
        """
        min_x = round(min(ox) - EXTEND_AREA / 2.0)
        min_y = round(min(oy) - EXTEND_AREA / 2.0)
        max_x = round(max(ox) + EXTEND_AREA / 2.0)
        max_y = round(max(oy) + EXTEND_AREA / 2.0)
        xw = int(round((max_x - min_x) / xy_resolution))
        yw = int(round((max_y - min_y) / xy_resolution))
        print("The grid map is ", xw, "x", yw, ".")
        return min_x, min_y, max_x, max_y, xw, yw

    @staticmethod
    def atan_zero_to_twopi(y, x):
        angle = math.atan2(y, x)
        if angle < 0.0:
            angle += math.pi * 2.0
        return angle

    @staticmethod
    def init_flood_fill(center_point, obstacle_points, xy_points, min_coord,
                        xy_resolution):
        """
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        xy_points: (x,y) point pairs
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        occupancy_map = (np.ones((xw, yw))) * 0.5
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            free_area = bresenham((prev_ix, prev_iy), (ix, iy))
            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
            prev_ix = ix
            prev_iy = iy
        return occupancy_map

    @staticmethod
    def flood_fill(center_point, occupancy_map):
        """
        center_point: starting point (x,y) of fill
        occupancy_map: occupancy map generated from Bresenham ray-tracing
        """
        # Fill empty areas with queue method
        sx, sy = occupancy_map.shape
        fringe = deque()
        fringe.appendleft(center_point)
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == 0.5:
                    occupancy_map[nx - 1, ny] = 0.0
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == 0.5:
                    occupancy_map[nx + 1, ny] = 0.0
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == 0.5:
                    occupancy_map[nx, ny - 1] = 0.0
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == 0.5:
                    occupancy_map[nx, ny + 1] = 0.0
                    fringe.appendleft((nx, ny + 1))

    @staticmethod
    def generate_ray_casting_grid_map(ox, oy, xy_resolution, breshen=True):
        """
        The breshen boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(ox, oy, xy_resolution)
        # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
        occupancy_map = np.ones((x_w, y_w)) / 2
        center_x = int(round(-min_x / xy_resolution))  # center x coordinate of the grid map
        center_y = int(round(-min_y / xy_resolution))  # center y coordinate of the grid map
        # occupancy grid computed with bresenham ray casting
        if breshen:
            for (x, y) in zip(ox, oy):
                # x coordinate of the the occupied area
                ix = int(round((x - min_x) / xy_resolution))
                # y coordinate of the the occupied area
                iy = int(round((y - min_y) / xy_resolution))
                laser_beams = bresenham((center_x, center_y), (ix, iy))  # line form the lidar to the occupied point
                for laser_beam in laser_beams:
                    occupancy_map[laser_beam[0]][
                        laser_beam[1]] = 0.0  # free area 0.0
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        # occupancy grid computed with with flood fill
        else:
            occupancy_map = init_flood_fill((center_x, center_y), (ox, oy),
                                            (x_w, y_w),
                                            (min_x, min_y), xy_resolution)
            flood_fill((center_x, center_y), occupancy_map)
            occupancy_map = np.array(occupancy_map, dtype=float)
            for (x, y) in zip(ox, oy):
                ix = int(round((x - min_x) / xy_resolution))
                iy = int(round((y - min_y) / xy_resolution))
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution


    def flood_fill(self, cpoint, pmap):
        """
        cpoint: starting point (x,y) of fill
        pmap: occupancy map generated from Bresenham ray-tracing
        """

        # Fill empty areas with queue method
        sx, sy = pmap.shape
        fringe = deque()
        fringe.appendleft(cpoint)
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if pmap[nx - 1, ny] == 0.5:
                    pmap[nx - 1, ny] = 0.0
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if pmap[nx + 1, ny] == 0.5:
                    pmap[nx + 1, ny] = 0.0
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if pmap[nx, ny - 1] == 0.5:
                    pmap[nx, ny - 1] = 0.0
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if pmap[nx, ny + 1] == 0.5:
                    pmap[nx, ny + 1] = 0.0
                    fringe.appendleft((nx, ny + 1))

    def laser_map(self):
        # Convert lidar data to x-y coordinates
        ox = np.sin(self.angulus) * self.distantiae
        oy = np.cos(self.angulus) * self.distantiae

        # Plot the lidar data
        plt.figure(figsize=(6, 10))
        plt.plot([oy, np.zeros(np.size(oy))], [ox, np.zeros(
            np.size(oy))], "ro-")  # lines from 0,0 to the
        plt.axis("equal")
        bottom, top = plt.ylim()  # return the current ylim
        # rescale y axis, to match the grid orientation
        plt.ylim((top, bottom))
        plt.grid(True)
        plt.show()

        xyreso = 0.02  # x-y grid resolution
        yawreso = math.radians(3.1)  # yaw angle resolution [rad]

        pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(ox, oy, xyreso, False)
        xyres = np.array(pmap).shape

        # Plot the laser map
        plt.figure(figsize=(20, 8))
        plt.subplot(122)
        plt.imshow(pmap, cmap="PiYG_r")
        plt.clim(-0.4, 1.4)
        plt.gca().set_xticks(np.arange(-.5, xyres[1], 1), minor=True)
        plt.gca().set_yticks(np.arange(-.5, xyres[0], 1), minor=True)
        plt.grid(True, which="minor", color="w", linewidth=.6, alpha=0.5)
        plt.colorbar()
        plt.show()

    # TODO create the update function 
    def update(self):
        if self.laser is not None and self.pose is not None:
            # Extrair a posição e orientação do robô
            x_robot = self.pose.position.x
            y_robot = self.pose.position.y
            orientation = self.pose.orientation
            _, _, yaw_robot = tf_transformations.euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])

            # Calcular as coordenadas do laser em relação à posição do robô
            ox = [x_robot + r * cos(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]
            oy = [y_robot + r * sin(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]

            # Atualizar o mapa
            pmap, min_x, max_x, min_y, max_y, xy_resolution = self.generate_ray_casting_grid_map(ox, oy, 0.02)

            # Plotar ou processar o pmap aqui conforme necessário
            plt.figure(figsize=(10, 10))
            plt.imshow(pmap, cmap="PiYG_r")
            plt.colorbar()
            plt.title("Mapa Atualizado")
            plt.show()


    def run(self):
        self.get_logger().info ('Iniciando o mapeamento do ambiente.')

        # Main loop
        while rclpy.ok():
            rclpy.spin_once(self)

            self.laser_map()

            # TODO create the fist map

            #update the map
            self.update()

            

def main(args=None):
    rclpy.init(args=args)
    node = mapa()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main() 