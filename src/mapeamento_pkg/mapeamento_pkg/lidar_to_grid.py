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

import lidar_to_grid_map as lg
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

        pmap, minx, maxx, miny, maxy, xyreso = lg.generate_ray_casting_grid_map(ox, oy, xyreso, False)
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

    def update():
        pass


    def run(self):
        self.get_logger().info ('Iniciando o mapeamento do ambiente.')

        # Main loop
        while rclpy.ok():
            rclpy.spin_once(self)

            self.laser_map()

            #create the fist map

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