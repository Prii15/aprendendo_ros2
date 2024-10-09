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

from .lidar_to_grid_map import * 
from collections import deque


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

        self.global_map = None  # Inicialização do mapa global

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

    # TODO create the update function l
    def update(self):
        if self.laser is not None and self.pose is not None:
            x_robot = self.pose.position.x
            y_robot = self.pose.position.y
            orientation = self.pose.orientation
            _, _, yaw_robot = tf_transformations.euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w])

            # Calcular as coordenadas do laser em relação à posição do robô
            ox = [x_robot + r * cos(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]
            oy = [y_robot + r * sin(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]

            # Atualizar o mapa chamando a função importada
            pmap, min_x, max_x, min_y, max_y, xy_resolution = generate_ray_casting_grid_map(ox, oy, 0.02)

            # Atualizar o mapa global
            if self.global_map is not None:
                self.global_map = np.maximum(self.global_map, pmap)
            else:
                self.global_map = pmap

            # Plotar o mapa global
            plt.figure(figsize=(10, 10))
            plt.imshow(self.global_map, cmap="PiYG_r")
            plt.colorbar()
            plt.title("Mapa Atualizado")
            plt.show()



    def run(self):
        self.get_logger().info ('Iniciando o mapeamento do ambiente.')

        # Main loop
        while rclpy.ok():
            rclpy.spin_once(self)

            # TODO create the fist map

            #update the map
            self.update()   

            if self.global_map is not None:
                plt.figure(figsize=(10, 10))
                plt.imshow(self.global_map, cmap="PiYG_r")
                plt.colorbar()
                plt.title("Mapa Atualizado")
                plt.pause(0.1)  # Pause para permitir que o gráfico atualize 

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