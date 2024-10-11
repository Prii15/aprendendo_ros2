# Victor Mello Ayres 11.121.224-7
# Pricila Vazquez 11.121.322-9
# Nityananda Saraswati 11.120.414-5

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


class Mapa(Node):
    # Construtor do nó
    def __init__(self):
        super().__init__('mapa')
        self.get_logger().debug ('Definido o nome do nó para "mapa"')

        qos_profile = QoSProfile(depth=10, reliability = QoSReliabilityPolicy.BEST_EFFORT)

        self.get_logger().debug ('Definindo o subscriber do laser: "/scan"')
        self.laser = None

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

        self.angulus = []
        self.distantiae = []
        self.ox = []
        self.oy = []

        self.xyreso = 0.02 # x-y grid resolution #? deve ser alterado para o tamanho do robo real?
        self.yawreso = math.radians(3.1) # yaw resolution [rad]

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges
        self.distantiae = list(msg.ranges)
        self.angulus = [msg.angle_min + i * msg.angle_increment for i in range(len(self.distantiae))]

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


    def run(self):
        #Inicializa o mapa
        plt.ion() # modo interativo do matplotlib (mostra o gráfico em tempo real)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))  # Cria subplot

        while rclpy.ok():
            rclpy.spin_once(self)

            #posisao do robo
            x_bot = self.pose.position.x
            y_bot = self.pose.position.y
            orientation = self.pose.orientation
            _, _, yaw_robot = tf_transformations.euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w])
        
            #Cria o mapa do Laser Ray Tracing
            self.ox = np.sin(self.angulus) * self.distantiae
            self.oy = np.cos(self.angulus) * self.distantiae

            ox_bot = [x_bot + r * cos(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]
            oy_bot = [y_bot + r * sin(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]


            pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(self.ox, self.oy, xyreso, False)
            xyres = np.array(pmap).shape


            # Atualiza o gráfico do LiDAR
            ax1.clear()  # Limpa o gráfico anterior
            ax1.plot([self.oy, np.zeros(np.size(self.oy))], [self.ox, np.zeros(np.size(self.oy))], "ro-")  # Plota os dados do LiDAR
            ax1.set_title("Dados do LiDAR")
            ax1.grid(True)

            # Atualizar o gráfico do mapa
            self.ax2.clear()
            self.im = self.ax2.imshow(self.global_map, cmap="PiYG_r", animated=True)
            self.ax2.set_title("Mapa Gerado")
            self.ax2.grid(True, which="minor", color="w", linewidth=0.6, alpha=0.5)
            plt.colorbar(self.im, ax=self.ax2)

            plt.draw()
            plt.pause(0.01)

            plt.pause(0.01)


#função principal
def main(args=None):
    rclpy.init(args=args)
    node = Mapa()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

#chamada da função principal
if __name__ == '__main__':
    main() 
