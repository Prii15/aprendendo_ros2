# Victor Mello Ayres 11.121.224-7
# Pricila Vazquez 11.121.322-9
# Nityananda Saraswati 11.120.414-5

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

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

        self.get_logger().debug('Definido o nome do nó para "mapa"')
        
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        self.get_logger().debug('Definindo o subscriber do laser: "/scan"')
        self.laser = None
        
        # Ajustando a assinatura para usar create_subscription diretamente
        self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)
        
        self.get_logger().debug('Definindo o subscriber do odometry: "/odom"')
        self.pose = None
        
        # Ajustando a assinatura para usar create_subscription diretamente
        self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)

        self.get_logger().debug('Definindo o publisher de controle do robô: "/cmd_vel"')
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('Definindo buffer, listener e timer para acessar as TFs.')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.on_timer)
        
        self.angulus = []
        self.distantiae = []
        
        # Definir as resoluções do mapa e o tamanho
        self.map_resolution = 0.1  # Tamanho da célula (10 cm por célula)
        self.map_size_x, self.map_size_y = 20, 20

        # Inicializar o mapa de ocupação: 0.5 = desconhecido, 0 = livre, 1 = ocupado
        self.occupancy_grid = np.full((300, 300), 0.5)

    def listener_callback_laser(self, msg):
        # time_stamp = msg.header.stamp
        # self.get_logger().info(f'lidar Timestemp: {time_stamp}')
        
        self.laser = msg.ranges
        self.distantiae = list(msg.ranges)
        self.angulus = [msg.angle_min + i * msg.angle_increment for i in range(len(self.distantiae))]  # Retorna em radianos -1.57 a 1.57

    def listener_callback_odom(self, msg):
        # time_stamp = msg.header.stamp
        # self.get_logger().info(f'odom Timestemp: {time_stamp}')
        
        self.pose = msg.pose.pose
        self.pos_x = self.pose.position.x
        self.pos_y = self.pose.position.y
        
        orientation = self.pose.orientation
        _, _, self.yaw_robot = tf_transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        
        # Debug: Verifica se a função é chamada e imprime os valores recebidos
        print("Odometry callback acionado.")
        print(f"Posição recebida: x = {self.pos_x}, y = {self.pos_y}")
        print(f"Orientação recebida: yaw = {self.yaw_robot}")

    # Função de callback do timer
    def on_timer(self):
        try:
            self.tf_right = self.tf_buffer.lookup_transform(
                "right_center_wheel",
                "right_leg_base",
                rclpy.time.Time())
            
            _, _, self.right_yaw = tf_transformations.euler_from_quaternion(
                [self.tf_right.transform.rotation.x, self.tf_right.transform.rotation.y, 
                self.tf_right.transform.rotation.z, self.tf_right.transform.rotation.w])
            
            self.get_logger().info(f'yaw right_leg_base to right_center_wheel: {self.right_yaw}')
            
        except TransformException as ex:
            self.get_logger().info(f'Could not transform right_leg_base to right_center_wheel: {ex}')
            
        try:
            self.tf_left = self.tf_buffer.lookup_transform(
                "left_center_wheel",
                "left_leg_base",
                rclpy.time.Time())
            
            _, _, self.left_yaw = tf_transformations.euler_from_quaternion(
                [self.tf_left.transform.rotation.x, self.tf_left.transform.rotation.y,
                self.tf_left.transform.rotation.z, self.tf_left.transform.rotation.w])
            
            self.get_logger().info(f'yaw left_leg_base to left_center_wheel: {self.left_yaw}')
            
        except TransformException as ex:
            self.get_logger().info(f'Could not transform left_leg_base to left_center_wheel: {ex}')

    # Função para converter coordenadas reais em índices de grade
    def coord_to_grid(self, x, y, map_size_x, map_size_y, map_resolution):
        grid_x = int((x + (map_size_x / 2)) / map_resolution)
        grid_y = int((y + (map_size_y / 2)) / map_resolution)
        return grid_x, grid_y

    # Função para atualizar o mapa de ocupação
    def update_occupancy_grid(self, occupancy_grid, pos_x, pos_y, ox_global, oy_global):
        # Converta a posição do robô para índices de grade
        robot_x_grid, robot_y_grid = self.coord_to_grid(pos_x, pos_y, self.map_size_x, self.map_size_y, self.map_resolution)
        
        for x_obstacle, y_obstacle in zip(ox_global, oy_global):
            # Converta a posição do obstáculo para índices de grade
            obs_x_grid, obs_y_grid = self.coord_to_grid(x_obstacle, y_obstacle, self.map_size_x, self.map_size_y, self.map_resolution)
            
            # Use o algoritmo de Bresenham para traçar uma linha do robô até o obstáculo
            points = bresenham((robot_x_grid, robot_y_grid), (obs_x_grid, obs_y_grid))
            
            # Marcar as células no caminho como livres (valor 0)
            for point in points[:-1]:  # Não incluir o último ponto (que será o obstáculo)
                occupancy_grid[point[1], point[0]] = 0
            
            # Marcar o último ponto (obstáculo) como ocupado (valor 1)
            occupancy_grid[points[-1][1], points[-1][0]] = 1

    def run(self):
        plt.ion()  # modo interativo do matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 12))

        while rclpy.ok():
            rclpy.spin_once(self)

            if not self.angulus or not self.distantiae:
                self.get_logger().warning('LiDAR data is empty. Skipping update.')
                continue

            if self.pose is None:
                self.get_logger().warning('Robot pose is not available. Skipping update.')
                continue

            # coordenadas locais
            x_local = [ox * np.cos(oy) for ox, oy in zip(self.distantiae, self.angulus)]
            y_local = [ox * np.sin(oy) for ox, oy in zip(self.distantiae, self.angulus)]

            # coordenadas globais
            ox_global = [np.sin(ang) * dist + self.pos_x for ang, dist in zip(self.angulus, self.distantiae)]
            oy_global = [np.cos(ang) * dist + self.pos_y for ang, dist in zip(self.angulus, self.distantiae)]
            
            self.update_occupancy_grid(self.occupancy_grid, self.pos_x, self.pos_y, ox_global, oy_global)

            ax1.clear()
            ax1.plot([y_local, np.zeros(np.size(y_local))], [x_local, np.zeros(np.size(y_local))], "ro-")
            ax1.set_title("Dados do LiDAR")
            ax1.grid(True)

            ax2.clear()
            ax2.imshow(self.occupancy_grid, cmap="PiYG_r", origin="lower",
                       extent=[-self.map_size_x * self.map_resolution / 2, self.map_size_x * self.map_resolution / 2,
                               -self.map_size_y * self.map_resolution / 2, self.map_size_y * self.map_resolution / 2])
            ax2.set_title("Mapa de Ocupação")
            ax2.grid(True)

            plt.draw()
            plt.pause(0.01)


# Função principal
def main(args=None):
    rclpy.init(args=args)
    node = Mapa()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass


# Chamada da função principal
if __name__ == '__main__':
    main()
