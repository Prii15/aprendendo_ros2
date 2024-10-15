# Victor Mello Ayres 11.121.224-7
# Pricila Vazquez 11.121.322-9
# Nityananda Saraswati 11.120.414-5

import pygame
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
    def __init__(self, width, height):
        self.window_size = (800, 800)

        # Inicializa o Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Mapa e Dados do LiDAR")
        
        # Inicializa o Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mapa de Ocupação")

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
        
        self.global_map = None  # Inicialização do mapa global
        self.global_map_shape = (450, 450)  # Defina o tamanho do mapa global conforme necessário
        self.xy_resolution = 0.02  # Resolução do grid

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges
        self.distantiae = list(msg.ranges)
        self.angulus = [msg.angle_min + i * msg.angle_increment for i in range(len(self.distantiae))]

    def listener_callback_odom(self, msg):
        self.pose = msg.pose.pose
        
        self.pos_x = self.pose.position.x
        self.pos_y = self.pose.position.y
        
        orientation = self.pose.orientation
        _, _, self.yaw_robot = tf_transformations.euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        
        # Debug: Verifica se a função é chamada e imprime os valores recebidos
        print("Odometry callback acionado.")
        print(f"Posição recebida: x = {self.pos_x}, y = {self.pos_y}")


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

    #desenha mapa
    def draw_map(self):
        for x in range(self.global_map.shape[0]):
            for y in range(self.global_map.shape[1]):
                value = self.global_map[x][y]
                color = (255, 255, 255)  # Cor padrão: branco
                if value == 1.0:  # Ocupado
                    color = (255, 0, 0)  # Vermelho
                elif value == 0.0:  # Livre
                    color = (0, 255, 0)  # Verde
                elif value == 0.5:  # Desconhecido
                    color = (0, 0, 255)  # Azul

                pygame.draw.rect(self.screen, color, (x * self.xy_resolution, y * self.xy_resolution, self.xy_resolution, self.xy_resolution))


    def run(self):
        # Inicializa o mapa
        clock = pygame.time.Clock()
        running = True
        
        while rclpy.ok():
            rclpy.spin_once(self)
            
            # Verificar se os dados do LiDAR estão disponíveis
            if not self.angulus or not self.distantiae:
                self.get_logger().warning('LiDAR data is empty. Skipping update.')
                continue
            
            # Verificar se a pose do robô está disponível
            if self.pose is None:
                self.get_logger().warning('Robot pose is not available. Skipping update.')
                continue
            
            # Cria o mapa do Laser Ray Tracing
            self.ox = np.sin(self.angulus) * self.distantiae
            self.oy = np.cos(self.angulus) * self.distantiae
            
            ox_bot = [self.pos_x + r * cos(self.yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]
            oy_bot = [self.pos_y + r * sin(self.yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]
            
            pmap, minx, maxx, miny, maxy, xyreso = generate_ray_casting_grid_map(ox_bot, oy_bot, self.xyreso, False)
            xyres = np.array(pmap).shape
            
            # Inicializar o mapa global na primeira vez
            if self.global_map is None:
                self.global_map = np.zeros(self.global_map_shape)
            
            # Calcular os índices no mapa global
            min_x_idx = int((minx - self.pos_x) / self.xy_resolution + self.global_map_shape[0] / 2)
            max_x_idx = int((maxx - self.pos_x) / self.xy_resolution + self.global_map_shape[0] / 2)
            min_y_idx = int((miny - self.pos_y) / self.xy_resolution + self.global_map_shape[1] / 2)
            max_y_idx = int((maxy - self.pos_y) / self.xy_resolution + self.global_map_shape[1] / 2)
            
            # Garantir que os índices estejam dentro dos limites do mapa global
            min_x_idx = max(0, min_x_idx)
            max_x_idx = min(self.global_map_shape[0], max_x_idx)
            min_y_idx = max(0, min_y_idx)
            max_y_idx = min(self.global_map_shape[1], max_y_idx)
            
            # Ajustar as dimensões do pmap para corresponder aos índices calculados
            pmap = pmap[:max_x_idx - min_x_idx, :max_y_idx - min_y_idx]
            
            # Atualizar o mapa global
            self.global_map[min_x_idx:max_x_idx, min_y_idx:max_y_idx] = np.maximum(
                self.global_map[min_x_idx:max_x_idx, min_y_idx:max_y_idx], pmap)
            
            # Limpa a tela
            self.screen.fill((255, 255, 255))

            # Desenha os dados do LiDAR
            for angle, distance in zip(self.angulus, self.distantiae):
                x_end = self.pos_x + distance * np.cos(angle)
                y_end = self.pos_y + distance * np.sin(angle)
                pygame.draw.line(self.screen, (255, 0, 0), (self.pos_x, self.pos_y), (x_end, y_end), 1)

            # Desenha o mapa
            for i in range(self.global_map.shape[0]):
                for j in range(self.global_map.shape[1]):
                    if self.global_map[i, j] == 1:  # área ocupada
                        pygame.draw.rect(self.screen, (0, 0, 0), (i, j, 1, 1))

            # Atualiza a tela
            pygame.display.flip()
            clock.tick(60)  # Limita a 60 quadros por segundo

            # Verifica eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()



#função principal
def main(args=None):
    rclpy.init(args=args)
    node = Mapa(800, 800)  # ! cria o mapa aqui. Ajuste o tamanho conforme necessário
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

#chamada da função principal
if __name__ == '__main__':
    main() 
