
#? BIBLIOTECAS 
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


#? Classe principal (MAPA)
class Mapa(Node):
    #? Construtor do nó
    def __init__(self):
        super().__init__('mapa')
        
        self.get_logger().debug('Definido o nome do nó para "mapa"')
        
        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        
        #? Ajustando a assinatura para usar create_subscription diretamente
        self.get_logger().debug('Definindo o subscriber do laser: "/scan"')
        self.laser = None
        self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)
        
        #? Ajustando a assinatura para usar create_subscription diretamente
        self.get_logger().debug('Definindo o subscriber do odometry: "/odom"')
        self.pose = None
        self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)

        self.get_logger().debug('Definindo o publisher de controle do robô: "/cmd_vel"')
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info('Definindo buffer, listener e timer para acessar as TFs.')
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.on_timer)
        
        #? Variaveis globais 
        self.angulus = []       # lista com os valorem dos angulos dos feixes do lidar.
        self.distantiae = []    # lista com as distancia obtidas de cada feixe.
        
        #? Definir as resoluções do mapa e o tamanho
        self.map_resolution = 0.1                   # Tamanho da célula (10 cm por célula)
        self.map_size_x, self.map_size_y = 20, 20   # Numero de celulas 

        #? Inicializar o mapa de ocupação: 
        self.occupancy_grid = np.full((210, 210), 0.5)  # 0.5 = desconhecido, 0 = livre, 1 = ocupado

        #? Criando os graficos para plotagem
        plt.ion()  # modo interativo do matplotlib
        self.fig, (self.ax_lidar, self.ax_map) = plt.subplots(1, 2, figsize=(10, 5))

    #? Processar a mensagem do laser_scan (LIDAR)
    def listener_callback_laser(self, msg):
        #// time_stamp = msg.header.stamp
        #// self.get_logger().info(f'lidar Timestemp: {time_stamp}')
        
        self.laser = msg.ranges
        self.distantiae = list(msg.ranges)
        self.angulus = [msg.angle_min + i * msg.angle_increment for i in range(len(self.distantiae))]  # Retorna em radianos -1.57 a 1.57

    #? Processar a mensagem do odom (ODOMETRIA)
    def listener_callback_odom(self, msg):
        #// time_stamp = msg.header.stamp
        #// self.get_logger().info(f'odom Timestemp: {time_stamp}')
        
        self.pose = msg.pose.pose

        self.pos_x = self.pose.position.x       # retorna a posicao X
        self.pos_y = self.pose.position.y       # retorna a posicao Y
        
        # retorna a orientacao do robo (YAW, PITCH, ROLL)
        orientation = [self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w]
        _, _, self.yaw_robot = tf_transformations.euler_from_quaternion(orientation)
        
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
    
    #? Função para converter coordenadas reais em índices de grade
    def coord_to_grid(self, x, y):
        # O centro do mapa é a célula no meio da matriz (105, 105).
        grid_x = int((x / self.map_resolution) + self.occupancy_grid.shape[0] // 2)
        grid_y = int((y / self.map_resolution) + self.occupancy_grid.shape[1] // 2)
        return grid_x, grid_y
    
    #? Função para atulizar o mapa de ocupancia com base nos coordenadas dos pontos de obstaculo
    def update_occupancy_grid(self, occupancy_grid, pos_x, pos_y, ox_global, oy_global):

        robot_x_grid, robot_y_grid = self.coord_to_grid(pos_x, pos_y) # Converta a posição do robô para índices de grade
        
        for x_obstacle, y_obstacle in zip(ox_global, oy_global):

            #? Aplicar filtro de limite de -10 a 10 nas coordenadas globais
            if -10 <= x_obstacle <= 10 and -10 <= y_obstacle <= 10:
                
                obs_x_grid, obs_y_grid = self.coord_to_grid(x_obstacle, y_obstacle) # Converta a posição do obstáculo para índices de grade
                
                #? Verifique se os índices estão dentro dos limites da grade
                if 0 <= obs_x_grid < occupancy_grid.shape[1] and 0 <= obs_y_grid < occupancy_grid.shape[0]:
                    
                    points = bresenham((robot_x_grid, robot_y_grid), (obs_x_grid, obs_y_grid)) # Use o algoritmo de Bresenham para traçar uma linha do robô até o obstáculo
                    
                    #? Marcar as células no caminho como livres (valor 0)
                    for point in points[:-1]:  # Não incluir o último ponto (que será o obstáculo)
                        if 0 <= point[0] < occupancy_grid.shape[1] and 0 <= point[1] < occupancy_grid.shape[0]:
                            occupancy_grid[point[1], point[0]] = 0
                    
                    #? Marcar o último ponto (obstáculo) como ocupado (valor 1)
                    occupancy_grid[points[-1][1], points[-1][0]] = 1

    def run(self):

        while rclpy.ok():
            rclpy.spin_once(self)

            #? Verificar se os dados do LiDAR e da pose estão disponíveis
            if not self.angulus or not self.distantiae:
                self.get_logger().warning('LiDAR data is empty. Skipping update.')
                continue

            if self.pose is None:
                self.get_logger().warning('Robot pose is not available. Skipping update.')
                continue

            #? CÁLCULO DAS COORDENADAS
            # coordenadas locais
            x_local = np.array(self.distantiae) * np.cos(self.angulus)
            y_local = np.array(self.distantiae) * np.sin(self.angulus)

            # coordenadas globais
            ox_global = self.pos_x + (x_local * np.cos(self.yaw_robot) - y_local * np.sin(self.yaw_robot))      
            oy_global = self.pos_y + (x_local * np.sin(self.yaw_robot) + y_local * np.cos(self.yaw_robot))
            
            # Atuliza o mapa de ocupancia
            self.update_occupancy_grid(self.occupancy_grid, self.pos_x, self.pos_y, ox_global, oy_global)

            #? PLOTAGEM DOS MAPAS
            # "visao" do lidar
            self.ax_lidar.clear()
            self.ax_lidar.plot([y_local, np.zeros(np.size(y_local))], [x_local, np.zeros(np.size(y_local))], "ro-")
            self.ax_lidar.set_title("Dados do LiDAR")
            self.ax_lidar.grid(True)
            
            #mapa de ocupancia
            self.ax_map.clear()
            self.ax_map.imshow(self.occupancy_grid, cmap="PiYG_r", origin="lower",
                       extent=[-self.map_size_x * self.map_resolution / 2, self.map_size_x * self.map_resolution / 2,
                               -self.map_size_y * self.map_resolution / 2, self.map_size_y * self.map_resolution / 2])
            self.ax_map.grid(True)
            self.ax_map.set_title("Mapa de Ocupação")
            
            plt.draw()
            plt.pause(0.01)


#? Função principal
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
