import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from tf_transformations import euler_from_quaternion

class LaserPlotter(Node):
    def __init__(self):
        super().__init__('laser_plotter')
        # Subscrição aos tópicos de LIDAR e odometria
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            'scan',
            self.lidar_callback,
            10
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        self.lidar_subscription
        self.odom_subscription

        # Inicialização de variáveis de odometria
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        self.robot_yaw = 0.0

        # Configurar os gráficos
        self.fig, (self.ax_lidar, self.ax_map) = plt.subplots(1, 2, figsize=(10, 5))
        plt.ion()

        # Limites dos gráficos
        self.ax_lidar.set_xlim([-15, 15])
        self.ax_lidar.set_ylim([-15, 15])
        self.ax_map.set_xlim([-15, 15])
        self.ax_map.set_ylim([-15, 15])

        # Grid nos gráficos
        self.ax_lidar.grid(True)
        self.ax_map.grid(True)

        # Matriz de ocupação (resolução de 0.1m)
        self.map_resolution = 0.1  # Cada célula representa 0.1m
        self.map_size = 2000  # Para representar um mapa de 200x200 metros (20 células)
        self.occupancy_grid = np.zeros((self.map_size, self.map_size), dtype=bool)  # Inicialmente tudo livre

    def odom_callback(self, msg):
        # Extrair a posição e orientação do robô a partir do tópico de odometria
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]

        #Usando apenas o yaw já que é a orientação paralela ao plano que o robô está
        _, _, self.robot_yaw = euler_from_quaternion(orientation_list)

    def lidar_callback(self, msg):
        # Processar os dados do LIDAR
        angles, distances = self.process_lidar_data(msg)

        if angles.size == 0 or distances.size == 0 or angles.size != distances.size:
            self.get_logger().warning('Invalid data received from LIDAR. Angles and distances must match in size.')
            return

        # Calcular as coordenadas do LIDAR no referencial local
        ox = distances * np.cos(angles)
        oy = distances * np.sin(angles)

        # Limpar gráfico de visão do LIDAR (somente)
        self.ax_lidar.clear()
        self.ax_lidar.set_xlim([-15, 15])
        self.ax_lidar.set_ylim([-15, 15])
        self.ax_lidar.grid(True)

        # Plotar as linhas do centro até os pontos do LIDAR
        for x, y in zip(ox, oy):
            self.ax_lidar.plot([0, y], [0, x], "b-", linewidth=0.2)

        # Plotar os pontos do LIDAR em tempo real
        self.ax_lidar.plot(oy, ox, "blue", markersize=1)

        # Calcular as coordenadas globais (mapa) levando em conta a posição e orientação do robô
        global_ox = self.robot_x + ox * np.cos(self.robot_yaw) - oy * np.sin(self.robot_yaw)
        global_oy = self.robot_y + ox * np.sin(self.robot_yaw) + oy * np.cos(self.robot_yaw)

        # Aplicar filtro de limite de -10 a 10 nas coordenadas globais -> Para remover as incertezas do laser
        # Aqui é necessário aplicar para o ambiente de onde está o robô, ou remover e plotar os ruidos mesmo assim
        valid_global_indices = (global_ox >= -10) & (global_ox <= 10) & (global_oy >= -10) & (global_oy <= 10)
        global_ox = global_ox[valid_global_indices]
        global_oy = global_oy[valid_global_indices]

        # Converter as coordenadas globais para índices da matriz de ocupação
        map_indices_x = ((global_ox / self.map_resolution) + (self.map_size / 2)).astype(int)
        map_indices_y = ((global_oy / self.map_resolution) + (self.map_size / 2)).astype(int)

        # Plotar as linhas no mapa somente se o local não estiver ocupado
        for gox, goy, ix, iy in zip(global_ox, global_oy, map_indices_x, map_indices_y):
            if not self.occupancy_grid[ix, iy]:  # Se a célula ainda não estiver ocupada
                self.ax_map.plot([self.robot_y, goy], [self.robot_x, gox], "red", linewidth=0.2)  # Linha do robô até o ponto
                self.occupancy_grid[ix, iy] = True  # Marcar a célula como ocupada

        # Atualizar o gráfico do mapa (sem limpar)
        self.ax_map.plot(global_oy, global_ox, "red", markersize=1)  # Mapa do ambiente

        # Atualizar ambos os gráficos
        plt.draw()
        plt.pause(0.01)

    def process_lidar_data(self, msg):
        """
        Processa os dados do LIDAR convertendo-os em ângulos e distâncias.
        """
        angles = np.arange(msg.angle_min, msg.angle_max + msg.angle_increment, msg.angle_increment)[:len(msg.ranges)]
        distances = np.array(msg.ranges)

        # Remover distâncias inválidas (inf e NaN) e tratá-las como "espaços livres"
        valid_indices = np.isfinite(distances)
        angles = angles[valid_indices]
        distances = distances[valid_indices]

        return angles, distances

def main(args=None):
    rclpy.init(args=args)
    laser_plotter = LaserPlotter()

    try:
        rclpy.spin(laser_plotter)
    except KeyboardInterrupt:
        pass

    laser_plotter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
