import math
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf_transformations
from .lidar_to_grid_map import *

class Mapa(Node):
    def __init__(self):
        super().__init__('mapa')
        self.get_logger().debug('Definido o nome do nó para "mapa"')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.laser = None
        self.angulus = []
        self.distantiae = []
        self.create_subscription(LaserScan, '/scan', self.listener_callback_laser, qos_profile)

        self.pose = None
        self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.timer = self.create_timer(0.1, self.on_timer)

        self.global_map = np.zeros((600, 1000))  # Inicializa o mapa global

        # Configura a plotagem do mapa
        plt.ion()  # Modo interativo
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.img = self.ax.imshow(self.global_map, cmap="PiYG_r", clim=(-0.4, 1.4))
        plt.colorbar(self.img, ax=self.ax)
        self.ax.set_title("Mapa Atualizado")

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges
        self.distantiae = list(self.laser)
        self.angulus = [msg.angle_min + i * msg.angle_increment for i in range(len(self.laser))]

    def listener_callback_odom(self, msg):
        self.pose = msg.pose.pose

    def on_timer(self):
        # Callback para atualizar TFs (transformações)
        pass

    def update(self):
        if self.laser is not None and self.pose is not None:
            x_robot = self.pose.position.x
            y_robot = self.pose.position.y
            orientation = self.pose.orientation
            _, _, yaw_robot = tf_transformations.euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            )

            ox = [x_robot + r * cos(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]
            oy = [y_robot + r * sin(yaw_robot + angle) for r, angle in zip(self.distantiae, self.angulus)]

            xy_resolution = 0.02
            pmap, _, _, _, _, _ = generate_ray_casting_grid_map(ox, oy, xy_resolution)

            # Atualiza o mapa global
            self.global_map = np.maximum(self.global_map, pmap)

            # Atualiza a visualização do mapa
            self.img.set_array(self.global_map)
            self.fig.canvas.draw_idle()  # Atualiza a figura
            plt.pause(0.01)  # Curta pausa para permitir que o gráfico atualize

    def run(self):
        self.get_logger().info('Iniciando o mapeamento do ambiente.')

        while rclpy.ok():
            rclpy.spin_once(self)  # Processa as mensagens
            self.update()  # Atualiza o mapa

def main(args=None):
    rclpy.init(args=args)
    node = Mapa()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
