#victor Ayres 11.121.224-7
#Priscila Vazquez 11.121.322-9

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

import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin

class R2D2(Node):

    #constructor do nó
    def __init__(self):
        super().__init__('R2D2')
        self.get_logger().debug ('Definido o nome do nó para "R2D2"')

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

        # variables and constants
        self.raio = 0.035
        self.distancia_rodas = 0.267
        self.posicao = [0, 0, 0] # x, y, theta  
        self.medidas = [0, 0] # esq, dir
        self.ultimas_medidas = [0, 0] # esq, dir
        self.distancias = [0, 0]
        
        #encoders
        self.left_yaw = 0
        self.right_yaw = 0

        # mapa
        self.estado_inicial = 0
        self.mapa = [1.5, 4.5, 7.5] # posição central das três “portas” existentes
        self.posicao[0] = self.estado_inicial # atualiza como estado_inicial a posição x de pose 

        # sigma
        self.sigma_odometria = 0.2 # rad
        self.sigma_lidar = 0.175 # meters
        self.sigma_movimento = 0.002 # m

    def gaussian(self, x, mu, sigma):
        return (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    def listener_callback_laser(self, msg):
        self.laser = msg.ranges
        
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

            self.get_logger().info (
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

            self.get_logger().info (
                f'yaw left_leg_base to left_center_wheel: {self.left_yaw}')

        except TransformException as ex:
            self.get_logger().info(
            f'Could not transform left_leg_base to left_center_wheel: {ex}')

    # update function
    def update(self):
        # Pega os valores dos encoders
        self.medidas[0] = self.left_yaw * pi / 180.0  # left_encoder.getValue()
        self.medidas[1] = self.right_yaw * pi / 180.0  # right_encoder.getValue()

        # Calcula a distância percorrida na roda esquerda
        diff = self.medidas[0] - self.ultimas_medidas[0] # Conta quanto a roda LEFT girou desde a última medida (rad)
        self.distancias[0] = diff * self.raio + np.random.normal(0, 0.002) # Determina distância percorrida em metros e adiciona um pequeno erro
        self.ultimas_medidas[0] = self.medidas[0]

        # Calcula a distância percorrida na roda direita
        diff = self.medidas[1] - self.ultimas_medidas[1] # Conta quanto a roda RIGHT girou desde a última medida (rad)
        self.distancias[1] = diff * self.raio + np.random.normal(0, 0.002) # Determina distância percorrida em metros + pequeno erro
        self.ultimas_medidas[1] = self.medidas[1]

        # Cálculo da distância linear e angular percorrida no timestep
        deltaS = (self.distancias[0] + self.distancias[1]) / 2.0
        deltaTheta = (self.distancias[1] - self.distancias[0]) / self.distancia_rodas
        self.posicao[2] = (self.posicao[2] + deltaTheta) % (2 * pi) # Atualiza o valor Theta (diferença da divisão por 2π)

        # Decomposição x e y baseado no ângulo
        deltaSx = deltaS * cos(self.posicao[2])
        deltaSy = deltaS * sin(self.posicao[2])

        # Atualização acumulativa da posição x e y
        self.posicao[0] = self.posicao[0] + deltaSx  # Atualiza x
        self.posicao[1] = self.posicao[1] + deltaSy  # Atualiza y

        # Atualiza a incerteza de movimento
        self.sigma_movimento += 0.002  # Aumenta a incerteza de movimento

        print("Postura:", self.posicao)
        print("distancias:", self.distancia_esquerda, self.distancia_direita)


    # Função principal do nó
    def run(self):
        
        # initial graphics
        # cria um vetor x de 500 valores entre -4.5 e 4.5
        x = np.linspace(-2, 10, 500)
        y = np.zeros(500)
        y2 = np.zeros(500)
        y3 = np.zeros(500)
        fig, ax = plt.subplots()

        controle = 0
        cont = 0
        porta = 0  # robo parte da porta 0

        self.get_logger().debug('Executando uma iteração do loop de processamento de mensagens.')
        rclpy.spin_once(self)

        self.get_logger().debug('Definindo mensagens de controle do robô.')
        self.ir_para_frente = Twist(linear=Vector3( x=0.5, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0))
        self.parar = Twist(linear=Vector3(x=0.0, y=0.0, z=0.0), angular=Vector3(x=0.0, y=0.0, z=0.0))

        self.get_logger().info('Ordenando o robô: "ir para a frente"')
        self.pub_cmd_vel.publish(self.ir_para_frente)
        controle = 1
        rclpy.spin_once(self)

        
        # MAIN LOOP --------------------------------------------------------
        self.get_logger().info('Entrando no loop principal do nó.')
        while rclpy.ok():
            rclpy.spin_once(self)

            self.get_logger().debug('Atualizando as distâncias lidas pelo laser.')
            self.distancia_direita = min(self.laser[0:80])
            self.distancia_frente = min(self.laser[80:100])
            self.distancia_esquerda = min(self.laser[100:180])

            self.get_logger().debug("Distância para o obstáculo: " + str(self.distancia_frente))
            if self.distancia_frente < 1.5:
                self.get_logger().info('Obstáculo detectado.')
                break

            #plotar gaussiana
            if cont % 4 == 0:
                for i in range(len(x)):
                    y[i] = self.gaussian( x[i], self.posicao[0], self.sigma_movimento)

            ax.clear()
            ax.set_ylim([0, 4])
            ax.plot(x, y, color="b")
            plt.pause(0.1)
            self.update()

            if controle == 1:
                # se movimento reto, aumenta a incerteza da posição em 0.002
                self.sigma_movimento += 0.002

            if self.distancia_direita > 1.65 and self.distancia_esquerda > 1.65:
                
                self.pub_cmd_vel.publish(self.parar)
                rclpy.spin_once(self)

                media_nova = (self.mapa[porta] * self.sigma_movimento + self.posicao[0] * self.sigma_lidar) / (self.sigma_movimento + self.sigma_lidar)
                sigma_novo = 1 / (1 / self.sigma_movimento + 1 / self.sigma_lidar)
                
                self.posicao[0] = media_nova
                self.sigma_movimento = sigma_novo

                for i in range(len(x)):
                    y2[i] = self.gaussian(x[i], self.mapa[porta], self.sigma_lidar)
                ax.plot(x, y2, color="r")
                plt.pause(0.1)
                    
                rclpy.spin_once(self, timeout_sec=3.0)

                for i in range(len(x)):
                    y3[i] = self.gaussian(x[i], media_nova, sigma_novo)
                ax.plot(x, y3, color="g")
                plt.pause(0.1)
                
                rclpy.spin_once(self, timeout_sec=3.0)
                
                self.pub_cmd_vel.publish(self.ir_para_frente)
                rclpy.spin_once(self)
                
                if porta == 0: porta = 1
                elif porta == 1: porta = 2
                
                rclpy.spin_once(self, timeout_sec=1.0)

                cont += 1

    # Destrutor do nó
    def __del__(self):
        self.get_logger().info('Finalizando o nó! Tchau, tchau...')


# Função principal
def main(args=None):
    rclpy.init(args=args)
    node = R2D2()
    try:
        node.run()
        node.destroy_node()
        rclpy.shutdown()
    except KeyboardInterrupt:
        pass
   
if __name__ == '__main__':
    main()  

