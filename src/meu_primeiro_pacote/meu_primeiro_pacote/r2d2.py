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
from numpy import random
import math

class R2D2(Node):

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
        self.raio = 0.033
        self.distancia_rodas = 0.178
        self.pose = [0, 0, 0] # x, y, theta
        self.medidas = [0, 0] # esq, dir
        self.ultimas_medidas = [0, 0] # esq, dir
        self.distancias = [0, 0]

        # mapa
        self.estado_inicial = -4
        self.mapa = [-2.7, -0.7, 2.7] # posição central das três “portas” existentes
        self.pose[0] = estado_inicial # atualiza como estado_inicial a posição x de pose

        # sigma
        self.sigma_odometria = 0.2 # rad
        self.sigma_lidar = 0.175 # meters
        self.sigma_movimento = 0.002 # m

    def gaussian(x, mean, sigma):
        return (1 / (sigma*sqrt(2*pi))) * exp(-((x-mean)**2) / (2*sigma**2))

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges
       
    def listener_callback_odom(self, msg):
        self.pose = msg.pose.pose

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
        self.medidas[0] = self.left_yaw       #left_encoder.getValue()
        self.medidas[1] = self.right_yaw      #right_encoder.getValue()
        diff = self.medidas[0] - self.ultimas_medidas[0] # conta quanto a roda LEFT girou desde a última medida (rad)
        self.distancias[0] = diff * self.raio + random.normal(0,0.002) # determina distância percorrida em metros e adiciona umpequeno erro
        self.ultimas_medidas[0] = self.medidas[0]
        diff = self.medidas[1] - self.ultimas_medidas[1] # conta quanto a roda LEFT girou desde a última medida (rad)
        self.distancias[1] = diff * self.raio + random.normal(0,0.002) # determina distância percorrida em metros + pequeno erro
        self.ultimas_medidas[1] = self.medidas[1]

        # ## cálculo da dist linear e angular percorrida no timestep
        deltaS = (self.distancias[0] + self.distancias[1]) / 2.0
        deltaTheta = (self.distancias[1] - self.distancias[0]) / self.distancia_rodas
        self.pose[2] = (self.pose[2] + deltaTheta) % 6.28 # atualiza o valor Theta (diferença da divisão por 2π)

        # decomposição x e y baseado no ângulo
        deltaSx = deltaS * cos(self.pose[2])
        deltaSy = deltaS * sin(self.pose[2])

        # atualização acumulativa da posição x e y
        self.pose[0] = self.pose[0] + deltaSx # atualiza x
        self.pose[1] = self.pose[1] + deltaSy # atualiza y

        print("Postura:", self.pose)

    def loop(self):
        # initial graph # em roxo tudo que for relativo ao gráfico de plotagem
        x = np.linspace(-4.5, 4.5, 500) # cria um vetor x de 500 valores entre -4.5 e 4.5
        y = np.zeros(500) # cria um vetor y de 500 valores zeros
        y2 = np.zeros(500)
        y3 = np.zeros(500)
        fig, ax = plt.subplots()

        # main loop ------------------------------------------------------------------------------------------
        controle = 0
        cont = 0
        porta = 0 # robô começa em frente antes da porta 0

        while rclpy.ok():
            rclpy.spin_once(self)
            self.update()
            cont += 1

            #PLOTAR GAUSSIANA DO ROBÔ
            if cont % 4 == 0: # a cada 4 passos, plotar em preto “b” a gaussiana da posição do robô em x (pose[0])
                for i in range(len(x)):
                    y[i] = self.gaussian(x[i], self.pose[0], self.sigma_movimento)

            ax.clear()
            ax.set_ylim([0, 4])
            ax.plot(x, y, color="b")
            plt.pause(0.1)
            leitura = self.laser
            
            if controle == 1:
                self.sigma_movimento = self.sigma_movimento + 0.002 # se movimento reto, aumenta a incerteza da posição em 0.002
                
            if leitura[72] == inf and leitura[108] == inf: # se a leitura indicar em frente a uma porta
                left.setVelocity(0)
                right.setVelocity(0)

                media_nova = (mapa[porta]*sigma_movimento + pose[0]*sigma_lidar) / (sigma_movimento+sigma_lidar)
                sigma_novo = 1 / (1/sigma_movimento + 1/sigma_lidar)
                pose[0] = media_nova # a nova posição x do robô
                sigma_movimento = sigma_novo # novo erro gaussiano do robô

                for i in range(len(x)): y2[i] = gaussian(x[i], mapa[porta], sigma_lidar)
                    ax.plot(x, y2, color="r")
                    plt.pause(0.1) # plota em vermelho “r” a gaussiana da leitura do laser com relação à porta
                    robot.step(3000)

                    for i in range(len(x)): y3[i] = gaussian(x[i], media_nova, sigma_novo)
                        ax.plot(x, y3, color="g")
                        plt.pause(0.1) # plota em verde “g” a gaussiana nova após interpolação das duas gaussianas.
                        robot.step(3000)
                        left.setVelocity(4)
                        right.setVelocity(4)

                if porta == 0: porta = 1 # altera para a próxima porta 0 → 1 ; 1 → 2
                elif porta == 1: porta = 2

                robot.step(1000)

                cont += 1

    def run(self):
         
        self.get_logger().debug ('Executando uma iteração do loop de processamento de mensagens.')
        rclpy.spin_once(self)

        self.get_logger().debug ('Definindo mensagens de controde do robô.')
        self.ir_para_frente = Twist(linear=Vector3(x= 0.5,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= 0.0))
        self.parar          = Twist(linear=Vector3(x= 0.0,y=0.0,z=0.0),angular=Vector3(x=0.0,y=0.0,z= 0.0))

        self.get_logger().info ('Ordenando o robô: "ir para a frente"')
        self.pub_cmd_vel.publish(self.ir_para_frente)
        rclpy.spin_once(self)

        self.get_logger().info ('Entrando no loop princial do nó.')
        while(rclpy.ok):
            rclpy.spin_once(self)

            self.get_logger().debug ('Atualizando as distancias lidas pelo laser.')
            self.distancia_direita   = min((self.laser[  0: 80])) # -90 a -10 graus
            self.distancia_frente    = min((self.laser[ 80:100])) # -10 a  10 graus
            self.distancia_esquerda  = min((self.laser[100:180])) #  10 a  90 graus

            self.get_logger().debug ("Distância para o obstáculo" + str(self.distancia_frente))
            if(self.distancia_frente < 1.5):
                self.get_logger().info ('Obstáculo detectado.')
                break

        self.get_logger().info ('Ordenando o robô: "parar"')
        self.pub_cmd_vel.publish(self.parar)
        rclpy.spin_once(self)


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

