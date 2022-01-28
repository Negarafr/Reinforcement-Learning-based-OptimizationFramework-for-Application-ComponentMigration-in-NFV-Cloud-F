
import numpy as np
from numpy.random import rand
import pickle as pic
import math
# define a Uniform Distribution
U = lambda MIN, MAX, SAMPLES: rand(*SAMPLES.shape) * (MAX - MIN) + MIN
# define a Truncated Power Law Distribution
P = lambda ALPHA, MIN, MAX, SAMPLES: ((MAX ** (ALPHA+1.) - 1.) * rand(*SAMPLES.shape) + 1.) ** (1./(ALPHA+1.))
# define an Exponential Distribution
E = lambda SCALE, SAMPLES: -SCALE*np.log(rand(*SAMPLES.shape))
#### Initalization ##########
simulation_time =
number_of_nodes = 
v_min =  # mph. Minimum velocity
v_max =   # mph. Maximum velocity
region_x =   # in meter
region_y =   # in meter
nr_nodes=    #number of nodes

# The x-coordinates of all the users.. initialize with zero for all the time instance
nodes_position_x = [[0 for _ in range(simulation_time)]
                    for _ in range(number_of_nodes)]
# The y-coordinates of all the users.. initialize with zero for all the time instance
nodes_position_y = [[0 for _ in range(simulation_time)]
                    for _ in range(number_of_nodes)]

# The z-coordinates of all the users.. initialize with zero for all the time instance
nodes_position_z = [[100 for _ in range(simulation_time)]
                    for _ in range(number_of_nodes)]

velocity_mean = 10.


def gauss_markov(theta,velocity, alpha=0.75, variance=1.):

    angle_mean = theta
    print(' angle_mean = theta', theta)

    alpha2 = 1.0 - alpha
    print('alpha2',alpha2)

    alpha3 = np.sqrt(1.0 - alpha * alpha) * variance
    print('alpha3',alpha3)

    # calculate new speed and direction based on the model
    velocity1 = (alpha * velocity + alpha2 * velocity_mean + alpha3 * np.random.normal(0.0, 1.0, nr_nodes))
    print('new velocity1', velocity1)

    theta1 = (alpha * theta + alpha2 * angle_mean + alpha3 * np.random.normal(0.0, 1.0, nr_nodes))
    print('new theta1', theta1)

    return velocity1,theta1

def mobility(point1,velocity,theta):  # I am putting all the codes in a function.
    x_new = point1[0] + velocity * np.cos(theta)
    y_new = point1[1] + velocity * np.sin(theta)
    point2 = [x_new, y_new]

    return point2
def main(simulation_time):
    MAX_X = region_x
    print('MAX_X', MAX_X)
    MAX_Y = region_y
    print('MAX_Y', MAX_Y)
    NODES = np.arange(nr_nodes)
    velocity = np.zeros(nr_nodes) + velocity_mean
    theta = U(0, 4 * np.pi, NODES)  # meghdare theata vase har node
    angle_mean = theta

    x = U(0, MAX_X, NODES)  # vase har node az 0 ta max-X yek number random generate mikone
    y = U(0, MAX_Y, NODES)

    pointNewX = np.empty((number_of_nodes,simulation_time))
    pointNewY = np.empty((number_of_nodes, simulation_time))

    for node in range(number_of_nodes):
        nodes_position_x[node][0]=x[node] #for the simulation time=0
        nodes_position_y[node][0]=y[node]
        node_current_x = nodes_position_x[node][0]
        node_current_y = nodes_position_y[node][0]
    # The main idea is to run the cod(e for simulation_time and update the position of each node every time.
    for time_instance in range(1, simulation_time):
        velAndthe=gauss_markov(theta, velocity, alpha=0.75, variance=1.)
        velocity=velAndthe[0]
        theta=velAndthe[1]
        for node in range(number_of_nodes):
            node_current_x = nodes_position_x[node][time_instance-1]
            node_current_y = nodes_position_y[node][time_instance-1]
            initial_poisition = [node_current_x, node_current_y]
            new_position = mobility(initial_poisition,velocity[node],theta[node])
            nodes_position_x[node][time_instance] = new_position[0]
            nodes_position_y[node][time_instance] = new_position[1]
    return nodes_position_x,nodes_position_y

x=main(simulation_time)
xposition=x[0]
yposition=x[1]

list_of_tuple = [[0 for _ in range(simulation_time)]
                    for _ in range(number_of_nodes)]

for i in range(number_of_nodes):
    for j in range(simulation_time):
        x=int(xposition[i][j])
        y=int(yposition[i][j])
        z=int(nodes_position_z[i][j])
        list_of_tuple[i][j]=(x,y,z)


outfile = open('Gauss','wb')
pic.dump(list_of_tuple,outfile)
outfile.close()

infile = open('Gauss','rb')
new_list_of_tuple = pic.load(infile)
infile.close()
infile = open('Gauss', 'rb')
Location_Fog = pic.load(infile)
infile.close()

infile = open('Random_way', 'rb')
Location_IoT = pic.load(infile)
infile.close()

x2 = Location_Fog[0][1][1]
print(x2)
def distance(simulationtime,nodeN):
    print('Location_Fog',Location_Fog)
    x2=Location_Fog[nodeN][simulationtime][0]
    y2=Location_Fog[nodeN][simulationtime][1]
    z2=Location_Fog[nodeN][simulationtime][2]
    print('x2:',x2,'y2:',y2,'z2:',z2)
distance(0,1) #simulationtime,nodeN

