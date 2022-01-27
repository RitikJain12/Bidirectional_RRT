from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from numpy.core.fromnumeric import shape
import math
import random


#For animation purpose 
class World: 

    def __init__(self,n):
        self.N=n
        self.radius = 0.52
        self.height_z = 1.22

    def points_for_cylinder(self,center_x,center_y,center_z): # create mesh grid for a cylinder
        z = np.linspace(center_z-self.height_z/2,center_z + self.height_z/2, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = self.radius*np.cos(theta_grid) + center_x
        y_grid = self.radius*np.sin(theta_grid) + center_y
        return x_grid,y_grid,z_grid

    def rotate_cylinder(self,X,Y,Z,roll,pitch,yaw,xc,yc,zc): # rotate the cylinder using roll pitch and yaw
        x_temp = np.array(X)
        y_temp = np.array(Y)

        for i in range(x_temp.shape[0]):
            for j in range(x_temp.shape[1]):
                X[i][j]= (math.cos(yaw)*(x_temp[i][j]-xc)-math.sin(yaw)*(y_temp[i][j]-yc))+xc
                Y[i][j]= (math.sin(yaw)*(x_temp[i][j]-xc)+math.cos(yaw)*(y_temp[i][j]-yc))+yc 

        x_temp = np.array(X)
        z_temp = np.array(Z)

        for i in range(x_temp.shape[0]):
            for j in range(x_temp.shape[1]):
                X[i][j]= (math.cos(pitch)*(x_temp[i][j]-xc)+math.sin(pitch)*(z_temp[i][j]-zc))+xc
                Z[i][j]= (-(math.sin(pitch))*(x_temp[i][j]-xc)+math.cos(pitch)*(z_temp[i][j]-zc))+zc

        y_temp = np.array(Y)
        z_temp = np.array(Z)

        for i in range(y_temp.shape[0]):
            for j in range(y_temp.shape[1]):
                Y[i][j]= (math.cos(roll)*(y_temp[i][j]-yc)-math.sin(roll)*(z_temp[i][j]-zc))+yc
                Z[i][j]= (math.sin(roll)*(y_temp[i][j]-yc)+math.cos(roll)*(z_temp[i][j]-zc))+zc
        return X,Y,Z

    def points_for_plane(self,upperx,lowerx,axis1,upper1,lower1,axis_const,const): # create mesh grid for the vending machine
        x= np.linspace(lowerx,upperx,self.N)
        temp = np.linspace(lower1,upper1,self.N)
        if axis1 == 'y':
            X,Y = np.meshgrid(x,temp)
        else:
            X,Z = np.meshgrid(x,temp)
        if axis_const == 'y':
            Y = np.full_like(X,const)
        else:
            Z = np.full_like(X,const)
        
        return X,Y,Z

    def world(self,ax,path,animate): #creating the final world and animating the can
        
        if animate:
            ax.view_init(0, 5)
            X,Y,Z = self.points_for_cylinder(path[0],path[1],path[2])
            X_,Y_,Z_ = self.rotate_cylinder(X,Y,Z,path[3],path[4],path[5],path[0],path[1],path[2])
            ax.plot_surface(X_, Y_, Z_, alpha=1)
        else:
            self.Xc,self.Yc,self.Zc = self.points_for_cylinder(5,2,0.52)
            self.Xr,self.Yr,self.Zr = self.rotate_cylinder(self.Xc,self.Yc,self.Zc,0,1.57,0,5,2,0.52)
            
            ax.plot_surface(self.Xr, self.Yr, self.Zr, alpha=1)

            self.Xc,self.Yc,self.Zc = self.points_for_cylinder(5,17.5,0.6)
            ax.plot_surface(self.Xc, self.Yc, self.Zc, alpha=1)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'y',10,0,'z',10)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'z',10,0,'y',0)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'y',10,0,'z',0)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'z',10,9.5,'y',10)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'z',6.5,0,'y',10)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(3.5,0,'z',10,0,'y',10)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,6.5,'z',10,0,'y',10)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'y',12.5,10,'z',10)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'y',12.5,10,'z',0)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'z',1.5,0,'y',12.5)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,0,'z',10,4.5,'y',12.5)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(3.5,0,'z',10,0,'y',12.5)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

        self.X,self.Y,self.Z = self.points_for_plane(10,6.5,'z',10,0,'y',12.5)
        ax.plot_surface(self.X,self.Y,self.Z,alpha=0.5)

# to store the RRT tree
class Tree:
    def __init__(self,points):
        super().__init__()
        self.x = points[0]
        self.y = points[1]
        self.z = points[2]
        self.roll = points[3]
        self.pitch = points[4]
        self.yaw = points[5]
        self.E = []
        self.E_dist = []
    
    def add_edge(self,node):
        self.E.append(node)
        dist = math.dist([self.x,self.y,self.z],[node.x,node.y,node.z])
        self.E_dist.append(dist)
    
    def get_neighbour(self):
        return self.E,self.E_dist

# Rapidly exploring random trees
class RRT:
    def __init__(self) :
        self.allowed_dist = 1
        self.xmin = 0 
        self.xmax = 10
        self.ymin = 0
        self.ymax = 17.5
        self.zmin = 0 
        self.zmax = 10

    def generate_random_point(self,vertex): #get random points 
        while True:
            self.points = []
            self.points.append(random.uniform(self.xmin,self.xmax)) #x
            self.points.append(random.uniform(self.ymin,self.ymax)) #y
            self.points.append(random.uniform(self.zmin,self.zmax)) #z
            if self.points not in vertex: # to avoid geting same points
                self.points.append(random.uniform(0,math.pi)) #roll
                self.points.append(random.uniform(0,math.pi)) #pitch
                self.points.append(random.uniform(0,math.pi)) #yaw
                return self.points
    
    def check_nearest(self,node,points): # find the nearest point in the tree for the random point 
        points = points[0:3]
        self.min = math.inf
        self.visited = []
        self.queue =[]
        self.queue.append(node)
        while len(self.queue)!=0:
            self.n = self.queue.pop(0)
            self.visited.append([self.n.x,self.n.y,self.n.z]) 
            self.dist = math.dist([self.n.x,self.n.y,self.n.z],points)
            if self.dist<self.min and ((points[0]!=self.n.x) or (points[1]!=self.n.y) or (points[2]!=self.n.z)):
                self.min = self.dist
                self.short = self.n
                self.magnitude = 1/((points[0]-self.n.x)**2+(points[1]-self.n.y)**2+(points[2]-self.n.z)**2)
                self.unit_vector = [(points[0]-self.n.x)*self.magnitude,(points[1]-self.n.y)*self.magnitude,(points[2]-self.n.z)*self.magnitude]
            for i in range(len(self.n.E)):
                if not [self.n.E[i].x,self.n.E[i].y,self.n.E[i].z] in self.visited:
                    self.queue.append(self.n.E[i])
        return self.min,self.short,self.unit_vector

    def new_config(self,node,unit_vector,points): # generate a new point based on the direction of the new point
        self.points = []
        self.points.append(node.x+(self.allowed_dist*unit_vector[0]))
        self.points.append(node.y+(self.allowed_dist*unit_vector[1]))
        self.points.append(node.z+(self.allowed_dist*unit_vector[2]))
        self.new_allowed_dist = self.allowed_dist
        if self.points[0]>self.xmax:
            self.points = []
            self.new_allowed_dist = (self.xmax-node.x)/unit_vector[0]
            self.points.append(node.x+(self.new_allowed_dist*unit_vector[0]))
            self.points.append(node.y+(self.new_allowed_dist*unit_vector[1]))
            self.points.append(node.z+(self.new_allowed_dist*unit_vector[2]))
        if self.points[0]<self.xmin:
            self.points = []
            self.new_allowed_dist = (self.xmin-node.x)/unit_vector[0]
            self.points.append(node.x+(self.new_allowed_dist*unit_vector[0]))
            self.points.append(node.y+(self.new_allowed_dist*unit_vector[1]))
            self.points.append(node.z+(self.new_allowed_dist*unit_vector[2]))
        if self.points[1]>self.ymax:
            self.points = []
            self.new_allowed_dist = (self.ymax-node.y)/unit_vector[1]
            self.points.append(node.x+(self.new_allowed_dist*unit_vector[0]))
            self.points.append(node.y+(self.new_allowed_dist*unit_vector[1]))
            self.points.append(node.z+(self.new_allowed_dist*unit_vector[2]))
        if self.points[1]<self.ymin:
            self.points = []
            self.new_allowed_dist = (self.ymin-node.y)/unit_vector[1]
            self.points.append(node.x+(self.new_allowed_dist*unit_vector[0]))
            self.points.append(node.y+(self.new_allowed_dist*unit_vector[1]))
            self.points.append(node.z+(self.new_allowed_dist*unit_vector[2]))
        if self.points[2]>self.zmax:
            self.points = []
            self.new_allowed_dist = (self.zmax-node.z)/unit_vector[2]
            self.points.append(node.x+(self.new_allowed_dist*unit_vector[0]))
            self.points.append(node.y+(self.new_allowed_dist*unit_vector[1]))
            self.points.append(node.z+(self.new_allowed_dist*unit_vector[2]))
        if self.points[2]<self.zmin:
            self.points = []
            self.new_allowed_dist = (self.zmin-node.z)/unit_vector[2]
            self.points.append(node.x+(self.new_allowed_dist*unit_vector[0]))
            self.points.append(node.y+(self.new_allowed_dist*unit_vector[1]))
            self.points.append(node.z+(self.new_allowed_dist*unit_vector[2]))
        self.points.append(points[3])
        self.points.append(points[4])
        self.points.append(points[5])
        self.dist = self.new_allowed_dist
        return self.points,self.dist

    def valid_point(self,xs,ys,zs,unit_vector,dist): # check if the point and line joining the point lie in the free space 
        self.w = World(10)
        self.dist = 0
        while self.dist<=dist:
            # getting each point on the line
            self.x = xs + self.dist*unit_vector[0]
            self.y = ys + self.dist*unit_vector[1]
            self.z = zs + self.dist*unit_vector[2]
            self.dist+=self.w.height_z/4
            # dividing the region into 5 blocks and defining the free space for it
            if self.y>=0 and self.y<=(10-self.w.height_z/2):
                if self.x<(self.w.height_z/2) or self.x>(10-self.w.height_z/2) or self.z<(self.w.height_z/2) or self.z>(10-self.w.height_z/2):
                    return False
            elif self.y<=(10+self.w.height_z/2):
                if self.x<(3.5+self.w.height_z/2) or self.x>(6.5-self.w.height_z/2) or self.z<(6.5+self.w.height_z/2) or self.z>(9.5-self.w.height_z/2):
                    return False
            elif self.y<=(12.5-self.w.height_z/2):
                if self.x<(self.w.height_z/2) or self.x>(10-self.w.height_z/2)  or self.z<(self.w.height_z/2) or self.z>(10-self.w.height_z/2):
                    return False
            elif self.y<=(12.5+self.w.height_z/2):
                if self.x<(3.5+self.w.height_z/2) or self.x>(6.5-self.w.height_z/2)  or self.z<(1.5+self.w.height_z/2) or self.z>(4.5-self.w.height_z/2):
                    return False

        return True

    def rrt_connect(self,start_point,goal_point): #bidirectional RRT
        self.tree_s = Tree(start_point)
        self.tree_g = Tree(goal_point)
        self.current_s = self.tree_s
        self.current_g = self.tree_g
        self.vertex_s = []
        self.vertex_g = []
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        w = World(10)
        while 1:
            self.points = self.generate_random_point(self.vertex_s)
            _,self.node,self.unit_vector = self.check_nearest(self.tree_s,self.points)
            self.points,self.dist = self.new_config(self.node,self.unit_vector,self.points)
            if self.valid_point(self.node.x,self.node.y,self.node.z,self.unit_vector,self.dist):
                self.vertex_s.append(self.points)
                self.current_s = Tree(self.points)
                self.node.add_edge(self.current_s)
                # plt.pause(0.01)
                ax.plot([self.node.x,self.current_s.x],[self.node.y,self.current_s.y],zs=[self.node.z,self.current_s.z],c='k')

                        
                self.dist,self.node,self.unit_vector = self.check_nearest(self.tree_g,self.points)
                if self.dist<self.allowed_dist:
                    if self.valid_point(self.node.x,self.node.y,self.node.z,self.unit_vector,self.dist):
                        self.current_s.add_edge(self.node)
                        ax.plot([self.node.x,self.current_s.x],[self.node.y,self.current_s.y],zs=[self.node.z,self.current_s.z],c='k')

                        break

            self.points = self.generate_random_point(self.vertex_g)
            _,self.node,self.unit_vector = self.check_nearest(self.tree_g,self.points)
            self.points,self.dist = self.new_config(self.node,self.unit_vector,self.points)
            if self.valid_point(self.node.x,self.node.y,self.node.z,self.unit_vector,self.dist):
                self.vertex_g.append(self.points)
                self.current_g = Tree(self.points)
                self.node.add_edge(self.current_g)
                self.current_g.add_edge(self.node)
                # plt.pause(0.01)
                ax.plot([self.node.x,self.current_g.x],[self.node.y,self.current_g.y],zs=[self.node.z,self.current_g.z],c='r')

                
                self.dist,self.node,self.unit_vector = self.check_nearest(self.tree_s,self.points)
                if self.dist<=self.allowed_dist:
                    if self.valid_point(self.current_g.x,self.current_g.y,self.current_g.z,self.unit_vector,self.dist):
                        self.node.add_edge(self.current_g)
                        ax.plot([self.node.x,self.current_g.x],[self.node.y,self.current_g.y],zs=[self.node.z,self.current_g.z],c='r')
                        break

        print(len(self.vertex_s)+len(self.vertex_g))
        w.world(ax,[],False)
        plt.show()
        return self.tree_s

# to find the final path from the graph created by RRT
class A_star():

    def __init__(self,goal,tree):
        self.agent_goal = goal[0:3]
        self.tree = tree

    #h(x) function
    def hurestic_function(self,x,y,z): 
        self.distance = math.dist([x,y,z],self.agent_goal) # for x,y
        return self.distance

    def priority(self,queue): #find the path with shortest distance that will be selected from the priority queue
        self.min = math.inf
        self.index = 0
        for self.check in range(len(queue)):
            _,self.value,_,_ = queue[self.check]
            if self.value<self.min:
                self.min = self.value
                self.index = self.check #index of the shortest path
        return self.index

    # to check for visited nodes for the A* algorithm
    def check_visited(self,current,visited):
        for self.x,self.y,self.z in visited:
            if current[0]== self.x and current[1]== self.y and current[2]== self.z:
                return True
        return False

    #A* algorithm to find the shortest path from the start orientation to goal orientation
    def a_star(self):
        self.open_set = []
        self.visited = []
        self.start = self.tree
        self.tcost = 0
        self.gcost = 0
        self.path = [[self.tree.x,self.tree.y,self.tree.z,self.tree.roll,self.tree.pitch,self.tree.yaw]]
        self.open_set.append((self.start,self.tcost,self.gcost,self.path))
        while len(self.open_set)>0:
            self.index = self.priority(self.open_set)
            (self.shortest,_,self.gvalue,self.path) = self.open_set[self.index] #select the node with lowest distance
            self.open_set.pop(self.index)
            if not (self.check_visited([(self.shortest.x),(self.shortest.y),(self.shortest.z)],self.visited)): # check if already visited
                self.visited.append([(self.shortest.x),(self.shortest.y),(self.shortest.z)])
                if (self.shortest.x) == self.agent_goal[0] and self.shortest.y == self.agent_goal[1] and self.shortest.z == self.agent_goal[2]: # goal conditions
                    return self.path
                self.neighbours,self.cost_function= self.shortest.get_neighbour()
                for self.neighbour,self.cost in zip(self.neighbours,self.cost_function):#calculate cost of each neighbor
                    self.temp_gcost = self.gvalue+(self.cost)
                    self.temp_tcost = self.temp_gcost+(self.hurestic_function(self.neighbour.x,self.neighbour.y,self.neighbour.z))
                    if not (self.check_visited([(self.neighbour.x),(self.neighbour.y),(self.neighbour.z)],self.visited)):
                        self.open_set.append((self.neighbour,self.temp_tcost,self.temp_gcost,self.path+ [[self.neighbour.x,self.neighbour.y,self.neighbour.z,self.neighbour.roll,self.neighbour.pitch,self.neighbour.yaw]]))
        print("not working")      
        return self.path

def main():

    start_point = [5,2,0.61,0,1.57,0] #start state 
    goal_point = [5,17.5,0.61,0,0,0] #goal state
    rrt = RRT()
    tree = rrt.rrt_connect(start_point,goal_point)
    aStar = A_star(goal_point,tree)
    path = aStar.a_star()

    w = World(10)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for i in range(len(path)-1):
        ax.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],zs=[path[i][2],path[i+1][2]],c='r') #display the final path
    
    w.world(ax,[],False)
    plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(path)): # animate the final path
        plt.cla()
        w.world(ax,path[i],True)
        plt.pause(0.01)
    plt.show()

if __name__ == "__main__":
    main()
