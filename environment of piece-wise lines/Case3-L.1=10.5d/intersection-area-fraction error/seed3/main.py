import numpy as np
from aeropy_fix import jiayou,chazhi,sampling,zhaoweizhi,runaero,Math_integrate,jixu
import os
import time
import torch
import numpy as np
import numpy.random as rd
import math
from copy import deepcopy
from agent import *
from numpy import cos,sin,tan,sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
#from gym.utils import seeding
import matplotlib
import shapely
from shapely.geometry import LineString, Polygon,Point
from pyDOE import lhs 
#from bezier import Line,Bezier,xianchazhi
from scipy import interpolate

from scipy.optimize import curve_fit

if os.path.exists('run.py'):
    from run import *
    
if os.path.exists('run_cpu.py'):
    from run_cpu import *

gongkuang = 10.5

bubu = 0

end = 0.6

flyingh = 0.4* end

allstep = 20 #int( gongkuang *1.75)

initial = 0.6 #0.2

kejieshoukuandai = 0.4*0.01  * 0.1  #0.0125

#chushisudubi = 0.6

#huluejibu = 3

xiangaochao = 3

shijianbu = 0.02
Tzhong = 2.943000000000000*0.6/0.25
flap_max = 9.99999


podu = 0.5

huatusize = 5

allji = [0.3,0.6,0.9]

v_0_ji = v_aim_ji = [8.59920292,8.879512099,9.124996488]

Thrust_0_ji = Thrust_end_ji = [1.277167533,1.54278313,1.615092432]

theta_0_ji = theta_end_ji = [1.805507789,1.77565838,1.773445899]

h_0_ji = h_aim_ji = [0.243938438,0.363940457,0.483940605]

flap_0_ji = flap_end_ji = [-2.622866624,-2.303687751,-2.093826378]

feigao_0_ji = feigao_aim_ji =[0.3*0.4,0.6*0.4,0.9*0.4]


for i in  range(3):
    
    if int(10*end) == int(10*allji[i]) :
        
        
        v_aim = v_aim_ji[i]
        
        Thrust_end = Thrust_end_ji[i]
        
        feigao_aim = feigao_aim_ji[i]
        
        theta_end = theta_end_ji[i]
        
        h_aim = h_aim_ji[i]
        
        flap_end = flap_end_ji[i]

for i in  range(3):
    
   # print('i',i)
    
    if int(10*initial)==int(10*allji[i])  :
        
        v_0 = v_0_ji[i]
        
        Thrust_0 = Thrust_0_ji[i]
        
        feigao_0 = feigao_0_ji[i]
        
        theta_0 = theta_0_ji[i]
        
        #print('theta_0',theta_0)
        
        h_0 = h_0_ji[i] #- kejieshoukuandai
        
        flap_0 = flap_0_ji[i]






L_1 = 100 #+ (gongkuang+0.5)*v_0*shijianbu

L_2 = (gongkuang)*v_0*shijianbu

#L_3 = 10*10*shijianbu   ## 底部平缓带

x_1 = -100
y_1 = 0

x_2 = x_1 + L_1
y_2 = 0

x_3 = x_2 + L_2* cos( -np.pi*podu/180)  
y_3 = y_2 + L_2* sin( -np.pi*podu/180)

# x_4 = x_3 + L_3
# y_4 = y_3 + 0

# X_ji = [x_1,x_2,x_3,x_4]
# Y_ji = [y_1,y_2,y_3,y_4]


X_ji = [x_1,x_2,x_3]
Y_ji = [y_1,y_2,y_3]

for i in range(1):
    
    qian_x = X_ji[-1]
    
    qian_y = Y_ji[-1]
    
    hou_x = qian_x +  L_2* cos( -np.pi*(podu/2.)*((-1)**(i))/180)  *10
    
    hou_y = qian_y +  L_2* sin( -np.pi*(podu/2.)*((-1)**(i))/180)  *10
    
    X_ji.append(hou_x)
    
    Y_ji.append(hou_y)


X_ji.append(X_ji[-1] + 1000)
    
Y_ji.append(0)

orderjieshu = 50

shapejieshu = 50





#allchang = 10*allstep*shijianbu

L_2_max = gongkuang*10*shijianbu

ylim_chuixiang =0.4*initial +  L_2_max*(sin( -np.pi*(podu/2.)*((-1)**(i))/180)  + sin( -np.pi*(podu)*((-1)**(i))/180))

#chushisudubi = 1




lhou=2*0.4
lqian=1.5*0.4
gabove=0.31*0.4
ji=np.load(os.path.abspath(os.path.join(os.getcwd(), "../../../..")) + '//'+ 'datacollection.npy')
vmin=5
vmax=18
lenthsudu=100
suduzu=np.linspace(vmin,vmax,lenthsudu)
lentheta=100
thetazu=np.linspace(-11.5,5.5,lentheta);
hmax=0.4*1.31;  ###相当于1的相对飞高
hmin=0.4*0.41;  ###相当于0.1的相对飞高
lenh=100
hzu=np.linspace(hmin,hmax,lenh)
lenflap=100
flapzu=np.linspace(-10,10,lenflap)





alpha_0 = theta_0

NavDDTheta= np.deg2rad(alpha_0)
NavDDPsi=0
NavDDPhi=0
 
MatDDC_g2b = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
MatDDC_g2b[0,0] = np.cos(NavDDTheta)*np.cos(NavDDPsi)
MatDDC_g2b[0,1] = np.cos(NavDDTheta)*np.sin(NavDDPsi)
MatDDC_g2b[0,2] = -np.sin(NavDDTheta)
MatDDC_g2b[1,0] = np.sin(NavDDPhi)*np.sin(NavDDTheta)*np.cos(NavDDPsi) - np.cos(NavDDPhi)*np.sin(NavDDPsi)
MatDDC_g2b[1,1] = np.sin(NavDDPhi)*np.sin(NavDDTheta)*np.sin(NavDDPsi) + np.cos(NavDDPhi)*np.cos(NavDDPsi)
MatDDC_g2b[1,2] = np.sin(NavDDPhi)*np.cos(NavDDTheta)
MatDDC_g2b[2,0] = np.cos(NavDDPhi)*np.sin(NavDDTheta)*np.cos(NavDDPsi) + np.sin(NavDDPhi)*np.sin(NavDDPsi)
MatDDC_g2b[2,1] = np.cos(NavDDPhi)*np.sin(NavDDTheta)*np.sin(NavDDPsi) - np.sin(NavDDPhi)*np.cos(NavDDPsi)
MatDDC_g2b[2,2] = np.cos(NavDDPhi)*np.cos(NavDDTheta)

Vx0 = v_0

Vz0 = 0

EarthDDG = 9.81

z0 = - h_0

transform_sudu =np.dot(MatDDC_g2b,np.array([Vx0,0,Vz0]).T)

transform_jiasudu=np.dot(MatDDC_g2b,np.array([0,0,EarthDDG]).T) 

state_0 = np.array([alpha_0,0,transform_sudu[0],transform_jiasudu[0],transform_sudu[2],transform_jiasudu[2],0,0,Vx0,Vz0,0,z0])  ###朝向下为正方向



nnum = 0 

class Fuckingfly:
    
    def __init__(self,initia_state_ori, Tzhong,    flap_max,    initial,  shijianbu):
                
        self.state_0 = initia_state_ori[[0,2,3,4,5,6,7,10,11]] ##输出使用  z是11
 
        self.initia_state_ori = initia_state_ori  ##
        
        self.state_ori = initia_state_ori
        
        self.max_step = allstep
        
        self.env_name = 'fuck'
        
        self.xunlian = 0
        
        self.state_dim = 9
        
        self.action_dim= 2
                
        self.if_discrete=False
        
        self.target_return=10000000
        
        self.Tzhong = Tzhong
        self.flap_max = flap_max
        self.initial = initial
        self.shijianbu = shijianbu

        self.jilustates = np.append(initia_state_ori,[0,0,0,0,0,0]) # 两个新加的状态、reward、两个控制量和飞高
     
        self.shineng_last = 0    

        
        self.wave_num = -1
        
        self.z_last = 0
        
        self.x_last = 0
        
        self.z_shuiping = 0 
        
        self.x_now = 0
        
        self.dijibu = 0
        
        

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    



    def bo(self,x):
        
        f = interpolate.interp1d(self.hengbeisaierdian, self.zongbeisaierdian, kind='slinear')
            
        Z = f(x)
        
        return Z 


    def dixing(self,x):

        Z = self.bo(x)
                     
        return Z  
    
    

    def op_xing(self,x):
        
        Z = self.bo(x) + 0.4*initial
                     
        return Z  
    
  
    
    def dixing_group(self,x):
            
            zz = []
            
            for i in x:

                ZZ = self.bo(i)
                                   
                zz.append(ZZ)
                
            Z = np.array(zz)
                
            return Z      
    
   
    def op_xing_group(self,x):
            
            zz = []
            
            for i in x:

                ZZ = self.bo(i) + 0.4*initial
                                   
                zz.append(ZZ)
                
            Z = np.array(zz)
                
            return Z    
        
    
        
    def reset(self,seed_gei=None):
        
        self.dijibu = 0
        
       
        self.state_ori = self.initia_state_ori  # 
        
        #state_0_yuan = self.state_0
        
        if seed_gei==None:
            
            self.T_wave =0#A
                        
        else:
            
            self.T_wave = 0
            

        self.hengbeisaierdian =np.array(X_ji)
        
               
        self.zongbeisaierdian = np.array(Y_ji)

        
        #self.w  = pinglv*np.pi/(pox_3 - pox_2)

        # self.duan = 5000
        
        # self.X=np.linspace(0,pox_3+10,self.duan+1)
        
        # self.Z=np.zeros_like(self.X)        

        # for i in range(len(self.X)):
            
        #     self.Z[i] = self.dixing(self.X[i])
        
        # self.xz = np.vstack((self.X,self.Z)).T
  
        self.X = np.array(X_ji)
        
        self.Z = np.array(Y_ji)
            
        self.xz = np.vstack((self.X,self.Z)).T    
  




        Theta_new = self.state_ori[0]
        
        Ggaodu = -1* self.state_ori[11]
        
        Xgaodu = self.state_ori[10]
   
        xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
        
        zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
                
        xhua_n_should = 0
        
        zhua_n_should = self.dixing(xhua_n_should) + initial*0.4
        
        Ggaodu_should_fu = zhua_n_should - zhua_n + Ggaodu
        
        Gx_should_fu =  xhua_n_should - xhua_n + Xgaodu
        
        self.state_ori[10] = Gx_should_fu
        
        self.state_ori[11] = - Ggaodu_should_fu

        L1  = zhua_n - self.dixing(xhua_n)

        
        self.state_0 = self.state_ori[[0,2,3,4,5,6,7,10,11]]


        state_0_wave =  self.state_0


        self.jilustates = np.append(self.state_ori,[L1,0,0,0,0,0])

        self.z_last = zhua_n_should

        self.x_last = xhua_n_should
        
        self.x_now = 0

    
        # ini,useless = shinengjisuan(self.initial*0.4,xhua_n)
        
        # self.shineng_last = ini

        return state_0_wave.astype(np.float32)    
    
    

    
    def step(self, action):
        
        self.dijibu = self.dijibu + 1
        
        state_next_ori, reward_wancheng, done, fuck_fu = envstep(self.state_ori, action, self.Tzhong,    self.flap_max,  self.initial,    self.shijianbu,   self.jilustates, self.dixing)
        
        self.state_ori = state_next_ori  # state_next_ori[10]是x
        
        indexyao = [0,2,3,4,5,6,7,10,11]
        
        state_next_yuan = state_next_ori[indexyao]


        Theta_new = state_next_ori[0]
        
        Ggaodu = -1* state_next_ori[11]
        
        Xgaodu = state_next_ori[10]
   
        xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
        
        zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
        xiangduigaodu = zhua_n - self.dixing(xhua_n)
        
        fuck_fu[-6] = xiangduigaodu


        state_next_wave = state_next_yuan


        reward_wancheng = 0
        
        self.x_now = xhua_n
        
        #reward = reward + shinengnow - 0*self.shineng_last + reward_bu # + (jiaodujiangli(state_next_yuan[0],L2) + sudujiangli(state_next_ori[8],state_next_ori[9],L2))*0.25
        
        #reward = shinengnow - 0*self.shineng_last + reward_bu # + (jiaodujiangli(state_next_yuan[0],L2) + sudujiangli(state_next_ori[8],state_next_ori[9],L2))*0.25
        
        
        # if self.dijibu < allstep:
            

        #     if done == True :
                
        #         reward = 0
            
    
        #     else:
                
        #         reward = self.shinengjisuan_shape(xhua_n,zhua_n,self.x_last, self.z_last)
                
        # elif  np.isclose(xhua_n, self.x_last):
            
        #     reward = self.shinengjisuan_shape(xhua_n,zhua_n,self.x_last, self.z_last)

        if ~np.isclose(xhua_n, self.x_last):
            
            reward = self.shinengjisuan_shape(xhua_n,zhua_n,self.x_last, self.z_last)
            
        else:
            
            reward = 0
            



        # if self.dijibu <= huluejibu:
            
        #     reward = 0


        fuck_fu[-4] = reward

        reward_out = reward + reward_wancheng
        #self.shineng_last = shinengnow

        if ~np.isclose(xhua_n, self.x_last):
    
            self.jilustates = np.vstack((self.jilustates, fuck_fu ))
        
        self.z_last = zhua_n

        self.x_last = xhua_n
        
        
        return state_next_wave.astype(np.float32) , reward_out , done , {}
 
    
    
    
    def quanrender(self,casenum):
        
        quanhuatu(self.jilustates, self.Tzhong, self.flap_max, self.initial, self.shijianbu, self.xz, casenum)
        
    def render(self,casenum):
        
        huatu(self.jilustates, self.Tzhong, self.flap_max, self.initial, self.shijianbu, self.xz,casenum)
        
        
            
# '''    
#     def shinengjisuan_shape(self,xhua_n,zhua_n,x_last,z_last):
        
        
#              ###### 虚假的
             
#             yinggaigao_n =  self.dixing(xhua_n) + flyingh
             
             
#             if  abs(zhua_n - yinggaigao_n)<= kejieshoukuandai:
                 
#                  zhua_n = yinggaigao_n
             
             
#              ###### 虚假的        
        
#              ###### 虚假的
             
#             yinggaigao_last =  self.dixing(x_last) + flyingh
             
             
#             if  abs(z_last - yinggaigao_last)<= kejieshoukuandai:
                 
#                  z_last = yinggaigao_last
             
             
#              ###### 虚假的      
             
#             if (self.dixing(xhua_n) + 0.4) <zhua_n:
                 
#                  return 0
        
#             if (self.dixing(x_last) + 0.4) <z_last:
                 
#                  return 0        
        
        
         
#             X_op = np.linspace(x_last, xhua_n,20)
            
#             #print('X_op',X_op)
            
#             Y_op = self.op_xing_group(X_op)
            
#             #print('Y_op',Y_op)
            
#             op_coords = np.vstack((X_op,Y_op)).T
            
#             #print('op_coords',op_coords)
                
#             X_wave = np.linspace(xhua_n,x_last,20)
            
#             #print('X_wave',X_wave)
            
#             Y_wave = self.dixing_group(X_wave)
                   
#             #print('Y_wave',Y_wave)
            
#             wave_coords = np.vstack((X_wave,Y_wave)).T
            
#             #print('wave_coords',wave_coords)
            
#             string_op_coords = np.vstack((op_coords,wave_coords))
            
#            # print('string_op_coords',string_op_coords)
            
#             polygon_op = Polygon([tuple(x) for x in string_op_coords.tolist()])
            
#             #print('polygon_op',polygon_op)
            
            
#             # lastdian_shuipingxian_coord = np.array([x_last,z_shuiping])
            
#             # nowdian_shuipingxian_coord = np.array([xhua_n,z_shuiping])        
            
#             Y_shangbound = Y_wave + 0.4
            
#            # print('Y_shangbound',Y_shangbound)
            
#             X_shangbound = X_wave 
            
#           #  print('X_shangbound',X_shangbound)
            
#             shangbound_coords = np.vstack((X_shangbound,Y_shangbound)).T
            
#          #   print('shangbound_coords',shangbound_coords)
            
#             last_tp_coord = np.array([x_last,z_last])
            
#          #   print('last_tp_coord',last_tp_coord)
            
#             now_tp_coord = np.array([xhua_n,zhua_n])
            
#         #    print('now_tp_coord',now_tp_coord)
#             ##开始判断
            
#             line_tp_coords = np.vstack((last_tp_coord,now_tp_coord))
            
#         #    print('line_tp_coords',line_tp_coords)
            
#             line_tp = LineString([tuple(x) for x in line_tp_coords.tolist()])
            
#          #   print('line_tp',line_tp)
            
#             line_shangbound = LineString([tuple(x) for x in shangbound_coords.tolist()])
    
#         #    print('line_shangbound',line_shangbound)
            
#             if ((line_tp.intersects(line_shangbound)) or (Y_shangbound[-1] <= z_last) or (Y_shangbound[0]<= zhua_n)):
            
#                 reward_bu = 0
     
#             else:
                
#                 string_tp_coords = np.vstack((last_tp_coord,now_tp_coord,shangbound_coords))
                
#         #        print('string_tp_coords',string_tp_coords)
                
#                 polygon_tp = Polygon([tuple(x) for x in string_tp_coords.tolist()])    
                
#         #        print('polygon_tp',polygon_tp)
                
                
#                 #v_op = polygon_op.area
                
#                 v_tp = polygon_tp.area
                
#        #         print('v_tp',v_tp)
        
#                 xiangjiao = polygon_tp.intersection(polygon_op)
        
#                 if xiangjiao.is_empty:
                    
#                     reward_bu_fenzi = v_tp
    
#                 else:
                    
#          #           print('xiangjiao.area',xiangjiao.area)
       
#                     reward_bu_fenzi = v_tp-2*xiangjiao.area
    
#                 #print('reward_bu_fenzi',reward_bu_fenzi)
                
#                 #string_op_and_shuiping_coords = np.vstack(( op_coords,nowdian_shuipingxian_coord,lastdian_shuipingxian_coord))
        
        
#                 string_op_and_shuiping_coords = np.vstack(( op_coords,shangbound_coords ))
        
#          #       print('string_op_and_shuiping_coords',string_op_and_shuiping_coords)
#                 polygon_op_and_shuiping = Polygon([tuple(x) for x in string_op_and_shuiping_coords.tolist()])  
        
#        #         print('polygon_op_and_shuiping',polygon_op_and_shuiping)
                
#                 reward_bu_fenmu = polygon_op_and_shuiping.area
        
#                 if reward_bu_fenmu <=0:
                    
#                     return 0
        
#        #         print('reward_bu_fenmu',reward_bu_fenmu)
        
#                 reward_bu = reward_bu_fenzi/reward_bu_fenmu
    
#                 if reward_bu < 0:
                    
#                     reward_bu = 0
    
#             return reward_bu**shapejieshu
                

# '''

    def shinengjisuan_shape(self,xhua_n,zhua_n,x_last,z_last):
        
        
             ###### 虚假的
             
            yinggaigao_n =  self.dixing(xhua_n) + flyingh
             
             
            if  abs(zhua_n - yinggaigao_n)<= kejieshoukuandai:
                 
                 zhua_n = yinggaigao_n
             
             
             ###### 虚假的        
        
             ###### 虚假的
             
            yinggaigao_last =  self.dixing(x_last) + flyingh
             
             
            if  abs(z_last - yinggaigao_last)<= kejieshoukuandai:
                 
                 z_last = yinggaigao_last
             
             
             ###### 虚假的      
             
            if (self.dixing(xhua_n) + 0.4) <zhua_n:
                 
                 return 0
        
            if (self.dixing(x_last) + 0.4) <z_last:
                 
                 return 0        
        
        
        

            
            
                
            X_wave = np.linspace(xhua_n,x_last,20)
            
            Y_wave = self.dixing_group(X_wave)        
        
        
    
        
            Y_shangbound = Y_wave + 0.4
            
            X_shangbound = X_wave 
            
            

            shangbound_coords = np.vstack((X_shangbound,Y_shangbound)).T
        
        
        
            last_tp_coord = np.array([x_last,z_last])
                        
            now_tp_coord = np.array([xhua_n,zhua_n])
                        
            line_tp_coords = np.vstack((last_tp_coord,now_tp_coord))
            
            line_tp = LineString([tuple(x) for x in line_tp_coords.tolist()])
            
            line_shangbound = LineString([tuple(x) for x in shangbound_coords.tolist()])        
        


        
            if ((line_tp.intersects(line_shangbound)) or (Y_shangbound[-1] <= z_last) or (Y_shangbound[0]<= zhua_n)):
            
                reward_bu = 0
     
            else:         

                

                X_op = np.linspace(x_last, xhua_n,20)
                
                Y_op = self.op_xing_group(X_op)
                
                
                
                
                            
                op_coords = np.vstack((X_op,Y_op)).T                

                wave_coords = np.vstack((X_wave,Y_wave)).T
                
                
                
                ## 下面的完整 交 上面的轨迹包围
                
                
                
                
                string_op_coords = np.vstack((op_coords,wave_coords))      
    
                
                polygon_op = Polygon([tuple(x) for x in string_op_coords.tolist()])
                

                string_tp_coords = np.vstack((last_tp_coord,now_tp_coord,shangbound_coords))
                
        #        print('string_tp_coords',string_tp_coords)
                
                polygon_tp = Polygon([tuple(x) for x in string_tp_coords.tolist()])    
                
        #        print('polygon_tp',polygon_tp)
        
        
        
        

        
                xiangjiao = polygon_tp.intersection(polygon_op)
                
                
                if xiangjiao.is_empty:
                    
                    part_1_area = 0
    
                else:
                    
         #           print('xiangjiao.area',xiangjiao.area)
       
                    part_1_area = xiangjiao.area                
                


                
                
                ## 上面的完整 交 下面的轨迹包围

                string_op_coords = np.vstack((op_coords,shangbound_coords))     
    
                
                polygon_op = Polygon([tuple(x) for x in string_op_coords.tolist()])
                

                string_tp_coords = np.vstack((last_tp_coord,now_tp_coord,wave_coords))
                
        #        print('string_tp_coords',string_tp_coords)
                
                polygon_tp = Polygon([tuple(x) for x in string_tp_coords.tolist()])    
                
        #        print('polygon_tp',polygon_tp)                
                
                
                
                xiangjiao = polygon_tp.intersection(polygon_op)
                
                
                if xiangjiao.is_empty:
                    
                    part_2_area = 0
    
                else:
                    
         #           print('xiangjiao.area',xiangjiao.area)
       
                    part_2_area = xiangjiao.area                 
                
                
                
                
                
                
                
                
                
                
                X_wave_turn = np.linspace(x_last,xhua_n,20)
                
                Y_wave_turn = self.dixing_group(X_wave_turn)        
            

                Y_shangbound_turn = Y_wave_turn + 0.4
                
                X_shangbound_turn = X_wave_turn 
                
                shangbound_coords_turn = np.vstack((X_shangbound_turn,Y_shangbound_turn)).T                   
                    
                    
                
                
                
                
                
    
        
                string_wave_and_shangbound_coords = np.vstack(( wave_coords,shangbound_coords_turn ))
        
         #       print('string_op_and_shuiping_coords',string_op_and_shuiping_coords)
                polygon_wave_and_shangbound = Polygon([tuple(x) for x in string_wave_and_shangbound_coords.tolist()])  
        
       #         print('polygon_op_and_shuiping',polygon_op_and_shuiping)
                
                reward_quan = polygon_wave_and_shangbound.area
        

        
       #         print('reward_bu_fenmu',reward_bu_fenmu)
        
                reward_bu = (reward_quan - part_2_area -  part_1_area)/reward_quan
    

    
            return reward_bu**shapejieshu
                









        
    def shinengjisuan_LLcha(self,L1): 
        
            xiangduigaodu = L1 
        
             ###### 虚假的
             
            yinggaigao =   flyingh
             
             
            if  abs(L1 - yinggaigao)<= kejieshoukuandai:
                 
                 xiangduigaodu = yinggaigao
             
             
             ###### 虚假的        
        
       
            # xiangduigaodu = L1 
           
            fenmu = 0.4*initial
                           
            #cost_n =  1 - np.clip(xiangduigaodu/fenmu,0,1) 
            
            cost_n =  (1 - np.clip(abs((xiangduigaodu-fenmu)/fenmu),0,1))
            
            return cost_n**orderjieshu     


    def suan_lingyifang(self):
        
        LLcha = self.jilustates[:,12]

        #LLcha = LLcha[huluejibu+1:]

        allsum = 0
        
        for nnum in range(len(LLcha)):
            
            if nnum ==0:
                
                #print('diyibu',shinengjisuan_LLcha(LLcha[nnum]))
                
                continue
            
            
            allsum = allsum + self.shinengjisuan_LLcha(LLcha[nnum])
            
        
        return allsum




    def suan_shencha_subhanshu(self,xhua_n,zhua_n,x_last,z_last):           ####检查最后的结果上上下下的子函数

         if (self.dixing(xhua_n) + 0.4) <zhua_n:
             
             return 0
    
         if (self.dixing(x_last) + 0.4) <z_last:
             
             return 0  
             
         X_op = np.linspace(x_last, xhua_n,20)
         
         #print('X_op',X_op)
         
         Y_op = self.op_xing_group(X_op)
         
         #print('Y_op',Y_op)
         
         op_coords = np.vstack((X_op,Y_op)).T
         
         #print('op_coords',op_coords)
             
         X_wave = np.linspace(xhua_n,x_last,20)
         
         #print('X_wave',X_wave)
         
         Y_wave = self.dixing_group(X_wave)
                
         #print('Y_wave',Y_wave)
         
         wave_coords = np.vstack((X_wave,Y_wave)).T
         
         #print('wave_coords',wave_coords)
         
         string_op_coords = np.vstack((op_coords,wave_coords))
         
        # print('string_op_coords',string_op_coords)
         
         polygon_op = Polygon([tuple(x) for x in string_op_coords.tolist()])
         
         #print('polygon_op',polygon_op)
         
         
         # lastdian_shuipingxian_coord = np.array([x_last,z_shuiping])
         
         # nowdian_shuipingxian_coord = np.array([xhua_n,z_shuiping])        
         
         Y_shangbound = Y_wave + 0.4
         
        # print('Y_shangbound',Y_shangbound)
         
         X_shangbound = X_wave 
         
       #  print('X_shangbound',X_shangbound)
         
         shangbound_coords = np.vstack((X_shangbound,Y_shangbound)).T
         
      #   print('shangbound_coords',shangbound_coords)
         
         last_tp_coord = np.array([x_last,z_last])
         
      #   print('last_tp_coord',last_tp_coord)
         
         now_tp_coord = np.array([xhua_n,zhua_n])
         
        
 
     #    print('line_shangbound',line_shangbound)
     
     
         string_shang_coords = np.vstack((op_coords,shangbound_coords))
     
         polygon_shang = Polygon([tuple(x) for x in string_shang_coords.tolist()])         
     
         v_shang = polygon_shang.area
         

         
         string_tp_coords = np.vstack((last_tp_coord,now_tp_coord,shangbound_coords))
         
 #        print('string_tp_coords',string_tp_coords)
         
         polygon_tp = Polygon([tuple(x) for x in string_tp_coords.tolist()])    
         
 #        print('polygon_tp',polygon_tp)

         
         v_tp = polygon_tp.area
         
#         print('v_tp',v_tp)
 
         xiangjiao = polygon_tp.intersection(polygon_op)
 
         if xiangjiao.is_empty:
             
             reward_bu_fenzi = v_shang - v_tp
 
         else:
             
  #           print('xiangjiao.area',xiangjiao.area)

             reward_bu_fenzi = 2*xiangjiao.area + v_shang - v_tp
 
         return reward_bu_fenzi
        
        
        
    def suan_shencha_zhuhanshu(self):           ####检查最后的结果上上下下的主函数
        

        xzuobiao = self.jilustates[:,10]
        
        zzuobiao = -self.jilustates[:,11]
        
        # xzuobiao = xzuobiao[huluejibu+1:]
        
        # zzuobiao = zzuobiao[huluejibu+1:]
        #LLcha = self.jilustates[:,12]
        
        jiao = self.jilustates[:,0]
        
        xhua, zhua =[] , [] 
        
        gabove = 0.31*0.4
        
        for a,b,c in zip(xzuobiao,zzuobiao,jiao):
             
             Theta_new = c
             
             Ggaodu = b
             
             Xgaodu = a
        
             xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
             
             zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
             
             xhua.append(xhua_n)
             
             
             
             ###### 虚假的
             
             yinggaigao =  self.dixing(xhua_n) + flyingh
             
             
             if  abs(zhua_n - yinggaigao)<= kejieshoukuandai:
                 
                 zhua_n = yinggaigao
             
             
             ###### 虚假的             
             

             
             
             
             
             zhua.append(zhua_n)
            
        allsum = 0
        
        for num in range(len(xhua)-2):
            
            AAA = self.suan_shencha_subhanshu(xhua[num+1],zhua[num+1],xhua[num],zhua[num])
            
            #print('AAA',AAA)
            
            allsum = allsum + AAA   
            
        
        return allsum

             
    # ############## 势能计算
   
    # def shinengjisuan(self,xhua_n,zhua_n): # op: optimal path ; tp: true path
       
    #     # qian_op_z = self.op_xing(self.x_last) 
        
    #     # now_op_z =  self.op_xing(xhua_n) 
        
    #     # qian_op_h = abs( self.z_shuiping - qian_op_z )
        
    #     # now_op_h = abs( self.z_shuiping - now_op_z )
        
    #     # qian_tp_h = abs( self.z_shuiping - self.z_last )
        
    #     # now_tp_h = abs( self.z_shuiping - zhua_n )
    #     if (self.dixing(xhua_n) + 0.4) <zhua_n:
             
    #          return 0
    

    #     X_op = np.linspace(self.x_last, xhua_n,20)
        
    #     Y_op = self.op_xing_group(X_op)
        
    #     op_coords = np.vstack((X_op,Y_op)).T
            
    #     X_wave = np.linspace(xhua_n,self.x_last,20)
        
    #     Y_wave = self.dixing_group(X_wave)
               
    #     wave_coords = np.vstack((X_wave,Y_wave)).T
    
    #     string_op_coords = np.vstack((op_coords,wave_coords))
        
    #     polygon_op = Polygon([tuple(x) for x in string_op_coords.tolist()])
        
        
        
    #     # lastdian_shuipingxian_coord = np.array([self.x_last,self.z_shuiping])
        
    #     # nowdian_shuipingxian_coord = np.array([xhua_n,self.z_shuiping])        
        
    #     Y_shangbound = Y_wave + 0.4
        
    #     X_shangbound = X_wave 
        
    #     shangbound_coords = np.vstack((X_shangbound,Y_shangbound)).T
        
    #     last_tp_coord = np.array([self.x_last,self.z_last])
        
    #     now_tp_coord = np.array([xhua_n,zhua_n])
        
        
    #     ##开始判断
        
    #     line_tp_coords = np.vstack((last_tp_coord,now_tp_coord))
        
    #     line_tp = LineString([tuple(x) for x in line_tp_coords.tolist()])

        
    #     line_shangbound = LineString([tuple(x) for x in shangbound_coords.tolist()])

        
    #     if ((line_tp.intersects(line_shangbound)) or (Y_shangbound[-1] <= self.z_last) or (Y_shangbound[0]<= zhua_n)):
        
    #         reward_bu = 0
 
    #     else:
            
    #         string_tp_coords = np.vstack((last_tp_coord,now_tp_coord,shangbound_coords))
            
    #         polygon_tp = Polygon([tuple(x) for x in string_tp_coords.tolist()])    
            
            
    #         #v_op = polygon_op.area
            
    #         v_tp = polygon_tp.area
            
    
    #         xiangjiao = polygon_tp.intersection(polygon_op)
    
    #         if xiangjiao.is_empty:
                
    #             reward_bu_fenzi = v_tp
            
    #         else:
                
    #             reward_bu_fenzi = v_tp-2*xiangjiao.area
    
            
    #         #string_op_and_shuiping_coords = np.vstack(( op_coords,nowdian_shuipingxian_coord,lastdian_shuipingxian_coord))
    
    
    #         string_op_and_shuiping_coords = np.vstack(( op_coords,shangbound_coords ))
    
    #         polygon_op_and_shuiping = Polygon([tuple(x) for x in string_op_and_shuiping_coords.tolist()])  
    
            
            
    #         reward_bu_fenmu = polygon_op_and_shuiping.area
    
    #         #print('reward_bu_fenmu',reward_bu_fenmu)
    
    #         reward_bu = reward_bu_fenzi/reward_bu_fenmu

    #         if reward_bu < 0:
                
    #             reward_bu = 0

    #     return reward_bu**shapejieshu
       
     

  
def envstep(statearg, actionarg, Tzhong,  flap_max,  initial,  shijianbu,  feixingjilu, dixing  ):

    done=False
   
    flap,Thrust = actionarg[0]*flap_max, (actionarg[1]+1) * Tzhong/2
    
    
    
    
    Theta = statearg[0]
    Thetadot = statearg[1]
    u=statearg[2]
    udot=statearg[3]
    w=statearg[4]
    wdot=statearg[5]
    q=statearg[6]
    qdot=statearg[7]
    Vx=statearg[8]
    Vz=statearg[9]  ##朝下为正
    x=statearg[10]
    
    z_wave = -dixing(x)  ## zheng
    
    z= statearg[11] - z_wave
    
    Ggaoduold = -z
    
    feigaoold=Ggaoduold-gabove*math.cos(Theta*math.pi/180)
    
    #print('feigaoold',feigaoold)
    
    cost_1 = 0
    
    Thetasuan= np.arctan(Vz/Vx)+Theta 
    
    if Thetasuan+flap>10 or Thetasuan+flap<-10:
        
       # print('fuck Thetasuan+flap')
        
        done = True
                
        cost_1 = -np.clip((abs(Thetasuan+flap)-10)/10,0,1)
        
        next_state = statearg
        
        feigao = feigaoold
                
        cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
        temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)

    if np.isnan(flap):
    
        done = True
                
        cost_1 = -1
        
        next_state = statearg
        
        feigao = feigaoold
                
        cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
        temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)



    if (-z)>=hmax:
        #print('fuck_no')
        
        signal_ori=chazhi(Vx,Thetasuan,hmax,flap,suduzu,thetazu,hzu,flapzu,ji)
        
        #print('fuck_yes')
        
    else:
        
        signal_ori=chazhi(Vx,Thetasuan,-1*z,flap,suduzu,thetazu,hzu,flapzu,ji)
    
    
    signal_sec=jiayou(Theta,-1*z,Thrust)
    signal=signal_ori+signal_sec
     
    Fx=signal[0]
    Fz=-1*signal[1]
    My=signal[2]
     
    Fy=0.  
    Mx=0.  
    Mz=0.
    Phi,Phidot=0.,0.   
    Psi,Psidot=0.,0.    
    v,vdot=0.,0.         
    p,pdot=0.,0.  
    r,rdot=0.,0.     
    Vy=0.   
    y=0.    
         
    outcome=runaero(Fx, Fy, Fz,Mx,My,Mz, \
             Phi,Theta,Psi,Phidot,Thetadot,Psidot, \
             u,v,w,udot,vdot,wdot, \
             p,q,r,pdot,qdot,rdot, \
             Vx,Vy,Vz,  \
             x,y,z,shijianbu,20,-20)
 
    Theta_new=outcome[1]
    Thetadot_new=outcome[4]
    u_new=outcome[6]
    udot_new=outcome[9]
    w_new=outcome[8]
    wdot_new=outcome[11]
    q_new=outcome[13]
    qdot_new=outcome[16]
    Vx_new=outcome[18]
    Vz_new=outcome[20]
    x_new=outcome[21]
    z_new=outcome[23]
    
    next_state=np.array([Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new + z_wave ])

    jieshoubu = allstep-2
    
    if x_new > (jieshoubu+2) * shijianbu * 20:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
            
            cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
            cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
            temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
        
            return next_state, cost, done, np.append(next_state,temp)
                
    if x_new < -1 * shijianbu * 5:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
            
            cost_2,cost_3,cost_4,cost_5,cost_6,cost_7,cost_8=-1,-1,-1,-1,-1,-1,-1
        
            cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
        
            temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])

            return next_state, cost, done, np.append(next_state,temp)            
    
    z_new_wave = -dixing(x_new)
     
    Ggaodu= z_new_wave -( z_new + z_wave )

    
    feigao = Ggaodu-gabove*math.cos(Theta_new*math.pi/180)
    
    Thetasuan = np.arctan(Vz_new/Vx_new) + Theta_new 
      


    cost_2 = 0

    if Thetasuan<-11.5:
        
        done=True
         
        next_state = statearg
         
        feigao = feigaoold
            
        cost_2 = -np.clip( (np.abs(Thetasuan)-11.5)/10 , 0 , 1 )
         
        cost_3,cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 

        temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)

    if Thetasuan>5.5:
        
        done=True
         
        next_state = statearg
        
        feigao = feigaoold
                    
        cost_2 = -np.clip( (np.abs(Thetasuan)-5.5)/10 , 0 , 1 )
        
        cost_3,cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1,-1
        
        cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 

        temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
        
        return next_state, cost, done, np.append(next_state,temp)



    cost_3 = -1
    
    # if dixing(x_new) <= ( z_new + z_wave):
    
    #     done=True
         
    #     next_state = statearg
        
    #     feigao = feigaoold
                    
    #     cost_3 = -1
        
    #     cost_4,cost_5,cost_6,cost_7,cost_8 = -1,-1,-1,-1,-1
        
    #     cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 

    #     temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
        
    #     return next_state, cost, done, np.append(next_state,temp)


    cost_4 = 0 
    
    
    zhongdez = - ( z_new + z_wave )
    
    
    qiantou_x = x_new + gabove*math.sin(Theta_new*math.pi/180) + lqian*math.cos(Theta_new*math.pi/180)
    
    qiantou_z = zhongdez-gabove*math.cos(Theta_new*math.pi/180)+lqian*math.sin(Theta_new*math.pi/180)
    
    houtou_x = x_new - (lhou*math.cos(Theta_new*math.pi/180) - gabove*math.sin(Theta_new*math.pi/180))
    
    houtou_z = zhongdez-gabove*math.cos(Theta_new*math.pi/180)-lhou*math.sin(Theta_new*math.pi/180)
    
    qian_wave_z = dixing(qiantou_x)
    
    hou_wave_z = dixing(houtou_x)
    
    
    #print('qiantou_z - qian_wave_z',qiantou_z - qian_wave_z)
    
    #print('houtou_z - hou_wave_z',houtou_z - hou_wave_z)

    
    
    if qiantou_z - qian_wave_z <= 0 :
        
        done=True
        
        next_state = statearg
        
        feigao = feigaoold
        
        #cost_4 = -np.clip( (0.1*0.4-(qiantou_z - qian_wave_z))/0.4, 0 , 1 )
        cost_4 = -np.clip( (0-(qiantou_z - qian_wave_z))/0.4, 0 , 1 )

    if  houtou_z - hou_wave_z  <=  0:
        
        done=True
        
        next_state = statearg
        
        feigao = feigaoold
        
        #cost_4 = -np.clip( (0.1*0.4-(houtou_z - hou_wave_z))/0.4, 0 , 1 )
        cost_4 = -np.clip( (0-(qiantou_z - qian_wave_z))/0.4, 0 , 1 )



    cost_5 = -1
    # if Theta_new>0:
    #     di=Ggaodu-gabove*math.cos(Theta_new*math.pi/180)-lhou*math.sin(Theta_new*math.pi/180)
         
    # if Theta_new<0:
    #     di=Ggaodu-gabove*math.cos(Theta_new*math.pi/180)+lqian*math.sin(Theta_new*math.pi/180)
         
    # if Theta_new==0:
    #     di=Ggaodu-gabove
     
    # if di < 0.1*0.4:
    #      #print('fuck di')
    #     done=True
        
    #     next_state = statearg
        
    #     feigao = feigaoold
        
    #     cost_5 =  -np.clip( (0.1*0.4-di)/0.4, 0 , 1 )

        
         

    cost_6 = 0
     
    if Ggaodu >= hmax:
         
          # done=True
         
          # next_state = statearg
         
          # feigao = feigaoold
          #cost_6 = -np.clip( (Ggaodu-hmax)/0.4, 0 , 1 )
          cost_6 = -(Ggaodu-hmax)/0.4
          
          
          if Ggaodu >= hmax + xiangaochao*0.4:
              
               done=True
             
               next_state = statearg
             
               feigao = feigaoold              
         
          
    elif Ggaodu <= hmin:
             
             done = True
             
             next_state = statearg
             
             feigao = feigaoold
             
             cost_6 =  -np.clip( (hmin-Ggaodu)/0.4, 0 , 1 )
             






    cost_7 = 0
     
    if Vx_new >= vmax:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_7 =  -np.clip( (Vx_new-vmax)/(vmax-vmin), 0 , 1 )
         

    if vmin >= Vx_new:
         
         done=True
         
         next_state = statearg
         
         feigao = feigaoold
         
         cost_7 = -np.clip( (vmin-Vx_new)/(vmax-vmin), 0 , 1 )
         


    cost_8 = 0
    
    if Ggaodu < 0.4*0.5:
        
    
        if done==False:
            
            if jixu(Vx_new,Thetasuan,Ggaodu,0,suduzu,thetazu,hzu,flapzu,ji)==False:
                
                done = True
                
                next_state = statearg
                
                feigao = feigaoold
                
                cost_8 = -1
        
        if done==True:
    
                cost_8 = -1    
            

          
     
    if feixingjilu.shape[0]==1:
        
        bushu = 0

    else:
        
        bushu = len(feixingjilu) - 1
    

        
    jieshoubu = allstep-2
    
    if x_new > (jieshoubu+2) * shijianbu * 20:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold       
                
    if x_new < -1 * shijianbu * 5:
        
            done = True
            
            next_state = statearg
            
            feigao = feigaoold                 
        

    if bushu >  jieshoubu :
         
         done = True
         



    cost = (cost_1 + cost_2 + cost_3 + cost_4 + cost_5 + cost_6 + cost_7 + cost_8 + 8  )/6 
    
    temp = np.append(np.zeros(bubu+1),[cost,0,flap,Thrust, feigao ])
    
    return next_state, cost, done, np.append(next_state,temp)





  
def envstepforMPC(statearg,  shijianbu=shijianbu  ):

    Theta = statearg[0]
    Thetadot = statearg[1]
    u=statearg[2]
    udot=statearg[3]
    w=statearg[4]
    wdot=statearg[5]
    q=statearg[6]
    qdot=statearg[7]
    Vx=statearg[8]
    Vz=statearg[9]  ##朝下为正
    x=statearg[10]
    z=statearg[11]

    Fx=0
    Fz=0
    My=0
     
    Fy=0.  
    Mx=0.  
    Mz=0.
    Phi,Phidot=0.,0.   
    Psi,Psidot=0.,0.    
    v,vdot=0.,0.         
    p,pdot=0.,0.  
    r,rdot=0.,0.     
    Vy=0.   
    y=0.    
         
    outcome=runaero(Fx, Fy, Fz,Mx,My,Mz, \
             Phi,Theta,Psi,Phidot,Thetadot,Psidot, \
             u,v,w,udot,vdot,wdot, \
             p,q,r,pdot,qdot,rdot, \
             Vx,Vy,Vz,  \
             x,y,z,shijianbu,20,-20)
 
    Theta_new=outcome[1]
    Thetadot_new=outcome[4]
    u_new=outcome[6]
    udot_new=outcome[9]
    w_new=outcome[8]
    wdot_new=outcome[11]
    q_new=outcome[13]
    qdot_new=outcome[16]
    Vx_new=outcome[18]
    Vz_new=outcome[20]
    x_new=outcome[21]
    z_new=outcome[23]
    
    next_state=np.array([Theta_new,Thetadot_new,u_new,udot_new,w_new,wdot_new,q_new,qdot_new,Vx_new,Vz_new,x_new,z_new ])
  
    return next_state

















matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


#np.set_printoptions(precision=3)

fucknum=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def huatu(jilustates,  Tzhong,    flap_max,    initial,  shijianbu,xz, casenum_ori):
    
    casenum=casenum_ori
    #print('jilustates',jilustates)
    
    xzuobiao = jilustates[:,10]
    
    zzuobiao = -jilustates[:,11]
    
    jiao = jilustates[:,0]
    
    global fucknum
    
    fucknum[casenum]+=1
    
    pathh = os.getcwd()
    
    initial_zifu = str( int(initial*10))
    
    end_zifu = str( int(end*10))
    
    wenjianjia = initial_zifu  + 'to' + end_zifu +'_'+ str(bubu )+'bu_case_' + str(casenum)
    
    if not os.path.exists(pathh+'\\' + wenjianjia):
            
        os.mkdir(pathh+'\\' + wenjianjia )
    
    mingcheng = pathh+'\\'+ wenjianjia+ '\\' + str(fucknum[casenum])+'.png'
    
    #data_save = os.path.join(pathh, '\\', wenjianjia, '\\', str(fucknum)+'.npy')
    data_save =pathh+ '\\'+ wenjianjia+ '\\'+ str(fucknum[casenum])+'.npy'
    
    #show_save =pathh+ '\\'+ wenjianjia+ '\\show'+ str(fucknum)+'.npy'
    
    np.save(data_save,jilustates)
    
    dixing_save = pathh+ '\\'+ wenjianjia+ '\\'+ 'dixing'+ str(fucknum[casenum])+'.npy'
    np.save(dixing_save,xz)   
    
    fig=plt.figure(figsize=(18,9))
        
    atr=[]
    
    linestylezu=['solid','dotted','dashed','dashdot']  #4
    markerzu=["v","^","<",">","s","p","P","X","D","d"]  #10
    
    for a in range(len(linestylezu)):
        for b in range(len(markerzu)):
            atr.append([a,b])
        
    random.shuffle(atr)
        
    axes1 = fig.add_subplot(1,1,1)
    #axes2 = fig.add_subplot(2,1,2)
    
    #RR=0.1
    
    xhua, zhua =[] , [] 
    
    gabove = 0.31*0.4
    
    for a,b,c in zip(xzuobiao,zzuobiao,jiao):
         
         Theta_new = c
         
         Ggaodu = b
         
         Xgaodu = a
    
         xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
         
         zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
         xhua.append(xhua_n)
         
         zhua.append(zhua_n)
        
    axes1.plot( xhua , zhua ,label='path',linestyle='-',marker="o",alpha=0.8,markersize= huatusize)
    
    # for a,b,c in zip(xzuobiao,zzuobiao,jiao):
        
        # axes1.arrow(a,b,RR * np.cos(np.pi*c/180),RR * np.sin(np.pi*c/180),lw=0.01, fc='k', ec='k',linestyle="-")
    # suosuoyin = np.where(xz[:,0]>max(xhua))
    
    # suoyinzhi = suosuoyin[0]
    
    # mo = suoyinzhi[0]+10
    
    #axes1.plot(xz[:,0],xz[:,1],linestyle='--')
    axes1.plot(xz[:,0],xz[:,1]+0.4*end,linestyle='-')
    axes1.plot(xz[:,0],xz[:,1]+0.4*end+kejieshoukuandai,linestyle='--')
    axes1.plot(xz[:,0],xz[:,1]+0.4*end-kejieshoukuandai,linestyle='--')
    # axes1.plot([0,xzuobiao[-1]],[0.4*initial,0.4*initial],linestyle='--')
    # axes1.plot([0,xzuobiao[-1]],[0.4*end,0.4*end],linestyle='--')
    
    axes1.spines['right'].set_color('none')
    axes1.spines['top'].set_color('none')
    
    
    font1 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 18,
    }
    font2 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 23,
    }
    font3 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 25,
    }
    legend = plt.legend(prop=font2)#,ncol=3,loc = 'upper center',borderaxespad=0.1,borderpad=0.1,columnspacing=0.5,handlelength=1.5,handletextpad=0.4)  
    
    axes1.set_xlabel('x',font3)
    
    axes1.set_ylabel('z',font3,rotation= 0)
    #ax.set_ylabel(r'$h$',font2,rotation=0)
    
    #ding  =  [initial, end]
    
    axes1.set_xlim([0, xzuobiao[-1]])
    axes1.set_ylim([ylim_chuixiang, 0.4*initial+0.4*0.005])   
    #axes1.set_ylim([min(xz[:,1]), 1*0.4+A])
    # axes1.set_xlim([xzuobiao[0],math.ceil(max(xhua))])
    
    # axes1.set_ylim([math.floor(min(zhua)),math.ceil(max(zhua))])
    
    
    #plt.tight_layout()
    
    axes1.xaxis.labelpad = 0
    axes1.yaxis.labelpad = 30
    
    axes1.tick_params(labelsize=20)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
    


    
    plt.savefig(mingcheng,dpi=300)
    
    plt.close()



fuckquannum=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def quanhuatu(jilustates,  Tzhong,    flap_max,    initial,    shijianbu, xz,casenum_ori):
    
    casenum=casenum_ori
    
    #print('jilustates',jilustates)
    
    xzuobiao = jilustates[:,10]
    
    zzuobiao = -jilustates[:,11]
    
    jiao = jilustates[:,0]
    
    global fuckquannum
    
    fuckquannum[casenum] +=1
    
    pathh = os.getcwd()
    
    initial_zifu = str( int(initial*10))
    
    end_zifu = str( int(end*10))
    
    wenjianjia = 'quanhua'+ initial_zifu + 'to' + end_zifu +'_'+ str(bubu )+'bu_case_' + str(casenum)
    
    if not os.path.exists(pathh+'\\' + wenjianjia):
            
        os.mkdir(pathh+'\\' + wenjianjia )
    
    mingcheng = pathh+'\\'+ wenjianjia+ '\\' + str(fuckquannum[casenum])+'.png'
    
    #data_save = os.path.join(pathh, '\\', wenjianjia, '\\', str(fucknum)+'.npy')
    data_save =pathh+ '\\'+ wenjianjia+ '\\'+ str(fuckquannum[casenum])+'.npy'
    
    #show_save =pathh+ '\\'+ wenjianjia+ '\\show'+ str(fucknum)+'.npy'
    
    np.save(data_save,jilustates)
    
    dixing_save = pathh+ '\\'+ wenjianjia+ '\\'+ 'dixing'+ str(fuckquannum[casenum])+'.npy'
    np.save(dixing_save,xz)
    
    
    
    fig=plt.figure(figsize=(18,9))
        
    atr=[]
    
    linestylezu=['solid','dotted','dashed','dashdot']  #4
    markerzu=["v","^","<",">","s","p","P","X","D","d"]  #10
    
    for a in range(len(linestylezu)):
        for b in range(len(markerzu)):
            atr.append([a,b])
        
    random.shuffle(atr)
        
    axes1 = fig.add_subplot(1,1,1)
    #axes2 = fig.add_subplot(2,1,2)
    
    #RR=0.1
    
    xhua, zhua =[] , [] 
    
    gabove = 0.31*0.4
    
    for a,b,c in zip(xzuobiao,zzuobiao,jiao):
         
         Theta_new = c
         
         Ggaodu = b
         
         Xgaodu = a
    
         xhua_n = Xgaodu + gabove*math.sin(Theta_new*math.pi/180)
         
         zhua_n = Ggaodu-gabove*math.cos(Theta_new*math.pi/180) 
         
         xhua.append(xhua_n)
         
         zhua.append(zhua_n)
        
    axes1.plot( xhua , zhua ,label='path',linestyle='-',marker="o",alpha=0.8,markersize= huatusize)
    
    # for a,b,c in zip(xzuobiao,zzuobiao,jiao):
        
        # axes1.arrow(a,b,RR * np.cos(np.pi*c/180),RR * np.sin(np.pi*c/180),lw=0.01, fc='k', ec='k',linestyle="-")

    # suosuoyin = np.where(xz[:,0]>max(xhua))
    
    # suoyinzhi = suosuoyin[0]
    
    # mo = suoyinzhi[0]+10
    
    #axes1.plot(xz[:,0],xz[:,1],linestyle='--')
    axes1.plot(xz[:,0],xz[:,1]+0.4*end,linestyle='-')
    axes1.plot(xz[:,0],xz[:,1]+0.4*end+kejieshoukuandai,linestyle='--')
    axes1.plot(xz[:,0],xz[:,1]+0.4*end-kejieshoukuandai,linestyle='--')
    # axes1.plot([0,xzuobiao[-1]],[0.4*end,0.4*end],linestyle='--')
    
    axes1.spines['right'].set_color('none')
    axes1.spines['top'].set_color('none')
    
    
    font1 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 18,
    }
    font2 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 23,
    }
    font3 = {'family' : 'STIXGeneral',
    'weight' : 'normal',
    'size'   : 25,
    }
    legend = plt.legend(prop=font2)#,ncol=3,loc = 'upper center',borderaxespad=0.1,borderpad=0.1,columnspacing=0.5,handlelength=1.5,handletextpad=0.4)  
    
    axes1.set_xlabel('x',font3)
    
    axes1.set_ylabel('z',font3,rotation= 0)
    #ax.set_ylabel(r'$h$',font2,rotation=0)
    
    #ding  =  [initial, end]
    
    #axes1.set_ylim([min(xz[:,1]), 1*0.4 + A])
    axes1.set_xlim([0, xzuobiao[-1]])
    axes1.set_ylim([ylim_chuixiang, 0.4*initial+0.4*0.005])   
    # axes1.set_xlim([xzuobiao[0],math.ceil(max(xhua))])
    
    # axes1.set_ylim([math.floor(min(zhua)),math.ceil(max(zhua))])    
    #plt.tight_layout()
    
    axes1.xaxis.labelpad = 0
    axes1.yaxis.labelpad = 30
    
    axes1.tick_params(labelsize=20)
    
    plt.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.15)
    


    
    plt.savefig(mingcheng,dpi=300)
    
    plt.close()











def demo3_custom_env_rl (    Tzhong,    flap_max,    initial, end,  shijianbu  ):
    
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 1943

    '''choose an DRL algorithm'''
    #from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    args.agent.if_use_gae = True
    
    args.bubushu = bubu

    jieshoubu = allstep -2


    args.env = Fuckingfly(state_0, Tzhong,    flap_max,    initial,   shijianbu)
    args.env_eval = Fuckingfly(state_0, Tzhong,    flap_max,    initial,   shijianbu)  
    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 <
      # break training after 'total_step > break_step'
    
    args.jieshoubu = jieshoubu
    
    
    args.if_remove = False
    
    args.if_allow_break = False
    
    #args.max_memo = 2 ** 21
    #args.batch_size = 2 ** 9
    #args.repeat_times = 2 ** 1
    args.eval_gap = 2 **4  # for Recorder
    # args.eval_times1 = 2 ** 1  # for Recorder
    # args.eval_times2 = 2 ** 3  # for Recorder

    # train_and_evaluate(args)
    #args.rollout_num = 4
    args.eval_times1 = 2 ** 0
    args.eval_times1 = 2 **0
    #args.if_per = True

    args.gamma =  1-1./allstep

    train_and_evaluate(args,initial, end)






    

if __name__ == '__main__':

    demo3_custom_env_rl (  Tzhong,    flap_max,    initial, end,shijianbu )

