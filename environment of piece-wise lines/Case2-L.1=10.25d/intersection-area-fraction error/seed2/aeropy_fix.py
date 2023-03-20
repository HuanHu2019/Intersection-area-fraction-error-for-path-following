# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 18:15:05 2020

@author: haha
"""
import numpy as np

from numpy import cos,sin,tan,sqrt

from scipy.interpolate import Rbf

from sklearn.preprocessing import StandardScaler

def Math_integrate( Para_last,Para_dot_last,Para_dot,DT ):
    
    Para_integrate = Para_last + 0.5*( Para_dot + Para_dot_last)*DT
    
    return  Para_integrate



def runaero(ForceFx, ForceFy, ForceFz,ForceMx, ForceMy, ForceMz,Nav_lastPhi, Nav_lastTheta, Nav_lastPsi, Nav_lastPhi_dot, Nav_lastTheta_dot, Nav_lastPsi_dot,Body_lastu, Body_lastv, Body_lastw, Body_lastu_dot, Body_lastv_dot, Body_lastw_dot,Body_lastp, Body_lastq, Body_lastr, Body_lastp_dot, Body_lastq_dot, Body_lastr_dot,Nav_lastVx,Nav_lastVy,Nav_lastVz,Nav_lastx, Nav_lasty, Nav_lastz, shijianbu, shangxianjiao, xiaxianjiao):

    
    
    ForceDDFx=ForceFx
    ForceDDFy=ForceFy
    ForceDDFz=ForceFz
    ForceDDMx=ForceMx
    ForceDDMy=ForceMy
    ForceDDMz=ForceMz
    Nav_lastDDPhi = np.deg2rad(Nav_lastPhi)
    Nav_lastDDTheta = np.deg2rad(Nav_lastTheta)
    Nav_lastDDPsi = np.deg2rad(Nav_lastPsi)
    Nav_lastDDPhi_dot = np.deg2rad(Nav_lastPhi_dot)
    Nav_lastDDTheta_dot = np.deg2rad(Nav_lastTheta_dot)
    Nav_lastDDPsi_dot = np.deg2rad(Nav_lastPsi_dot)
    Body_lastDDu = Body_lastu
    Body_lastDDv = Body_lastv
    Body_lastDDw = Body_lastw
    Body_lastDDu_dot = Body_lastu_dot
    Body_lastDDv_dot =Body_lastv_dot
    Body_lastDDw_dot = Body_lastw_dot
    Body_lastDDp = np.deg2rad(Body_lastp)
    Body_lastDDq = np.deg2rad(Body_lastq)
    Body_lastDDr = np.deg2rad(Body_lastr)
    Body_lastDDp_dot = np.deg2rad(Body_lastp_dot)
    Body_lastDDq_dot = np.deg2rad(Body_lastq_dot)
    Body_lastDDr_dot = np.deg2rad(Body_lastr_dot)
    Nav_lastDDVx=Nav_lastVx
    Nav_lastDDVy=Nav_lastVy
    Nav_lastDDVz=Nav_lastVz
    Nav_lastDDx=Nav_lastx
    Nav_lastDDy=Nav_lasty
    Nav_lastDDz=Nav_lastz
    
    
    
    
    
    
    
    NavDDVx=Nav_lastDDVx
    NavDDVy=Nav_lastDDVy
    NavDDVz=Nav_lastDDVz
    NavDDx=Nav_lastDDx
    NavDDy=Nav_lastDDy
    NavDDz=Nav_lastDDz
    
    NavDDPhi=Nav_lastDDPhi
    NavDDTheta=Nav_lastDDTheta
    NavDDPsi=Nav_lastDDPsi
    NavDDPhi_dot=Nav_lastDDPhi_dot
    NavDDTheta_dot=Nav_lastDDTheta_dot
    NavDDPsi_dot=Nav_lastDDPsi_dot
    
    BodyDDu=Body_lastDDu
    BodyDDv=Body_lastDDv
    BodyDDw=Body_lastDDw
    BodyDDu_dot=Body_lastDDu_dot
    BodyDDv_dot=Body_lastDDv_dot
    BodyDDw_dot=Body_lastDDw_dot
    BodyDDp=Body_lastDDp
    BodyDDq=Body_lastDDq
    BodyDDr=Body_lastDDr
    BodyDDp_dot=Body_lastDDp_dot
    BodyDDq_dot=Body_lastDDq_dot
    BodyDDr_dot=Body_lastDDr_dot

    
    # Nav=Nav_last;
    #Body=Body_last;
    
    SimDDDT = shijianbu
    
    MassDDWeight = 1.6*0.75##0.06
    MassDDIxx = 0 #np.nan
    MassDDIxy = 0
    MassDDIxz = 0
    MassDDIyx = 0
    MassDDIyy = 0.25*0.75
    MassDDIyz = 0
    MassDDIzx = 0
    MassDDIzy = 0
    MassDDIzz = 0#np.nan
    
    MassDDI = np.array([[MassDDIxx,MassDDIxy,MassDDIxz],[MassDDIyx,MassDDIyy,MassDDIyz],[MassDDIzx,MassDDIzy,MassDDIzz]])
    
    EarthDDG0 = 9.81
    EarthDDG = EarthDDG0
     
    
    # 四元数
    NavDDQ_0 = cos(NavDDPhi/2)*cos(NavDDTheta/2)*cos(NavDDPsi/2) + sin(NavDDPhi/2)*sin(NavDDTheta/2)*sin(NavDDPsi/2)               
    NavDDQ_1 = sin(NavDDPhi/2)*cos(NavDDTheta/2)*cos(NavDDPsi/2) - cos(NavDDPhi/2)*sin(NavDDTheta/2)*sin(NavDDPsi/2)
    NavDDQ_2 = cos(NavDDPhi/2)*sin(NavDDTheta/2)*cos(NavDDPsi/2) + sin(NavDDPhi/2)*cos(NavDDTheta/2)*sin(NavDDPsi/2)
    NavDDQ_3 = cos(NavDDPhi/2)*cos(NavDDTheta/2)*sin(NavDDPsi/2) - sin(NavDDPhi/2)*sin(NavDDTheta/2)*cos(NavDDPsi/2)
    
    ##### 坐标变换矩阵初始化
    #  地面坐标系转到机体坐标系
    MatDDC_g2b = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    MatDDC_g2b[0,0] = cos(NavDDTheta)*cos(NavDDPsi)
    MatDDC_g2b[0,1] = cos(NavDDTheta)*sin(NavDDPsi)
    MatDDC_g2b[0,2] = -sin(NavDDTheta)
    MatDDC_g2b[1,0] = sin(NavDDPhi)*sin(NavDDTheta)*cos(NavDDPsi) - cos(NavDDPhi)*sin(NavDDPsi)
    MatDDC_g2b[1,1] = sin(NavDDPhi)*sin(NavDDTheta)*sin(NavDDPsi) + cos(NavDDPhi)*cos(NavDDPsi)
    MatDDC_g2b[1,2] = sin(NavDDPhi)*cos(NavDDTheta)
    MatDDC_g2b[2,0] = cos(NavDDPhi)*sin(NavDDTheta)*cos(NavDDPsi) + sin(NavDDPhi)*sin(NavDDPsi)
    MatDDC_g2b[2,1] = cos(NavDDPhi)*sin(NavDDTheta)*sin(NavDDPsi) - sin(NavDDPhi)*cos(NavDDPsi)
    MatDDC_g2b[2,2] = cos(NavDDPhi)*cos(NavDDTheta)
    
    
    MatDDQ_g2b = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    MatDDQ_g2b[0,0] = 1 - 2 * (NavDDQ_2**2 + NavDDQ_3**2);
    MatDDQ_g2b[0,1] = 2 * (NavDDQ_1 * NavDDQ_2 + NavDDQ_0 * NavDDQ_3)
    MatDDQ_g2b[0,2] = 2 * (NavDDQ_1 * NavDDQ_3 - NavDDQ_0 * NavDDQ_2)
    MatDDQ_g2b[1,0] = 2 * (NavDDQ_1 * NavDDQ_2 - NavDDQ_0 * NavDDQ_3)
    MatDDQ_g2b[1,1] = 1 - 2 * (NavDDQ_1**2 + NavDDQ_3**2)
    MatDDQ_g2b[1,2] = 2 * (NavDDQ_2 * NavDDQ_3 + NavDDQ_0 * NavDDQ_1)
    MatDDQ_g2b[2,0] = 2 * (NavDDQ_1 * NavDDQ_3 + NavDDQ_0 * NavDDQ_2)
    MatDDQ_g2b[2,1] = 2 * (NavDDQ_2 * NavDDQ_3 - NavDDQ_0 * NavDDQ_1)
    MatDDQ_g2b[2,2] = 1 - 2 * (NavDDQ_1**2 + NavDDQ_2**2)
    
    #  机体坐标系角速度到欧拉角变化率
    MatDDC_body2euler=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    
    MatDDC_body2euler[0,0] = 1.
    MatDDC_body2euler[0,1] = sin(NavDDPhi)*tan(NavDDTheta)
    MatDDC_body2euler[0,2] = cos(NavDDPhi)*tan(NavDDTheta)
    MatDDC_body2euler[1,0] = 0.
    MatDDC_body2euler[1,1] = cos(NavDDPhi)
    MatDDC_body2euler[1,2] = -sin(NavDDPhi)
    MatDDC_body2euler[2,0] = 0.
    MatDDC_body2euler[2,1] = sin(NavDDPhi)*1./cos(NavDDTheta)
    MatDDC_body2euler[2,2] = cos(NavDDPhi)*1./cos(NavDDTheta)
    
    #  机体坐标系转到地面坐标系
    MatDDC_b2g=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    
    MatDDC_b2g = MatDDC_g2b.T
    MatDDQ_b2g = MatDDQ_g2b.T
    
    
    #无论什么式子，得到的宗伟numpy的行向量
    
    
    # 机体坐标系速度变化率（加速度）
    temp = np.cross(np.array([BodyDDu,BodyDDv,BodyDDw]).T,np.array([BodyDDp,BodyDDq,BodyDDr]).T) + np.dot(MatDDC_g2b,np.array([ForceDDFx,ForceDDFy,ForceDDFz]).T)/MassDDWeight + np.dot(MatDDC_g2b,np.array([0,0,EarthDDG]).T)                                  
    BodyDDu_dot = temp[0]
    BodyDDv_dot = temp[1]
    BodyDDw_dot = temp[2]
    

    MassDDIforinvers=MassDDI.copy()
    MassDDIforinvers[0,0]=np.Inf
    MassDDIforinvers[2,2]=np.Inf
    
    # 机体坐标系角速度变化率（角加速度）
    temp = np.dot( np.linalg.inv(MassDDIforinvers) , -np.cross(np.array([BodyDDp,BodyDDq,BodyDDr]).T,np.dot(MassDDI,np.array([BodyDDp,BodyDDq,BodyDDr]).T).T) + np.array([ForceDDMx,ForceDDMy,ForceDDMz]).T)
    BodyDDp_dot = temp[0]
    BodyDDq_dot = temp[1]
    BodyDDr_dot = temp[2]
    
    # 对机体坐标系速度变化率积分获得速度，角速度变化率积分获得角速度
    BodyDDu = Math_integrate(Body_lastDDu,Body_lastDDu_dot,BodyDDu_dot,SimDDDT)
    BodyDDv = Math_integrate(Body_lastDDv,Body_lastDDv_dot,BodyDDv_dot,SimDDDT)
    BodyDDw = Math_integrate(Body_lastDDw,Body_lastDDw_dot,BodyDDw_dot,SimDDDT)
    BodyDDp = Math_integrate(Body_lastDDp,Body_lastDDp_dot,BodyDDp_dot,SimDDDT)
    BodyDDq = Math_integrate(Body_lastDDq,Body_lastDDq_dot,BodyDDq_dot,SimDDDT)
    BodyDDr = Math_integrate(Body_lastDDr,Body_lastDDr_dot,BodyDDr_dot,SimDDDT)

    # 将机体坐标系角速度转为欧拉角变化率
    temp = np.dot(MatDDC_body2euler, np.array([BodyDDp,BodyDDq,BodyDDr]).T);
    NavDDPhi_dot = temp[0]
    NavDDTheta_dot = temp[1]
    NavDDPsi_dot = temp[2]

    NavDDPhi = Math_integrate(Nav_lastDDPhi,Nav_lastDDPhi_dot,NavDDPhi_dot,SimDDDT)
    NavDDTheta = Math_integrate(Nav_lastDDTheta,Nav_lastDDTheta_dot,NavDDTheta_dot,SimDDDT)
    NavDDPsi = Math_integrate(Nav_lastDDPsi,Nav_lastDDPsi_dot,NavDDPsi_dot,SimDDDT)
    
    # if NavDDPhi > np.pi:
    #     NavDDPhi = NavDDPhi - 2*np.pi
    
    # if NavDDPhi < -np.pi:
    #     NavDDPhi = NavDDPhi + 2*np.pi
        
    # if NavDDTheta > np.pi:
    #     NavDDTheta = NavDDTheta - 2*np.pi
    
    # if NavDDTheta < -np.pi:
    #     NavDDTheta = NavDDTheta + 2*np.pi
        
    # if NavDDPsi > np.pi:
    #     NavDDPsi = NavDDPhi - 2*np.pi
    
    # if NavDDPsi < -np.pi:
    #     NavDDPsi = NavDDPhi + 2*np.pi
        
    if NavDDTheta>np.deg2rad(shangxianjiao) or NavDDTheta < np.deg2rad(xiaxianjiao) :
        
        jiaopan = 0
        
    else:
        
        jiaopan = 1
        
    # 更新地面坐标系到机体坐标系转换矩阵
    MatDDC_g2b=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    
    MatDDC_g2b[0,0] = cos(NavDDTheta)*cos(NavDDPsi)
    MatDDC_g2b[0,1] = cos(NavDDTheta)*sin(NavDDPsi)
    MatDDC_g2b[0,2] = -sin(NavDDTheta)
    MatDDC_g2b[1,0] = sin(NavDDPhi)*sin(NavDDTheta)*cos(NavDDPsi) - cos(NavDDPhi)*sin(NavDDPsi)
    MatDDC_g2b[1,1] = sin(NavDDPhi)*sin(NavDDTheta)*sin(NavDDPsi) + cos(NavDDPhi)*cos(NavDDPsi)
    MatDDC_g2b[1,2] = sin(NavDDPhi)*cos(NavDDTheta)
    MatDDC_g2b[2,0] = cos(NavDDPhi)*sin(NavDDTheta)*cos(NavDDPsi) + sin(NavDDPhi)*sin(NavDDPsi)
    MatDDC_g2b[2,1] = cos(NavDDPhi)*sin(NavDDTheta)*sin(NavDDPsi) - sin(NavDDPhi)*cos(NavDDPsi)
    MatDDC_g2b[2,2] = cos(NavDDPhi)*cos(NavDDTheta)
    
    MatDDC_body2euler=np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
    MatDDC_body2euler[0,0] = 1
    MatDDC_body2euler[0,1] = sin(NavDDPhi)*tan(NavDDTheta)
    MatDDC_body2euler[0,2] = cos(NavDDPhi)*tan(NavDDTheta)
    MatDDC_body2euler[1,0] = 0
    MatDDC_body2euler[1,1] = cos(NavDDPhi)
    MatDDC_body2euler[1,2] = -sin(NavDDPhi)
    MatDDC_body2euler[2,0] = 0
    MatDDC_body2euler[2,1] = sin(NavDDPhi)*1./cos(NavDDTheta)
    MatDDC_body2euler[2,2] = cos(NavDDPhi)*1./cos(NavDDTheta)
    # 更新机体坐标系到地面坐标系转换矩阵
    MatDDC_b2g = MatDDC_g2b.T

 
    # 将机体坐标系速度转换到导航坐标系
    temp = np.dot(MatDDC_b2g , np.array([BodyDDu,BodyDDv,BodyDDw]).T)
    
    NavDDVx = temp[0]
    NavDDVy = temp[1]
    NavDDVz = temp[2]
    
    NavDDVg = sqrt(NavDDVx**2 + NavDDVy**2)
    
    NavDDx = Math_integrate(Nav_lastDDx,Nav_lastDDVx,NavDDVx,SimDDDT)
    NavDDy = Math_integrate(Nav_lastDDy,Nav_lastDDVy,NavDDVy,SimDDDT)
    NavDDz = Math_integrate(Nav_lastDDz,Nav_lastDDVz,NavDDVz,SimDDDT)


    NavPhi=np.rad2deg(NavDDPhi)
    NavTheta=np.rad2deg( NavDDTheta)
    NavPsi=np.rad2deg( NavDDPsi)
    NavPhi_dot=np.rad2deg(NavDDPhi_dot)
    NavTheta_dot=np.rad2deg(NavDDTheta_dot)
    NavPsi_dot=np.rad2deg(NavDDPsi_dot)
    Bodyu=BodyDDu
    Bodyv=BodyDDv
    Bodyw=BodyDDw
    Bodyu_dot=BodyDDu_dot
    Bodyv_dot=BodyDDv_dot
    Bodyw_dot=BodyDDw_dot
    Bodyp=np.rad2deg( BodyDDp)
    Bodyq=np.rad2deg( BodyDDq)
    Bodyr=np.rad2deg( BodyDDr)
    Bodyp_dot=np.rad2deg(BodyDDp_dot)
    Bodyq_dot=np.rad2deg(BodyDDq_dot)
    Bodyr_dot=np.rad2deg(BodyDDr_dot)
    NavVx=NavDDVx
    NavVy=NavDDVy
    NavVz=NavDDVz
    Navx=NavDDx
    Navy=NavDDy
    Navz=NavDDz
    
    return [NavPhi, NavTheta, NavPsi, NavPhi_dot, NavTheta_dot, NavPsi_dot, \
                Bodyu, Bodyv, Bodyw, Bodyu_dot, Bodyv_dot, Bodyw_dot, \
				Bodyp, Bodyq, Bodyr, Bodyp_dot, Bodyq_dot, Bodyr_dot, \
				NavVx,NavVy,NavVz,\
				Navx, Navy, Navz, jiaopan]
 

#fuck=runaero(25.,0.,-100.,0.,29.,0.,0.,2.,0.,0.,5.,0.,  7.,0., -9. ,10.,0.,-12.,0.,14.,0.,0.,17.,0.,19.,0.,-21.,22.,0.,-24.,0.01,180,0)

def zhaoweizhi(shulie,yao):
    xia=np.array(np.where(shulie>yao))
    shang=np.array(np.where(shulie<yao))
    # print('xia')
    # print(xia)
    # print('shang')
    # print(shang)
    
    ashuxia=xia[0,0]
    ashushang=shang[0,-1]
    return [ashuxia,ashushang]





def sampling(shuzu,hou,lentheta=100,lenh=100,lenflap=100):
    
    samples=[]
    
    ge1=lentheta*lenh*lenflap

    ge2=lenh*lenflap

    ge3=lenflap
    
    for q1 in hou[0]:
        for q2 in hou[1]:
            for q3 in hou[2]:
                for q4 in hou[3]:
                    samples.append(q1*ge1+q2*ge2+q3*ge3+q4)
    
    return samples


def jixu(Vx,Thetasuan,gaodu,flap,suduzu,thetazu,hzu,flapzu,ji):
    
    he=[zhaoweizhi(suduzu,Vx),zhaoweizhi(thetazu,Thetasuan),zhaoweizhi(hzu,gaodu),zhaoweizhi(flapzu,flap)]
    
    suoyin=sampling(ji,he)
    
    sample_untre=ji[suoyin]
    
    sample=sample_untre[~np.isnan(sample_untre).any(axis=1), :] 
    
    if sample.size ==0:
        
        return False
    
    else:
        
        return True

    
    



def chazhi(Vx,Thetasuan,gaodu,flap,suduzu,thetazu,hzu,flapzu,ji):
    
    # print('Vx')
    
    # print(Vx)
    
    # print('Thetasuan')
    
    # print(Thetasuan)
    
    # print('gaodu')
    
    # print(gaodu)    
    
    # print('flap')
    
    # print(flap)    
    

    
    gaodu = gaodu -0.00000001
    
    # print('Vx',Vx)
    # print('Thetasuan',Thetasuan)
    # print('gaodu',gaodu)
    # print('flap',flap)    
    
    
    if np.isnan(flap):
        print('flap',flap)
    
    
    he=[zhaoweizhi(suduzu,Vx),zhaoweizhi(thetazu,Thetasuan),zhaoweizhi(hzu,gaodu),zhaoweizhi(flapzu,flap)]
    
    suoyin=sampling(ji,he)
    
    sample_untre=ji[suoyin]
    
    sample=sample_untre[~np.isnan(sample_untre).any(axis=1), :] 
    
    F1,F2,F3,F4 = StandardScaler(),StandardScaler(),StandardScaler(),StandardScaler()
    
    Vx_std=F1.fit_transform(sample[:,0].reshape(-1, 1))
    Thetasuan_std=F2.fit_transform(sample[:,1].reshape(-1, 1))
    gaodu_std=F3.fit_transform(sample[:,2].reshape(-1, 1))
    flap_std=F4.fit_transform(sample[:,3].reshape(-1, 1))
    
    Vx_t = F1.transform(np.array(Vx).reshape(-1, 1))
    
    Thetasuan_t = F2.transform(np.array(Thetasuan).reshape(-1, 1))
    
    gaodu_t = F3.transform(np.array(gaodu).reshape(-1, 1))
    
    flap_t = F4.transform(np.array(flap).reshape(-1, 1))
    
    
    rbfiFx = Rbf(Vx_std, Thetasuan_std, gaodu_std,flap_std, sample[:,5]) 
    Fx = rbfiFx(Vx_t, Thetasuan_t, gaodu_t, flap_t)  
    
    rbfiFz = Rbf(Vx_std, Thetasuan_std, gaodu_std,flap_std, sample[:,6])  
    Fz = rbfiFz(Vx_t, Thetasuan_t, gaodu_t, flap_t)    
    
    rbfiMy = Rbf(Vx_std, Thetasuan_std, gaodu_std,flap_std, sample[:,7]) 
    My = rbfiMy(Vx_t, Thetasuan_t, gaodu_t, flap_t) 

    
    
    return np.array([float(Fx),float(Fz),float(My)])


def jiayou(Theta,gaodu,tuili):
    
    zxgaocha=0.25*0.4
    
    engjuchang = -0.125*0.4
    
    engjugao = 0.5*0.4- zxgaocha
    
    
    engx = engjuchang * cos(Theta)+ engjugao*sin(Theta) 
    engz = -engjuchang * sin(Theta)+ engjugao*cos(Theta) + gaodu
    

    enginexyz =np.array([engx,0,engz])
     
    enginethrust =tuili
    
    engineunitvector = np.array([-cos(Theta), 0, sin(Theta)])
    
    georef_point=np.array([0,0,gaodu])
    
    enginemoments=( np.cross(enginexyz-georef_point,enginethrust*engineunitvector) ) * 2 
    
    engineF =( enginethrust*engineunitvector ) * 2    
    
   
    return np.array([engineF[0],engineF[2],enginemoments[1]])



