import numpy as np
from PIL import Image
image=np.array(Image.open("image.jpg"),np.int)

# 定数
## 画像のサイズ 
H=image.shape[0]
W=image.shape[1]
## 状態数
K=2
S1=2
S2=3
## 変分ベイズの計算回数
I=30
## カラー画像
D=3
##分母が0になることを防ぐ
epsilon=10E-10
##修正領域
###(I_Hs,I_ws),(I_Hs,I_Wf),(I_Hf,I_Ws),(I_Hf,I_Wf)の長方形
I_Hs=60
I_Hf=60
I_Ws=60
I_Wf=60

#import
import numpy as np
from scipy.special import digamma
from matplotlib import pyplot as plt
from PIL import Image

#乱数の固定
np.random.seed(seed=2)

# 拡張分離型格子隠れマルコフモデル
class Extended_HMMs:
    
    def __init__(self,IMAGE):
        
        # 画像データ
        self.IMAGE=IMAGE
        
        #q(\theta)のパラメーター
        
        ##ディリクレ分布(初期状態確率)
        ###初期値
        self.ETA1_0=np.array([0.6,0.4])
        self.ETA2_0=np.array([0.3,0.4,0.3])
        ###更新用
        self.ETA1=self.ETA1_0.copy()
        self.ETA2=self.ETA2_0.copy()
        
        ##ディリクレ分布(遷移確率)
        ###初期値
        self.EPSILON1_0=np.array([[3,1],[1,3]])
        self.EPSILON2_0=np.array([[1,3,1],[1,1,3],[3,1,1]])
        ###更新用
        self.EPSILON1=self.EPSILON1_0.copy()
        self.EPSILON2=self.EPSILON2_0.copy()
        
        ##ガウスウィシャート(平均ベクトル，分散共分散行列)
        ###初期値
        self.BETA_0=np.ones((S1,S2))*10.0
        self.M_0=np.zeros((S1,S2,D,1))
        self.M_0[0,0,:,:]=np.array([[0.0],[0.0],[255.0]])
        self.M_0[0,1,:,:]=np.array([[0.0],[255.0],[0.0]])
        #self.M_0[0,2,:,:]=np.array([[255.0],[0.0],[0.0]])
        self.M_0[1,0,:,:]=np.array([[255.0],[255.0],[0.0]])
        self.M_0[1,1,:,:]=np.array([[0.0],[255.0],[255.0]])
        #self.M_0[1,2,:,:]=np.array([[255.0],[0.0],[255.0]])
        #self.M_0[2,0,:,:]=np.array([[255.0],[255.0],[255.0]])
        #self.M_0[2,1,:,:]=np.array([[0.0],[0.0],[0.0]])
        #self.M_0[2,2,:,:]=np.array([[128.0],[128.0],[128.0]])
        self.NU_0=np.ones((S1,S2))*3
        self.W_INV_0=np.zeros((S1,S2,D,D))
        for s1 in range(S1):
            for s2 in range(S2):
                self.W_INV_0[s1,s2,:,:]=np.identity(D)*1000
        ###更新用
        self.BETA=self.BETA_0.copy()
        self.M=self.M_0.copy()
        self.NU=self.NU_0.copy()
        self.W_INV=self.W_INV_0.copy()
        
        #q(O_I)のパラメーター
        
        ##平均ベクトル
        self.MU=np.zeros((H,W,D,1))
        
        ##分散共分散行列
        self.LAMBDA=np.zeros((H,W,D,D))
        
        #logのついているもの
        self.LOG_N1=np.zeros((H,W,S1))
        self.LOG_N2=np.zeros((H,W,S2))
        self.LOG_PI1=np.zeros((S1))
        self.LOG_PI2=np.zeros((S2))
        self.LOG_A1=np.zeros((S1,S1))
        self.LOG_A2=np.zeros((S2,S2))
        
        #q(S)
        #self.Q_S1=Q_S1
        #self.Q_S1/=self.Q_S1.sum(axis=2,keepdims=True)
        #self.Q_S2=Q_S2
        #self.Q_S2/=self.Q_S2.sum(axis=2,keepdims=True)
        #self.Q_SS1=Q_SS1
        #self.Q_SS1/=self.Q_SS1.sum(axis=(2,3),keepdims=True)
        #self.Q_SS2=Q_SS2
        #self.Q_SS2/=self.Q_SS2.sum(axis=(2,3),keepdims=True)
        self.Q_S1=np.ones((H,W,S1))
        self.Q_S1/=self.Q_S1.sum(axis=2,keepdims=True)
        self.Q_S2=np.ones((H,W,S2))
        self.Q_S2/=self.Q_S2.sum(axis=2,keepdims=True)
        self.Q_SS1=np.ones((H,W,S1,S1))
        self.Q_SS1/=self.Q_SS1.sum(axis=(2,3),keepdims=True)
        self.Q_SS2=np.ones((H,W,S2,S2))
        self.Q_SS2/=self.Q_SS2.sum(axis=(2,3),keepdims=True)
        
        #backforward
        self.alpha_s1=np.zeros((H,W,S1))
        self.alpha_s2=np.zeros((H,W,S2))
        self.beta_s1=np.zeros((H,W,S1))
        self.beta_s2=np.zeros((H,W,S2))
        
    #q(\theta)の更新
    def update_q_theta(self):
    
        #\etaの更新
        
        ##\eta1の更新
        self.ETA1=self.ETA1_0.copy()
        for s1 in range(S1):
            for i in range(0,H):
                self.ETA1[s1]+=self.Q_S1[i,0,s1]
        
        ##\eta2の更新
        self.ETA2=self.ETA2_0.copy()
        for s2 in range(S2):
            for j in range(0,W):
                self.ETA2[s2]+=self.Q_S2[0,j,s2]
        
        ##epsilon1の更新
        self.EPSILON1=self.EPSILON1_0.copy()
        for s1_1 in range(S1):
            for s1_2 in range(S1):
                for i in range(0,H):
                    for j in range(1,W):
                        self.EPSILON1[s1_1,s1_2]+=self.Q_SS1[i,j,s1_1,s1_2]
                
        ##epsilon2の更新
        self.EPSILON2=self.EPSILON2_0.copy()
        for s2_1 in range(S2):
            for s2_2 in range(S2):
                for j in range(0,W):
                    for i in range(1,H):
                        self.EPSILON2[s2_1,s2_2]+=self.Q_SS2[i,j,s2_1,s2_2]
                        
        #\betaの更新
        self.BETA=self.BETA_0.copy()
        for s1 in range(S1):
            for s2 in range(S2):
                for i in range(0,H):
                    for j in range(0,W):
                        if not(i>=I_Hs and i<=I_Hf and j>=I_Ws and j<=I_Wf):
                            self.BETA[s1,s2]+=self.Q_S1[i,j,s1]*self.Q_S2[i,j,s2]                 
        #Mの更新
        self.M=self.M_0.copy()
        for s1 in range(S1):
            for s2 in range(S2):
                self.M[s1,s2,:,:]=self.BETA_0[s1,s2]*self.M_0[s1,s2,:,:]
                for i in range(0,H):
                    for j in range(0,W):
                        #(i,j) \in U \setminus I
                        if not(i>=I_Hs and i<=I_Hf and j>=I_Ws and j<=I_Wf):
                            self.M[s1,s2,:,:]+=self.Q_S1[i,j,s1]*self.Q_S2[i,j,s2]*np.array([self.IMAGE[i,j,:]]).T
                self.M[s1,s2,:,:]=self.M[s1,s2,:,:]/self.BETA[s1,s2]
                
        #NUの変更
        self.NU=self.NU_0.copy()
        for s1 in range(S1):
            for s2 in range(S2):
                for i in range(0,H):
                    for j in range(0,W):
                        if not(i>=I_Hs and i<=I_Hf and j>=I_Ws and j<=I_Wf):
                            self.NU[s1,s2]+=self.Q_S1[i,j,s1]*self.Q_S2[i,j,s2]
    
                            
        #W_INVの更新
        self.W_INV=self.W_INV_0.copy()
        for s1 in range(S1):
            for s2 in range(S2):
                self.W_INV[s1,s2,:,:]=self.W_INV_0[s1,s2,:,:]+self.BETA_0[s1,s2]*np.dot(self.M_0[s1,s2,:,:],self.M_0[s1,s2,:,:].T)-self.BETA[s1,s2]*np.dot(self.M[s1,s2,:,:],self.M[s1,s2,:,:].T)
                for i in range(0,H):
                    for j in range(0,W):
                        #(i,j) \in U \setminus I
                        if not(i>=I_Hs and i<=I_Hf and j>=I_Ws and j<=I_Wf):
                            self.W_INV[s1,s2,:,:]+=self.Q_S1[i,j,s1]*self.Q_S2[i,j,s2]*np.array([self.IMAGE[i,j,:]]).T*np.array([self.IMAGE[i,j,:]]) 
                                                        
    #LOGの項
    def update_e_log(self):
        
        #LOG_N1,LOG_N2
        W_D=np.zeros((S1,S2,D,D))
        E_LOG_LAMBDA=np.zeros((S1,S2))
        E_LOG_N=np.zeros((H,W,S1,S2))
        for s1 in range(S1):
            for s2 in range(S2):
                W_D[s1,s2,:,:]=np.linalg.inv(self.W_INV[s1,s2,:,:])
                E_LOG_LAMBDA[s1,s2]=-D*np.log(2*np.pi)/2+digamma(self.NU[s1,s2]/2)/2+digamma((self.NU[s1,s2]-1)/2)/2+digamma((self.NU[s1,s2]-2)/2)/2+D*np.log(2)/2+np.log(np.linalg.det(W_D[s1,s2,:,:]))/2
         
        for i in range (0,H):
            for j in range(0,W):
                #(i,j)\in I
                if not(i>=I_Hs and i<=I_Hf and j>=I_Ws and j<=I_Wf):
                    for s1 in range(S1):
                        for s2 in range(S2):
                            E_LOG_N[i,j,s1,s2]=-np.dot(np.dot(self.IMAGE[i,j,:],(self.NU[s1,s2]*W_D[s1,s2,:,:])),np.array([self.IMAGE[i,j,:]]).T)/2-np.dot(np.dot(self.M[s1,s2,:,:].T,(self.NU[s1,s2]*W_D[s1,s2,:,:])),self.M[s1,s2,:,:])/2+np.dot(np.dot(self.M[s1,s2,:,:].T,(self.NU[s1,s2]*W_D[s1,s2,:,:])),np.array([self.IMAGE[i,j,:]]).T)-D/(2*self.BETA[s1,s2])                            
                for s1_1 in range(S1):
                    for s2_1 in range(S2):
                        self.LOG_N1[i,j,s1_1]+=self.Q_S2[i,j,s2_1]*(E_LOG_LAMBDA[s1_1,s2_1]+E_LOG_N[i,j,s1_1,s2_1])
                for s2_2 in range(S2):
                    for s1_2 in range(S1):
                        self.LOG_N2[i,j,s2_2]+=self.Q_S1[i,j,s1_2]*(E_LOG_LAMBDA[s1_2,s2_2]+E_LOG_N[i,j,s1_2,s2_2])                          
                
        #LOG_PI,LOG_A
        eta1=0
        for s1 in range(S1):
            eta1+=self.ETA1[s1]
        
        for s1 in range(S1):
            self.LOG_PI1[s1]=digamma(self.ETA1[s1])-digamma(eta1)
            
        eta2=0
        for s2 in range(S2):
            eta2+=self.ETA2[s2]
        
        for s2 in range(S2):
            self.LOG_PI2[s2]=digamma(self.ETA2[s2])-digamma(eta2)
        
        ep1=np.zeros((S1))
        for s1_1 in range(S1):
            for s1_2 in range(S1):
                ep1[s1_1]+=self.EPSILON1[s1_1,s1_2]
                
        for s1_1 in range(S1):
            for s1_2 in range(S1):
                self.LOG_A1[s1_1,s1_2]=digamma(self.EPSILON1[s1_1,s1_2])-digamma(ep1[s1_1])        
        
        ep2=np.zeros((S2))
        for s2_1 in range(S2):
            for s2_2 in range(S2):
                ep2[s2_1]+=self.EPSILON2[s2_1,s2_2]
                
        for s2_1 in range(S2):
            for s2_2 in range(S2):
                self.LOG_A2[s2_1,s2_2]=digamma(self.EPSILON2[s2_1,s2_2])-digamma(ep2[s2_1])         
        
    #forward1(横方向)
    def forward1(self,p_init,p_trans,p_out,alpha1):
        for j in range(0,W):
            if (j==0):
                for i in range(0,H):
                    for s1 in range(S1):
                        alpha1[i,j,s1]=p_init[s1]*p_out[i,j,s1]*100
                    alpha1[i,j,:]/=alpha1[i,j,:].sum()
                    

            else:
                for i in range(0,H):
                    for s1_2 in range(S1):
                        for s1_1 in range(S1):
                            alpha1[i,j,s1_2]+=(alpha1[i,j-1,s1_1]*100)*p_out[i,j,s1_2]*p_trans[s1_1,s1_2]
                    A=alpha1[i,j,:].sum()
                    alpha1[i,j,:]/=A
                    
                    

    #forward2(縦方向)
    def forward2(self,p_init,p_trans,p_out,alpha2):
        for i in range (0,H):
            if(i==0):
                for j in range(0,W):
                    for s2 in range(S2):
                        alpha2[i,j,s2]=p_init[s2]*p_out[i,j,s2]*100
                    alpha2[i,j,:]/=alpha2[i,j,:].sum()
                    
            else:
                for j in range(0,W):
                    for s2_2 in range(S2):
                        for s2_1 in range(S2):
                            alpha2[i,j,s2_2]+=(alpha2[i-1,j,s2_1]*100)*p_out[i,j,s2_2]*p_trans[s2_1,s2_2]
                    alpha2[i,j,:]=alpha2[i,j,:]/alpha2[i,j,:].sum()

    #backward1(横方向)
    def backward1(self,p_trans,p_out,beta1):
        for j in range (W-1,-1,-1):
            if (j==W-1):
                for i in range(0,H):
                    beta1[i,j,:]=100
            else:
                for i in range(0,H):
                    for s1_2 in range(S1):
                        for s1_1 in range(S1):
                            beta1[i,j,s1_2]+=(beta1[i,j+1,s1_1]*10000)*p_out[i,j+1,s1_1]*p_trans[s1_1,s1_2]
                    B=beta1[i,j,:].sum()
                    beta1[i,j,:]=beta1[i,j,:]/B
    #backward2(縦方向)
    def backward2(self,p_trans,p_out,beta2):
        for i in range(H-1,-1,-1):
            if(i==H-1):
                for j in range(0,W):
                    beta2[i,j,:]=100
            else:
                for j in range(0,W):
                    for s2_2 in range(S2):
                        for s2_1 in range(S2):
                            beta2[i,j,s2_2]+=(beta2[i+1,j,s2_1]*10000)*p_out[i+1,j,s2_1]*p_trans[s2_1,s2_2]
                    beta2[i,j,:]=beta2[i,j,:]/beta2[i,j,:].sum()

                
    #q(S1)
    def update_q_s1(self):
        p_init=np.exp(self.LOG_PI1)
        p_trans=np.exp(self.LOG_A1)
        p_out=np.exp(self.LOG_N1)
        self.forward1(p_init,p_trans,p_out,self.alpha_s1)
        self.backward1(p_trans,p_out,self.beta_s1)
        self.Q_S1=self.alpha_s1*self.beta_s1
        self.Q_S1[self.Q_S1<epsilon]=epsilon
        self.Q_S1/=self.Q_S1.sum(axis=2,keepdims=True)
        for i in range(0,H):
            for j in range(1,W):
                for s1_1 in range(S1):
                    for s1_2 in range(S1):
                        self.Q_SS1[i,j,s1_1,s1_2]=self.alpha_s1[i,j-1,s1_1]*p_trans[s1_1,s1_2]*p_out[i,j,s1_2]*self.beta_s1[i,j,s1_2]
        self.Q_SS1[self.Q_SS1<epsilon]=epsilon
        self.Q_SS1/=self.Q_SS1.sum(axis=(2,3),keepdims=True)
        
    #q(S2)
    def update_q_s2(self):
        p_init=np.exp(self.LOG_PI2)
        p_trans=np.exp(self.LOG_A2)
        p_out=np.exp(self.LOG_N2)
        self.forward2(p_init,p_trans,p_out,self.alpha_s2)
        self.backward2(p_trans,p_out,self.beta_s2)
        self.Q_S2=self.alpha_s2*self.beta_s2
        self.Q_S2[self.Q_S2<epsilon]=epsilon
        self.Q_S2/=self.Q_S2.sum(axis=2,keepdims=True)
        for j in range(0,W):
            for i in range(1,H):
                for s2_1 in range(S2):
                    for s2_2 in range(S2):
                        self.Q_SS2[i,j,s2_1,s2_2]=self.alpha_s2[i-1,j,s2_1]*p_trans[s2_1,s2_2]*p_out[i,j,s2_2]*self.beta_s2[i,j,s2_2]
        self.Q_SS2[self.Q_SS2<epsilon]=epsilon
        self.Q_SS2/=self.Q_SS2.sum(axis=(2,3),keepdims=True)
        
##############################################
#main
###############################################
#image=np.array(Image.open("image.png"),np.int)
model=Extended_HMMs(image)
#image_X=np.array(Image.open("tower.png"),np.int)
#MSE=np.zeros((I,H,W))
#変分ベイズ
for k in range(I):
    model.update_e_log()
    #print("e_log")
    model.update_q_s1()
    #print(model.Q_S1)
    model.update_q_s2()
    #print("q_s2")
    model.update_q_theta()
    #print("q_theta")
    #for i in range(I_Hs,I_Hf+1):
        #for j in range(I_Ws,I_Wf+1):
            #a=model.Q_S1[i,j,0]*model.Q_S2[i,j,0]
            #b=model.Q_S1[i,j,0]*model.Q_S2[i,j,1]
            #c=model.Q_S1[i,j,1]*model.Q_S2[i,j,0]
            #d=model.Q_S1[i,j,1]*model.Q_S2[i,j,1]
            #if(max(a,b,c,d)==a):
                #MSE[k,i,j]=np.dot((np.round(model.M[0,0,:,:].T)-image_X[i,j,:]),(np.round(model.M[0,0,:,:].T)-image_X[i,j,:]).T)
            #elif(max(a,b,c,d)==b):
                #MSE[k,i,j]=np.dot((np.round(model.M[0,1,:,:].T)-image_X[i,j,:]),(np.round(model.M[0,1,:,:].T)-image_X[i,j,:]).T)
            #elif(max(a,b,c,d)==c):
                #MSE[k,i,j]=np.dot((np.round(model.M[1,0,:,:].T)-image_X[i,j,:]),(np.round(model.M[1,0,:,:].T)-image_X[i,j,:]).T)
            #else:
                #MSE[k,i,j]=np.dot((np.round(model.M[1,1,:,:].T)-image_X[i,j,:]),(np.round(model.M[1,1,:,:].T)-image_X[i,j,:]).T)
    print(k)