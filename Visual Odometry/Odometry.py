from glob import glob
import cv2, skimage, os
import numpy as np
import math

class OdometryClass:
    def __init__(self, frame_path):
        self.frame_path = frame_path
        self.frames = sorted(glob(os.path.join(self.frame_path, 'images', '*')))
        with open(os.path.join(frame_path, 'calib.txt'), 'r') as f:
            lines = f.readlines()
            self.focal_length = float(lines[0].strip().split()[-1])
            lines[1] = lines[1].strip().split()
            self.pp = (float(lines[1][1]), float(lines[1][2]))
            
        with open(os.path.join(self.frame_path, 'gt_sequence.txt'), 'r') as f:
            self.pose = [line.strip().split() for line in f.readlines()]
        
        """MY CHANGES START"""
        
        self.K = np.array([[self.focal_length,0,self.pp[0]],[0,self.focal_length,self.pp[1]],[0,0,1]])
        
        self.K_inv = np.linalg.inv(self.K)
        
        self.frame1_pose = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        
        self.D = np.array([[0,1,0],[-1,0,0],[0,0,1]])
                
        self.fast = cv2.FastFeatureDetector_create(threshold=10,nonmaxSuppression=True) #10 #25
        

        """MY CHANGES END"""
        
    def imread(self, fname):
        """
        read image into np array from file
        """
        return cv2.imread(fname, 0)

    def imshow(self, img):
        """
        show image
        """
        skimage.io.imshow(img)

    def get_gt(self, frame_id):
        pose = self.pose[frame_id]
        x, y, z = float(pose[3]), float(pose[7]), float(pose[11])
        return np.array([[x], [y], [z]])

    def get_scale(self, frame_id):
        '''Provides scale estimation for mutliplying
        translation vectors
        
        Returns:
        float -- Scalar value allowing for scale estimation
        '''
        prev_coords = self.get_gt(frame_id - 1)
        curr_coords = self.get_gt(frame_id)
        return np.linalg.norm(curr_coords - prev_coords)
    
    """-------------------------------------------------MY FUNCTIONS START-------------------------------------------------"""
    
    def get_matches_FAST_optical_flow(self,img1,img2):
   
        p_fast = self.fast.detect(img1)
        
        p0 = []
        for i in p_fast:
            p0.append(i.pt)
        
        p0 = np.array(p0,dtype=np.float32).reshape(-1,1,2)
                
        p1, st, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None) 
     
        points_1 = p0[st==1]
        points_2 = p1[st==1]
        
        return np.array(points_1),np.array(points_2)
    
    def get_norm_transform (self,pts):
        
        t_x = (1.0/pts.shape[0])*np.sum(pts[:,0])
        t_y = (1.0/pts.shape[0])*np.sum(pts[:,1])
      
        mc = pts-np.array([t_x,t_y])
        
        dc = math.sqrt((1.0/pts.shape[0])*np.sum(np.square(mc))) 
              
        s = math.sqrt(2)/dc
        
        transform = np.array([[s,0,-s*t_x],[0,s,-s*t_y],[0,0,1]])
        
        return transform
    
    def get_tranformed_pts (self,T,pts):
        
        pts_2 = np.hstack((pts,np.ones((pts.shape[0],1)))).T 
        
        pts_3 = np.dot(T,pts_2).T 
        
        pts_4 = np.array([pts_3[:,0]/pts_3[:,2],pts_3[:,1]/pts_3[:,2]]).T
        
        return pts_4
        
    
    def get_F_normalized (self,pts1,pts2):
        
        T_a = self.get_norm_transform (pts1)
        T_b = self.get_norm_transform (pts2)
                
        pts1 = self.get_tranformed_pts (T_a,pts1)
        pts2 = self.get_tranformed_pts (T_b,pts2)
        
        A = []
        
        for i in range(pts1.shape[0]):
            
            x = pts1[i][0]
            x_2 = pts2[i][0]
            y = pts1[i][1]
            y_2 = pts2[i][1]
            
            A.append([x_2*x, x_2*y, x_2, y_2*x, y_2*y, y_2, x, y, 1.0])
                
        A = np.array(A)
        
        _,_,V = np.linalg.svd(A)
        
        F = np.reshape(V[-1,:],(3,3))
        
        U,S,V = np.linalg.svd(F)
        
        S[-1]=0
        
        F = U @ np.diag(S) @ V
        
        F = F/F[-1,-1] 
        
        F_final = np.transpose(T_b)@F@T_a
        F_final = F_final/F_final[-1,-1]
        
        return F_final
    
    def get_E (self,pts1,pts2):
         
        
        E,_ = cv2.findEssentialMat(pts1, pts2,self.K)
        
        U,S,V = np.linalg.svd(E)
        S = np.diag([1,1,0])
        E = U@S@V
        
        """
        F=self.get_F_normalized(pts1,pts2)
        E = np.transpose(self.K)@F@self.K    
        
        U,S,V = np.linalg.svd(E)
        S = np.diag([S[0],S[1],0])
        
        E = U@S@V
        
        #"""      
        
        return E
    
    def triangulation(self, pt1, proj_1, pt2, proj_2):

        a_pt1 = np.array([[0,-1,pt1[1]],[1,0,-pt1[0]],[-pt1[1],pt1[0],0]])
        a_pt2 = np.array([[0,-1,pt2[1]],[1,0,-pt2[0]],[-pt2[1],pt2[0],0]])

        axb_pt1 = np.dot(a_pt1,proj_1)
        axb_pt2 = np.dot(a_pt2,proj_2)
        
        axb = np.vstack((axb_pt1[0:2],axb_pt2[0:2]))
        
        _,_,V = np.linalg.svd(np.array(axb))

        P = V[-1,:]
        P = P/P[-1]
        
        return P

    def premultiply_K(self,pt):
        pt_homo = np.array([[pt[0]],[pt[1]],[1]])
        pt_K = np.dot(self.K_inv,pt_homo) 
        pts_4 = np.array([pt_K[0]/pt_K[2],pt_K[1]/pt_K[2]])
        return pts_4.reshape(-1,)
    
    def get_R_t (self, E, pts1, pts2):
        
        _,R,t,_ = cv2.recoverPose(E,pts1, pts2,self.K)
        return R,t
  
        """
    
        U,S,V = np.linalg.svd(E)
        
        if(np.linalg.det(U)<0):
            U = -1*U
        if(np.linalg.det(V)<0):
            V = -1*V
        
        #t = U[:,-1].reshape(3,1)
        t_x_a = U @ self.D.T @ np.diag([1,1,0]) @ U.T
        t_x_b = U @ self.D @ np.diag([1,1,0]) @ U.T
        
        t_a = np.array([[t_x_a[2,1]],[t_x_a[0,2]],[t_x_a[1,0]]])
        t_b = np.array([[t_x_b[2,1]],[t_x_b[0,2]],[t_x_b[1,0]]])
        
        R_a = U @ self.D @ V
        R_b = U @ self.D.T @ V
        
        
        H_t = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[-2*V[-1,0],-2*V[-1,1],-2*V[-1,2],-1]]) #[-2*V[0,-1],-2*V[1,-1],-2*V[2,-1],-1] #
        P_A = np.hstack([R_a,t_a.reshape(3,1)])
        P_I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        
        
        pt1_K = self.premultiply_K(pts1[0,:])
        pt2_K = self.premultiply_K(pts2[0,:])
        
        Q = self.triangulation(pt1_K, P_I, pt2_K, P_A)
                
        c1 = Q[2]*Q[3]
        c2 = (P_A@Q)[2] * Q[3]
        
        if(c1>0 and c2>0):
            return R_a,t_a#t
        elif(c1<0 and c2<0):
            return R_a,t_b#-1*t
        elif(c1*c2<0):
            c3 = Q[2]*(H_t@Q)[3]
            if c3>0:
                return R_b,t_a #t
            else:
                return R_b,t_b #-1*t
        else:
            return R_a,t_a
        
        #"""
    
    def update_R_t (self, pose_old, R, t, frame_no):
        
        R_old = pose_old[:3,:3]
        t_old = pose_old[:,3].reshape(3,1)

        t_new = self.get_scale(frame_no)*(R_old @ t) + t_old

        R_new = R_old @ R 

        pose_new = np.hstack([R_new,t_new.reshape(3,1)])

        return pose_new

    
    """-------------------------------------------------MY FUNCTIONS END-------------------------------------------------"""
    
    
    def run(self):
        """
        Uses the video frame to predict the path taken by the camera
        
        The reurned path should be a numpy array of shape nx3
        n is the number of frames in the video  
        """

        prediction = np.zeros((len(self.pose),3))
        
        prediction[0,:] = np.array([self.frame1_pose[0,3],self.frame1_pose[1,3],self.frame1_pose[2,3]])
        
        for i in range(1, len(self.frames)):

            
            img1=self.imread(self.frames[i-1])
            img2=self.imread(self.frames[i])

            
            match_pts1, match_pts2 = self.get_matches_FAST_optical_flow(img1, img2)

            E = self.get_E(match_pts2, match_pts1)

            R,t = self.get_R_t(E,match_pts2, match_pts1)
           
            new_pose = self.update_R_t(self.frame1_pose, R,t, i)
             
            prediction[i,:] = np.array([new_pose[0,3],new_pose[1,3],new_pose[2,3]])
            
            self.frame1_pose = new_pose
            
            print(i,end="\r")

        np.save('predictions.npy',prediction)
        
        return prediction
        

if __name__=="__main__":
    frame_path = 'video_train'
    odemotryc = OdometryClass(frame_path)
    path = odemotryc.run()
    print(path,path.shape)
