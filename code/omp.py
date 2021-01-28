import numpy as np
from itertools import count
from timeit import repeat
'''
the Algorithm of Orthogonal Matching Pursuit (OMP)

References:
[1] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation of the K-SVD Algorithm 
using Batch Orthogonal Matching Pursuit", Technical Report - CS, Technion, April 2008.
[2] Usman, Koredianto, (2017), Introduction to Orthogonal Matching Pursuit, Telkom University 
Online : http://korediantousman.staff.telkomuniversity.ac.id, access : your access time.
'''

### Algorithm 1: OMP-Origin
def omp_1(A,y,k=None):
    '''description:
           reconstruct x from A and y as y=Ax, y is a matrix consists of several columnn vectors 
           but it is inefficient (see Algorithm 3 for improvement) and it is more suitable for 
           single column vector reconstruction. For more details, please read paper[2]
       args:
           A: the sensing matrix or compression matrix
           y: compressed signals
           k: the num of iteration, if None, it will be the num of A's rows, but the loop will 
            always stop only if the residue be zero
       returns:
           x: the original signals
    '''
    
    def f(A,y,k=None):
        h,w=A.shape
        if k==None:
            k=h
        A_hat=A/np.sqrt(np.sum(A*A,0)) #列向量单位化
        residue=y.copy() #残差
        I=[] #选中的基索引
        x_rec=np.zeros((w,1)) #复原出的x
        for i in range(k):
            rest=sorted(set(range(w))-set(I)) #排除选中基后剩下的基索引
            scores=A_hat[:,rest].T.dot(residue) #剩余基对残差的贡献
            I.append(rest[np.argmax(np.abs(scores))]) #有最大贡献的基下标，追加到b
            A_new=A[:,I]
            Lp=np.linalg.inv(A_new.T.dot(A_new)).dot(A_new.T).dot(y) #重新计算A_new中的基对y的贡献值，作为复原x的值
            x_rec[I]=Lp
            residue=y-A_new.dot(Lp) #再次计算残差
            if np.sum(residue*residue)<0.000001: #如果残差为0，结束迭代
                break
        return x_rec
    
    n=y.shape[1]
    ret=np.zeros((A.shape[1],n))
    for i in range(n):
        ret[:,i]=f(A,y[:,i].reshape(-1,1),k).flatten()
    return ret

### Algorithm 2: OMP-Cholesky
def omp_2(D,x,k=None):
    '''description:
           reconstruct y from D and x as x=Dy, moreover we improve Algorithm 1 with an 
           optimization calculation of the sparse approximation (i.e. Lp in Algorithm 1) at 
           each iteration. Note that k hereon represents the sparsity of signal (i.e. the 
           num of 0 in y), if None, the loop will stop until the residue be zero or all basis 
           have been selected. For more details , please read paper[1]
       issues:
           we assume that the colums of D are normalized to unit l²-length, though this 
           restriction may easily be removed, i don't how to do. This problem does not exist 
           in Algorithm 1
    '''
    w=D.shape[1]
    I=[] #选中基索引
    L=np.array([[1]])
    r=x.copy() #残差
    y=np.zeros((w,1)) #复原的信号
    alpha=D.T.dot(x)
    for i in count():
        I_=sorted(set(range(w))-set(I)) #剩余基索引
        k_=I_[np.argmax(np.abs(D[:,I_].T.dot(r)))] #新选中基索引
        if i>0:
            w_=np.linalg.solve(L,D[:,I].T.dot(D[:,[k_]]))
            L=np.hstack((L,np.zeros((L.shape[0],1))))
            L=np.vstack((L,np.hstack((w_.reshape(-1),np.sqrt(1-w_.reshape(-1).dot(w_))))))
        I.append(k_)
        Lp=np.linalg.solve(L.dot(L.T),alpha[I])
        y[I]=Lp
        r=x-D[:,I].dot(Lp)
        if np.sum(r*r)<0.000001 or i==(w-1) or (k!=None and len(y[np.abs(y)<0.000001])>=k):
            break
    return y

### Algorithm 3: Batch-OMP
def omp_3(alpha0,e0,G,err=1e-6,k=None):
    '''description:
           reconstruct y from D and x as x=Dy, but it is suitable for one case that there are many 
           singals to be coded with the same dictionary D, the total amount of work is much smaller 
           than Algorithm 1. This algorithm derives from Algorithm 2 with two changes: 1) at each 
           iteration we don't need to know residue r but only Dᵀr. 2) don't use residue but design 
           a new error-based stopping criterion, in fact it is ||residue||_2^2. For more details , 
           please read paper[1]
       args:
           alpha0: Dᵀx, D is a dictionary with normalized columns
           e0: xᵀx, x is a column signal. Acting as the initial value of error
           G: DᵀD
           err: the target error
           k: the iteration num, maximum is seted as D's atom num
    '''
    w=G.shape[0]
    I=[]
    L=np.array([[1]])
    y=np.zeros((w,1))
    alpha=alpha0
    e=e0
    delta=0
    n=0
    k=k if k!=None and k<w else w
    while n<k:
        I_=sorted(set(range(w))-set(I))
        k_=I_[np.argmax(np.abs(alpha[I_]))]
        if n>0:
            w_=np.linalg.solve(L,G[I][:,[k_]])
            L=np.hstack((L,np.zeros((L.shape[0],1))))
            L=np.vstack((L,np.hstack((w_.reshape(-1),np.sqrt(1-w_.reshape(-1).dot(w_))))))
        I.append(k_)
        Lp=np.linalg.solve(L.dot(L.T),alpha0[I])
        beta=G[:,I].dot(Lp)
        alpha=alpha0-beta
        t=Lp.reshape(-1).dot(beta[I])
        e=e-t+delta
        if e<err:
            break
        delta=t
        y[I]=Lp
        n+=1
    return y

'''
对于函数omp_3，在多数时候，都应避免使用数组拼接操作（np.concatenate()或np.stack()），否则可能会带来极差的体验，修改如下：
def omp_3(alpha0,e0,G,err=1e-6,k=None):
    w=G.shape[0]
    I=[]
    y=np.zeros((w,1))
    alpha=alpha0
    e=e0
    delta=0
    n=0
    k=k if k!=None and k<w else w
    L=np.zeros((w,w)) #
    L[0,0]=1
    while n<k:
        I_=sorted(set(range(w))-set(I))
        k_=I_[np.argmax(np.abs(alpha[I_]))]
        if n>0:
            w_=np.linalg.solve(L[:n,:n],G[I][:,[k_]])
            L[n,:n]=w_.reshape(-1)
            L[n,n]=np.sqrt(1-w_.reshape(-1).dot(w_))
        I.append(k_)
        Lp=np.linalg.solve(L[:n+1,:n+1].dot(L[:n+1,:n+1].T),alpha0[I])
        beta=G[:,I].dot(Lp)
        alpha=alpha0-beta
        t=Lp.reshape(-1).dot(beta[I])
        e=e-t+delta
        if e<err:
            break
        delta=t
        y[I]=Lp
        n+=1
    return y
沿用之前的测试示例，在G4560处理器上，100000次取5次平均用时22秒，未修改前为25秒。尽管这不足为道，因为使用C++版本仅用时1.5秒
'''

if __name__=='__main__':
    A=np.array([[-0.707,0.8,0],[0.707,0.6,-1]])
    y=np.array([1.65,-0.25])
    print(omp_3(A.T.dot(y.reshape(-1,1)),y.dot(y.reshape(-1,1)),A.T.dot(A)))
    print(np.mean(repeat('omp_3(A.T.dot(y.reshape(-1,1)),y.dot(y.reshape(-1,1)),A.T.dot(A))','from __main__ import omp_3,A,y',number=1000)))
