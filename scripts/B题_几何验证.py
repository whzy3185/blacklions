# -*- coding: utf-8 -*-
"""2026 CUMCM B题：几何模型独立验证脚本
不连接官方模拟器；只验证建模中的关键几何结论。
依赖: numpy, scipy, shapely
"""
import math
import itertools
import random
import numpy as np
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from shapely.ops import unary_union

DELTA = math.radians(1.0)
R_TARGET = 1800.0
R_MIN = 1000.0

# ---------------- Q1: MEC ----------------
def circle2(a,b):
    return ((a[0]+b[0])/2, (a[1]+b[1])/2, math.dist(a,b)/2)

def circle3(a,b,c, eps=1e-14):
    ax,ay=a; bx,by=b; cx,cy=c
    d=2*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by))
    if abs(d)<eps: return None
    ux=((ax*ax+ay*ay)*(by-cy)+(bx*bx+by*by)*(cy-ay)+(cx*cx+cy*cy)*(ay-by))/d
    uy=((ax*ax+ay*ay)*(cx-bx)+(bx*bx+by*by)*(ax-cx)+(cx*cx+cy*cy)*(bx-ax))/d
    return (ux,uy,math.hypot(ux-ax,uy-ay))

def mec_bruteforce(pts):
    pts=list(pts)
    best=(0,0,float('inf'))
    def cover(c):
        x,y,r=c
        return all((px-x)**2+(py-y)**2 <= (r+1e-9)**2 for px,py in pts)
    for p in pts:
        c=(p[0],p[1],0.0)
        if cover(c) and c[2]<best[2]: best=c
    for a,b in itertools.combinations(pts,2):
        c=circle2(a,b)
        if c[2]<best[2] and cover(c): best=c
    for a,b,c0 in itertools.combinations(pts,3):
        c=circle3(a,b,c0)
        if c and c[2]<best[2] and cover(c): best=c
    return best

def diameter(pts):
    return max(math.dist(a,b) for a,b in itertools.combinations(pts,2))

def test_q1(seed=0, cases=1000):
    rng=np.random.default_rng(seed)
    worst=0.0
    for _ in range(cases):
        n=int(rng.integers(3,10))
        pts=[tuple(x) for x in rng.normal(size=(n,2))]
        D=diameter(pts); R=mec_bruteforce(pts)[2]
        assert D/2-1e-8 <= R <= D/math.sqrt(3)+1e-8
        worst=max(worst,R/D)
    return worst

# ---------------- Q2: local formula ----------------
def d_local(r1,r2,alpha_deg):
    a=math.radians(alpha_deg)
    return 2*math.tan(DELTA)*math.sqrt(
        r1*r1+r2*r2+2*r1*r2*abs(math.cos(a))
    )/abs(math.sin(a))

def wedge_poly(S, theta, L=1e6):
    u1=(math.cos(theta-DELTA),math.sin(theta-DELTA))
    u2=(math.cos(theta+DELTA),math.sin(theta+DELTA))
    return Polygon([S,(S[0]+L*u1[0],S[1]+L*u1[1]),(S[0]+L*u2[0],S[1]+L*u2[1])])

def exact_two_wedge_diameter(r1,r2,alpha_deg):
    a=math.radians(alpha_deg)
    S1=(-r1,0.0)
    S2=(-r2*math.cos(a),-r2*math.sin(a))
    g=wedge_poly(S1,0.0).intersection(wedge_poly(S2,a))
    coords=list(g.exterior.coords)[:-1]
    return diameter(coords)

def q2_minimax_proxy():
    lo,hi=5.0,1500.0
    gs=np.linspace(lo,hi,5001)
    def ymax(x):
        q=min(R_MIN**2-(x-lo)**2, R_MIN**2-(x-hi)**2)
        return math.sqrt(max(0.0,q))
    def objective(z):
        x,frac=z
        y=max(1e-8,frac*ymax(x))
        dx=gs-x
        r2=np.sqrt(dx*dx+y*y)
        sin=np.abs(y)/r2
        cos=dx/r2
        vals=2*math.tan(DELTA)*np.sqrt(gs*gs+r2*r2+2*gs*r2*np.abs(cos))/sin
        return float(vals.max())
    res=differential_evolution(objective,[(500,1005),(0.02,1.0)],seed=42,tol=1e-8,polish=True)
    x,frac=res.x; y=frac*ymax(x)
    M=(lo+hi)/2
    h=math.sqrt(R_MIN**2-((hi-lo)/2)**2)
    tip=objective((M,1.0))
    return (x,y,res.fun),(M,h,tip)

# ---------------- Q3: 7-point cover ----------------
def cover_radius_7(a):
    return max(a/math.sqrt(3), math.sqrt(R_TARGET**2+a*a-math.sqrt(3)*R_TARGET*a))

def ring(radius,n,offset_deg=0.0):
    return [(radius*math.cos(math.radians(offset_deg+360*k/n)),
             radius*math.sin(math.radians(offset_deg+360*k/n))) for k in range(n)]

def mc_q3(n=500000,seed=321):
    Q=np.array([(0.0,0.0)]+ring(1125.0,6,0.0))
    rng=np.random.default_rng(seed)
    r=R_TARGET*np.sqrt(rng.random(n)); th=2*np.pi*rng.random(n)
    P=np.c_[r*np.cos(th),r*np.sin(th)]
    mx=0.0
    for chunk in np.array_split(P,20):
        d=np.sqrt(((chunk[:,None,:]-Q[None,:,:])**2).sum(axis=2)).min(axis=1)
        mx=max(mx,float(d.max()))
    return mx

# ---------------- Q4: angle-gap + 21-point certificate ----------------
def directional_guaranteed(G, Qs, r=1000.0, eps=1e-10):
    gx,gy=G; ang=[]
    for qx,qy in Qs:
        dx,dy=qx-gx,qy-gy
        dd=math.hypot(dx,dy)
        if dd<=eps: return True,0.0
        if dd<=r+eps: ang.append(math.atan2(dy,dx)%(2*math.pi))
    if len(ang)<2: return False,2*math.pi
    ang.sort(); gaps=[ang[i+1]-ang[i] for i in range(len(ang)-1)]
    gaps.append(2*math.pi-ang[-1]+ang[0])
    mg=max(gaps)
    return mg<=math.pi+eps,mg

def regular_polygon(center,radius,n,circumscribed=False):
    cx,cy=center
    rr=radius/math.cos(math.pi/n) if circumscribed else radius
    return Polygon([(cx+rr*math.cos(2*math.pi*k/n), cy+rr*math.sin(2*math.pi*k/n)) for k in range(n)])

def q21_network():
    return [(0.0,0.0)]+ring(1000.0,8,0.0)+ring(1870.0,12,15.0)

def q21_certificate(target_radius=1806.0):
    Q=q21_network()
    recv=[regular_polygon(q,1000.0,256,False) for q in Q]
    cores=[]
    for I in itertools.combinations(range(len(Q)),3):
        tri=Polygon([Q[i] for i in I])
        if tri.area<1e-9: continue
        g=tri
        for i in I: g=g.intersection(recv[i])
        if not g.is_empty: cores.append(g)
    U=unary_union(cores)
    Dplus=regular_polygon((0.0,0.0),target_radius,720,True)
    return Dplus.difference(U).area

def mc_q21(n=200000,seed=123):
    Q=q21_network(); rng=np.random.default_rng(seed)
    r=R_TARGET*np.sqrt(rng.random(n)); th=2*np.pi*rng.random(n)
    maxgap=0.0
    for rr,tt in zip(r,th):
        G=(rr*math.cos(tt),rr*math.sin(tt))
        ok,gap=directional_guaranteed(G,Q)
        assert ok
        maxgap=max(maxgap,gap)
    return math.degrees(maxgap)

if __name__=='__main__':
    print('Q1 worst R/D in random tests:', test_q1())
    for case in [(500,500,90),(1000,1000,90),(1000,800,60),(1500,1000,30),(1500,1000,10)]:
        ex=exact_two_wedge_diameter(*case); ap=d_local(*case)
        print('Q2',case,'exact=',ex,'approx=',ap,'relerr=',(ap-ex)/ex)
    opt,tip=q2_minimax_proxy(); print('Q2 minimax proxy:',opt,'lens tip:',tip)
    print('Q3 analytic cover radius a=1125:',cover_radius_7(1125.0))
    print('Q3 MC max nearest distance:',mc_q3())
    print('Q4 21-point certificate residual area @1806m:',q21_certificate(1806.0))
    print('Q4 21-point certificate residual area @1806.25m:',q21_certificate(1806.25))
    print('Q4 21-point MC max angular gap:',mc_q21())
