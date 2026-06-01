# script_single_CSSW.py — CS-SW 复杂机动单目标轨迹生成，全程在 WGS84 ECEF 坐标系建模
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import datetime
from datetime import timedelta
import traceback
import collections
import numpy as np
import scipy.io

from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.nonlinear import CartesianToElevationBearingRange
from stonesoup.base import Property
from stonesoup.sensor.sensor import SimpleSensor

from src.utils.ecef_utils import (
    geodetic_to_ecef, ecef_to_geodetic,
    ecef_to_enu_matrix, enu_to_ecef_matrix,
    get_body_frame_axes, reconstruct_ecef_from_body_angles,
)

START_TIME = datetime.datetime.now()
DURATION   = 10.0
DT         = 0.1

SENSOR_PERIODS = {
    0: timedelta(seconds=0.1),
    1: timedelta(seconds=0.1),
    2: timedelta(seconds=0.2),
}

# ─────────── CSSW 机动模式（ENU 坐标系定义加速度） ────────────────────
# 加速度分量：[东向/前向, 北向/侧向, 垂向]
# 实际施加时会在当地 ENU 系中依据当前速度方向重投影
MODES = [
    {"a_bar": [ 0.0,  0.0,  0.0], "omega0": 0.0},   # 0: 平稳
    {"a_bar": [-15.,  20.,  0.0], "omega0": 1.0},    # 1: 左急转
    {"a_bar": [ 15., -20.,  0.0], "omega0": 1.0},    # 2: 右急转
    {"a_bar": [ 0.0,  0.0, -20.], "omega0": 0.5},    # 3: 俯冲
    {"a_bar": [ 0.0,  0.0,  25.], "omega0": 0.5},    # 4: 拉起
    {"a_bar": [ 25.,  25.,  0.0], "omega0": 4.0},    # 5: 蛇形
]

T_MATRIX = np.zeros((6, 6))
for _i in range(1, 6):
    T_MATRIX[_i][_i] = 1.0
T_MATRIX[0] = [0.96, 0.012, 0.012, 0.006, 0.006, 0.004]


class DatalinkNode:
    def __init__(self, delay_s=0.1, interval_s=0.1, noise_r_m=100.0):
        self.delay_s=delay_s; self.interval_s=interval_s; self.noise_r=noise_r_m
        self.queue=collections.deque(); self.last_tx=-999.0
    def ingest(self, t_s, xyz):
        if t_s-self.last_tx>=self.interval_s-1e-5:
            self.queue.append({"arrival":t_s+self.delay_s,"age":self.delay_s,
                               "xyz":np.array(xyz,float)+np.random.randn(3)*self.noise_r})
            self.last_tx=t_s
    def pop(self, t_s):
        latest=None
        while self.queue and self.queue[0]["arrival"]<=t_s+1e-5: latest=self.queue.popleft()
        return latest


class ECEFPlatform:
    def __init__(self, pos_ecef, vel_ecef, dt=0.1, phase=0.0):
        self.pos=np.array(pos_ecef,float); self.vel=np.array(vel_ecef,float)
        self.base_speed=np.linalg.norm(vel_ecef); self.dt=dt; self.time=0.; self.phase=phase
    def step(self):
        self.time+=self.dt
        lat,lon,_=ecef_to_geodetic(*self.pos)
        Re=ecef_to_enu_matrix(lat,lon); Rn=Re.T
        vu=Re@self.vel; vE,vN=vu[0],vu[1]; sh=np.hypot(vE,vN)
        ld=np.array([-vN/sh,vE/sh,0.]) if sh>1e-6 else np.array([1.,0.,0.])
        al=3.*np.sin(2*np.pi*self.time/20.+self.phase)
        av=1.*np.sin(2*np.pi*self.time/15.+self.phase*1.5)
        ae=al*ld+np.array([0.,0.,av])
        cs=np.linalg.norm(self.vel); fe=Re@(self.vel/max(cs,1e-6))
        ae+=0.2*(self.base_speed-cs)*fe
        ac=Rn@ae; self.vel+=ac*self.dt; self.pos+=self.vel*self.dt
        return self.pos.copy(),self.vel.copy(),ac.copy()


class ECEFBodySensor(SimpleSensor):
    measurement_model=Property("mm"); position_mapping=Property("pm",default=(0,2,4))
    az_fov=Property("az",default=None); el_fov=Property("el",default=None)
    max_range=Property("mr",default=None)
    def _bv(self,truth,ps):
        st=truth.state_vector.flatten(); sp=ps.state_vector.flatten()
        rel=np.array([st[0]-sp[0],st[2]-sp[2],st[4]-sp[4]])
        f,r,u=get_body_frame_axes(np.array([sp[1],sp[3],sp[5]]),np.array([sp[0],sp[2],sp[4]]))
        return np.dot(rel,f),np.dot(rel,r),np.dot(rel,u)
    def measure(self,gts,noise=True,platform_state=None,**kw):
        dets=[]
        for truth in gts:
            if not self.is_detectable(truth,platform_state): continue
            dx,dy,dz=self._bv(truth,platform_state)
            rs=truth.state_vector.copy()
            rs[self.position_mapping[0]]=dx; rs[self.position_mapping[1]]=dy; rs[self.position_mapping[2]]=dz
            mv=self.measurement_model.function(GroundTruthState(rs,timestamp=truth.timestamp),noise=noise,**kw)
            dets.append(TrueDetection(mv,measurement_model=self.measurement_model,
                                      timestamp=truth.timestamp,groundtruth_path=truth))
        return dets
    def is_detectable(self,truth,ps):
        dx,dy,dz=self._bv(truth,ps); r=np.sqrt(dx**2+dy**2+dz**2)
        if r<1e-6: return False
        az=np.arctan2(dy,dx); el=np.arcsin(np.clip(dz/r,-1.,1.))
        if self.az_fov and not(self.az_fov[0]<=az<=self.az_fov[1]): return False
        if self.el_fov and not(self.el_fov[0]<=el<=self.el_fov[1]): return False
        if self.max_range and r>self.max_range: return False
        return True
    def is_clutter_detectable(self,c): return True


def _ro(sv):
    a=np.asarray(sv).flatten(); return float(a[1]),float(a[0]),float(a[2])

def _make_sensors():
    R=ECEFBodySensor(measurement_model=CartesianToElevationBearingRange(ndim_state=6,mapping=(0,2,4),
        noise_covar=np.diag([4e-6,4e-6,6400.])),az_fov=(-np.pi/3,np.pi/3),
        el_fov=(-np.radians(3),np.radians(3)),max_range=200_000.)
    E=ECEFBodySensor(measurement_model=CartesianToElevationBearingRange(ndim_state=6,mapping=(0,2,4),
        noise_covar=np.diag([4e-6,4e-6,1e20])),az_fov=(-np.pi/3,np.pi/3),
        el_fov=(-np.radians(3),np.radians(3)),max_range=200_000.)
    S=ECEFBodySensor(measurement_model=CartesianToElevationBearingRange(ndim_state=6,mapping=(0,2,4),
        noise_covar=np.diag([2.5e-5,2.5e-5,1e20])),az_fov=(-np.radians(60),np.radians(60)),
        el_fov=(-np.radians(3),np.radians(3)),max_range=200_000.)
    return R,E,S


# ──────────── CS-SW 加速度模型（在 ENU 系中运作） ────────────────────
class CSSWAccelModel:
    """二阶随机过程 (CS-SW)，在目标当地 ENU 坐标系中产生加速度"""
    def __init__(self, a_bar, omega0=1.2, alpha=0.3, sigma=5.0, dt=0.1):
        self.a_bar=np.array(a_bar,float); self.omega0=omega0; self.alpha=alpha
        self.sigma=sigma; self.dt=dt; self.z1=np.zeros(3); self.z2=np.zeros(3)
    def set(self, a_bar, omega0):
        self.a_bar=np.array(a_bar,float); self.omega0=omega0
    def step(self):
        w=np.random.randn(3)*self.sigma
        dz2=(-2*self.alpha*self.z2-(self.alpha**2+self.omega0**2)*self.z1+w)
        self.z2+=dz2*self.dt; self.z1+=self.z2*self.dt
        self.z1=np.clip(self.z1,-80.,80.)
        return np.clip(self.a_bar+self.z1,-150.,150.)


# ──────────────── CSSW 目标运动模型 ─────────────────────────────────
def generate_cssw_target_ecef(pos0, vel0, steps, dt):
    """
    CS-SW 复合机动目标。
    加速度在当地 ENU 系中生成（机动参数面向 ENU 轴），再转入 ECEF 积分。
    """
    pos=np.array(pos0,float); vel=np.array(vel0,float)
    acc_model=CSSWAccelModel(MODES[0]["a_bar"],dt=dt)
    acc_model.sigma=5.0

    states=[]; t=START_TIME; mode_idx=0; burst=0.; maneuvered=False

    for step_i in range(steps):
        # 模式转换
        if burst>0:
            burst-=dt
            if burst<=0: mode_idx=0
        else:
            probs=T_MATRIX[mode_idx]
            nxt=np.random.choice(6,p=probs)
            if not maneuvered and step_i>=30 and mode_idx==0:
                nxt=np.random.randint(1,6)
            if nxt!=mode_idx:
                mode_idx=nxt; maneuvered=True
                if mode_idx in [1,2]: burst=np.random.uniform(3.,6.)
                elif mode_idx in [3,4]: burst=np.random.uniform(2.,3.5)
                elif mode_idx==5: burst=np.random.uniform(4.,7.)

        if mode_idx==0:
            # 高度阻尼（在 ENU 中 vU 分量）
            lat,lon,_=ecef_to_geodetic(*pos)
            Re=ecef_to_enu_matrix(lat,lon)
            vU=(Re@vel)[2]
            damp=float(np.clip(-0.5*vU,-10.,10.))
            acc_model.set([0.,0.,damp],0.)
        else:
            acc_model.set(**MODES[mode_idx])

        # ENU 加速度
        acc_enu=acc_model.step()

        # 转回 ECEF
        lat,lon,_=ecef_to_geodetic(*pos)
        Re=ecef_to_enu_matrix(lat,lon); Rn=Re.T
        acc_ecef=Rn@acc_enu

        vel=vel+acc_ecef*dt
        pos=pos+vel*dt+0.5*acc_ecef*dt**2

        sv=np.array([pos[0],vel[0],pos[1],vel[1],pos[2],vel[2]])
        st=GroundTruthState(sv,timestamp=t); st.true_acc=acc_ecef
        states.append(st); t+=timedelta(seconds=dt)
    return states


def run_simulation(master_lat, master_lon, master_alt_m,
                   master_heading_deg, master_speed_mps,
                   target_pos_ecef, target_vel_ecef, seed=None):
    if seed is not None: np.random.seed(seed)
    steps=int(DURATION/DT)
    truth=generate_cssw_target_ecef(target_pos_ecef,target_vel_ecef,steps,DT)

    R_n2e=enu_to_ecef_matrix(master_lat,master_lon)
    h=np.radians(master_heading_deg)
    mv=R_n2e@np.array([master_speed_mps*np.sin(h),master_speed_mps*np.cos(h),0.])
    mp=geodetic_to_ecef(master_lat,master_lon,master_alt_m)
    wm_pos=[mp+R_n2e@off for off in [np.array([0.,3000.,0.]),np.array([0.,-3000.,0.]),
                                      np.array([3000.,3000.,0.]),np.array([3000.,-3000.,0.])]]
    cfgs=["all","radar_only","radar_eo","radar_esm","radar_only"]
    platforms=[]
    for i,cfg in enumerate(cfgs):
        p0=mp if i==0 else wm_pos[i-1]
        plat={"motion":ECEFPlatform(p0,mv.copy(),dt=DT,phase=float(i)),
              "is_master":i==0,"datalink":None if i==0 else DatalinkNode(),"cached_dl":None}
        r,e,s=_make_sensors()
        plat["sensors"]=({"all":[r,e,s],"radar_only":[r,None,None],
                           "radar_esm":[r,None,s],"radar_eo":[r,e,None]})[cfg]
        plat["last_meas"]={idx:START_TIME-p for idx,p in SENSOR_PERIODS.items()}
        platforms.append(plat)

    T=len(truth); Data_Input=np.empty((T,),dtype=object)
    Data_Label=np.zeros((T,6),dtype=np.float64); ci=0

    for t,state in enumerate(truth):
        t_s=float((state.timestamp-START_TIME).total_seconds())
        sv=state.state_vector.flatten(); Data_Label[t]=sv[[0,1,2,3,4,5]]
        uav_list=np.empty((len(platforms),),dtype=object); sv_ok=False

        for u,plat in enumerate(platforms):
            p_pos,p_vel,p_acc=plat["motion"].step()
            psv=np.array([p_pos[0],p_vel[0],p_pos[1],p_vel[1],p_pos[2],p_vel[2]])
            ps=GroundTruthState(psv,timestamp=state.timestamp)
            so={};re=None
            for idx,sensor in enumerate(plat["sensors"]):
                if sensor is None: continue
                if state.timestamp-plat["last_meas"][idx]>=SENSOR_PERIODS[idx]-timedelta(microseconds=10):
                    dets=sensor.measure([state],platform_state=ps)
                    obs=[]
                    for d in dets:
                        if isinstance(d,TrueDetection):
                            az,el,r=_ro(d.state_vector); obs.append({"meas":[az,el,r]})
                            if idx==0 and re is None:
                                re=reconstruct_ecef_from_body_angles(az,el,r,p_pos,p_vel)
                    if obs and np.random.rand()>=0.15: so[f"sensor_{idx}"]=obs
                    plat["last_meas"][idx]=state.timestamp
            if plat["is_master"]:
                dl,da=None,0.; sens=so
            else:
                if re is not None: plat["datalink"].ingest(t_s,re)
                arr=plat["datalink"].pop(t_s)
                if arr is not None: plat["cached_dl"]=arr
                dl=plat["cached_dl"]["xyz"] if plat["cached_dl"] else None
                da=plat["cached_dl"]["age"] if plat["cached_dl"] else -1.; sens={}
            def _g(k,d):
                o=sens.get(k,[]); return (o[0]["meas"][:d],1.) if o else ([0.]*d,0.)
            rm,rmk=_g("sensor_0",3); em,emk=_g("sensor_1",2); sm,smk=_g("sensor_2",2)
            if so: sv_ok=True
            lp,lop,_=ecef_to_geodetic(*p_pos); vu=ecef_to_enu_matrix(lp,lop)@p_vel
            yaw=np.arctan2(vu[0],vu[1]); pit=np.arctan2(vu[2],np.hypot(vu[0],vu[1]))
            ud={"Pose":{"PositionWorld":p_pos.tolist(),"EulerRPY":[0.,float(pit),float(yaw)],
                        "QuaternionWXYZ":[1.,0.,0.,0.]},
                "IMU":{"Vector":[0.,0.,0.,float(p_acc[0]),float(p_acc[1]),float(p_acc[2])]},
                "FastLink":{"Meas":dl.tolist() if dl is not None else [0.,0.,0.],
                            "Age":da,"IsUpdate":1. if dl is not None else 0.}}
            if plat["is_master"]:
                ud["Radar"]={"Meas":rm,"Mask":rmk}; ud["EO"]={"Meas":em,"Mask":emk}
                ud["ESM"]={"Meas":sm,"Mask":smk,"ModeCode":0.}
                ud["SlowLink"]={"Meas":[0.,0.,0.],"Age":0.,"IsUpdate":0.}
            uav_list[u]=ud
        Data_Input[t]={"GlobalTime":t_s,"UAV":uav_list}
        ci=0 if sv_ok else ci+1
        if ci>4: return None
    if ci>0: return None
    return {"Data_Input":Data_Input,"Data_Label":Data_Label,"T":DT}


def _random_scenario():
    lat=np.random.uniform(30.,50.); lon=np.random.uniform(100.,130.)
    alt=np.random.uniform(5000.,8000.); hdg=np.random.uniform(0.,360.); spd=np.random.uniform(220.,300.)
    mp=geodetic_to_ecef(lat,lon,alt); R=enu_to_ecef_matrix(lat,lon); h=np.radians(hdg)
    mv=R@np.array([spd*np.sin(h),spd*np.cos(h),0.])
    fwd,rgt,up=get_body_frame_axes(mv,mp)
    rng=np.random.uniform(8000.,25000.); az=np.random.uniform(-np.radians(45),np.radians(45))
    el=np.random.uniform(-np.radians(2),np.radians(2))
    tp=mp+rng*np.cos(el)*np.cos(az)*fwd+rng*np.cos(el)*np.sin(az)*rgt+rng*np.sin(el)*up
    tl,tlo,_=ecef_to_geodetic(*tp); th=np.radians(hdg+np.random.uniform(-108.,108.))
    ts=np.random.uniform(200.,350.)
    tv=enu_to_ecef_matrix(tl,tlo)@np.array([ts*np.sin(th),ts*np.cos(th),np.random.uniform(-30.,30.)])
    return lat,lon,alt,hdg,spd,tp,tv


if __name__ == "__main__":
    try:
        num_traj=800; save_dir="data/train_data/train_data_cssw"
        os.makedirs(save_dir,exist_ok=True)
        print(f"🚀 开始生成 {num_traj} 条 CS-SW 复杂机动单目标轨迹 [ECEF 地心坐标系]...")
        count=0; attempts=0
        while count<num_traj:
            attempts+=1
            data=run_simulation(*_random_scenario())
            if data is None: continue
            scipy.io.savemat(os.path.join(save_dir,f"track_data_cssw_{count:04d}.mat"),data)
            count+=1
            if count%50==0: print(f"  ✅ {count}/{num_traj}  (尝试 {attempts})")
        print(f"🎉 完成！数据保存于 {save_dir}/")
    except Exception: traceback.print_exc()
