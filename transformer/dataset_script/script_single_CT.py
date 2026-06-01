# script_single_CT.py — 协同转弯 (CT) 单目标轨迹生成，全程在 WGS84 ECEF 坐标系建模
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

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

from ecef_utils import (
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
        R_e2n=ecef_to_enu_matrix(lat,lon); R_n2e=R_e2n.T
        vu=R_e2n@self.vel; vE,vN=vu[0],vu[1]; sh=np.hypot(vE,vN)
        ld=np.array([-vN/sh,vE/sh,0.]) if sh>1e-6 else np.array([1.,0.,0.])
        al=3.*np.sin(2*np.pi*self.time/20.+self.phase)
        av=1.*np.sin(2*np.pi*self.time/15.+self.phase*1.5)
        ae=al*ld+np.array([0.,0.,av])
        cs=np.linalg.norm(self.vel); fe=R_e2n@(self.vel/max(cs,1e-6))
        ae+=0.2*(self.base_speed-cs)*fe
        ac=R_n2e@ae; self.vel+=ac*self.dt; self.pos+=self.vel*self.dt
        return self.pos.copy(), self.vel.copy(), ac.copy()


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


# ──────────────── CT 目标运动模型 ────────────────────────────────────
def generate_ct_target_ecef(pos0, vel0, steps, dt):
    """
    协同转弯 (CT)：在当地 ENU 水平面内匀速转弯，Z 速度保持不变。
    状态积分在 ECEF 中完成。
    """
    pos = np.array(pos0, float); vel = np.array(vel0, float)
    omega = np.random.uniform(-0.15, 0.15)
    if abs(omega) < 0.03:
        omega = 0.03 * (1 if omega >= 0 else -1)

    states = []; t = START_TIME
    for _ in range(steps):
        lat, lon, _ = ecef_to_geodetic(*pos)
        R_e2n = ecef_to_enu_matrix(lat, lon)
        R_n2e = R_e2n.T
        vel_enu = R_e2n @ vel
        vE, vN, vU = vel_enu

        # CT 精确解（水平面）
        sin_o = np.sin(omega * dt); cos_o = np.cos(omega * dt)
        vE_new =  vE * cos_o - vN * sin_o
        vN_new =  vE * sin_o + vN * cos_o
        pE_new =  vE * sin_o/omega - vN*(1-cos_o)/omega   if abs(omega)>1e-8 else vE*dt
        pN_new =  vE*(1-cos_o)/omega + vN*sin_o/omega     if abs(omega)>1e-8 else vN*dt
        pU_new = vU * dt

        # ENU 位移/速度增量
        dpos_enu = np.array([pE_new, pN_new, pU_new])
        vel_enu_new = np.array([vE_new, vN_new, vU])

        # 转回 ECEF
        pos = pos + R_n2e @ dpos_enu
        vel = R_n2e @ vel_enu_new

        acc_enu = np.array([-omega*vN, omega*vE, 0.0])
        acc_ecef = R_n2e @ acc_enu

        sv = np.array([pos[0],vel[0],pos[1],vel[1],pos[2],vel[2]])
        st = GroundTruthState(sv, timestamp=t); st.true_acc = acc_ecef
        states.append(st); t += timedelta(seconds=dt)
    return states


def run_simulation(master_lat, master_lon, master_alt_m,
                   master_heading_deg, master_speed_mps,
                   target_pos_ecef, target_vel_ecef, seed=None):
    if seed is not None: np.random.seed(seed)
    steps = int(DURATION/DT)
    truth = generate_ct_target_ecef(target_pos_ecef, target_vel_ecef, steps, DT)

    R_n2e = enu_to_ecef_matrix(master_lat, master_lon)
    h = np.radians(master_heading_deg)
    mv = R_n2e @ np.array([master_speed_mps*np.sin(h), master_speed_mps*np.cos(h), 0.])
    mp = geodetic_to_ecef(master_lat, master_lon, master_alt_m)
    wm_pos = [mp+R_n2e@off for off in [np.array([0.,3000.,0.]),np.array([0.,-3000.,0.]),
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
        num_traj=500; save_dir="train_data_ct"
        os.makedirs(save_dir,exist_ok=True)
        print(f"🚀 开始生成 {num_traj} 条协同转弯 (CT) 单目标轨迹 [ECEF 地心坐标系]...")
        count=0; attempts=0
        while count<num_traj:
            attempts+=1
            data=run_simulation(*_random_scenario())
            if data is None: continue
            scipy.io.savemat(os.path.join(save_dir,f"track_data_ct_{count:04d}.mat"),data)
            count+=1
            if count%50==0: print(f"  ✅ {count}/{num_traj}  (尝试 {attempts})")
        print(f"🎉 完成！数据保存于 {save_dir}/")
    except Exception: traceback.print_exc()
