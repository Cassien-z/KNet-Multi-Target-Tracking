# script_single_CA.py — 匀加速 (CA) 单目标轨迹生成，全程在 WGS84 ECEF 坐标系建模
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


class DatalinkNode:
    def __init__(self, delay_s=0.1, interval_s=0.1, noise_r_m=100.0):
        self.delay_s = delay_s; self.interval_s = interval_s
        self.noise_r = noise_r_m
        self.queue = collections.deque(); self.last_tx = -999.0

    def ingest(self, t_s, target_ecef):
        if t_s - self.last_tx >= self.interval_s - 1e-5:
            noisy = np.array(target_ecef, float) + np.random.randn(3) * self.noise_r
            self.queue.append({"arrival": t_s + self.delay_s, "age": self.delay_s, "xyz": noisy})
            self.last_tx = t_s

    def pop(self, t_s):
        latest = None
        while self.queue and self.queue[0]["arrival"] <= t_s + 1e-5:
            latest = self.queue.popleft()
        return latest


class ECEFPlatform:
    def __init__(self, pos_ecef, vel_ecef, dt=0.1, phase=0.0):
        self.pos = np.array(pos_ecef, float)
        self.vel = np.array(vel_ecef, float)
        self.base_speed = np.linalg.norm(vel_ecef)
        self.dt = dt; self.time = 0.0; self.phase = phase

    def step(self):
        self.time += self.dt
        from src.utils.ecef_utils import ecef_to_enu_matrix, ecef_to_geodetic
        lat, lon, _ = ecef_to_geodetic(*self.pos)
        R_e2n = ecef_to_enu_matrix(lat, lon); R_n2e = R_e2n.T
        vel_enu = R_e2n @ self.vel
        vE, vN = vel_enu[0], vel_enu[1]; speed_h = np.hypot(vE, vN)
        lat_dir = np.array([-vN/speed_h, vE/speed_h, 0.0]) if speed_h > 1e-6 else np.array([1.,0.,0.])
        a_lat  = 3.0 * np.sin(2*np.pi*self.time/20.0 + self.phase)
        a_vert = 1.0 * np.sin(2*np.pi*self.time/15.0 + self.phase*1.5)
        acc_enu = a_lat*lat_dir + np.array([0., 0., a_vert])
        cur = np.linalg.norm(self.vel)
        fwd_enu = R_e2n @ (self.vel/max(cur,1e-6))
        acc_enu += 0.2*(self.base_speed - cur)*fwd_enu
        acc_ecef = R_n2e @ acc_enu
        self.vel += acc_ecef * self.dt
        self.pos += self.vel * self.dt
        return self.pos.copy(), self.vel.copy(), acc_ecef.copy()


class ECEFBodySensor(SimpleSensor):
    measurement_model = Property("量测模型")
    position_mapping  = Property("位置映射", default=(0,2,4))
    az_fov = Property("方位视场", default=None)
    el_fov = Property("俯仰视场", default=None)
    max_range = Property("最大距离", default=None)

    def _body_vec(self, truth, ps):
        sv_t = truth.state_vector.flatten(); sv_p = ps.state_vector.flatten()
        pos_t = np.array([sv_t[0],sv_t[2],sv_t[4]])
        pos_p = np.array([sv_p[0],sv_p[2],sv_p[4]])
        vel_p = np.array([sv_p[1],sv_p[3],sv_p[5]])
        rel = pos_t - pos_p
        fwd,rgt,up = get_body_frame_axes(vel_p, pos_p)
        return np.dot(rel,fwd), np.dot(rel,rgt), np.dot(rel,up)

    def measure(self, ground_truths, noise=True, platform_state=None, **kwargs):
        dets = []
        for truth in ground_truths:
            if not self.is_detectable(truth, platform_state): continue
            dx,dy,dz = self._body_vec(truth, platform_state)
            rs = truth.state_vector.copy()
            rs[self.position_mapping[0]]=dx; rs[self.position_mapping[1]]=dy; rs[self.position_mapping[2]]=dz
            rt = GroundTruthState(rs, timestamp=truth.timestamp)
            mv = self.measurement_model.function(rt, noise=noise, **kwargs)
            dets.append(TrueDetection(mv, measurement_model=self.measurement_model,
                                      timestamp=truth.timestamp, groundtruth_path=truth))
        return dets

    def is_detectable(self, truth, ps):
        dx,dy,dz = self._body_vec(truth, ps)
        r = np.sqrt(dx**2+dy**2+dz**2)
        if r < 1e-6: return False
        az = np.arctan2(dy,dx); el = np.arcsin(np.clip(dz/r,-1.,1.))
        if self.az_fov  and not (self.az_fov[0] <=az<=self.az_fov[1]):  return False
        if self.el_fov  and not (self.el_fov[0] <=el<=self.el_fov[1]):  return False
        if self.max_range and r>self.max_range: return False
        return True

    def is_clutter_detectable(self, c): return True


def _reorder(sv):
    a = np.asarray(sv).flatten()
    return float(a[1]), float(a[0]), float(a[2])


def _make_sensors():
    radar = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0,2,4), noise_covar=np.diag([0.002**2,0.002**2,80.**2])),
        az_fov=(-np.pi/3,np.pi/3), el_fov=(-np.radians(3),np.radians(3)), max_range=200_000.)
    eo = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0,2,4), noise_covar=np.diag([0.002**2,0.002**2,1e20])),
        az_fov=(-np.pi/3,np.pi/3), el_fov=(-np.radians(3),np.radians(3)), max_range=200_000.)
    esm = ECEFBodySensor(
        measurement_model=CartesianToElevationBearingRange(
            ndim_state=6, mapping=(0,2,4), noise_covar=np.diag([0.005**2,0.005**2,1e20])),
        az_fov=(-np.radians(60),np.radians(60)), el_fov=(-np.radians(3),np.radians(3)), max_range=200_000.)
    return radar, eo, esm


# ──────────────── CA 目标运动模型 ─────────────────────────────────────
def generate_ca_target_ecef(pos0, vel0, steps, dt):
    """匀加速 (CA)：随机恒定 ECEF 加速度"""
    pos = np.array(pos0, float); vel = np.array(vel0, float)
    acc = np.array([
        np.random.uniform(-15., 15.),
        np.random.uniform(-15., 15.),
        np.random.uniform(-5.,  5.),
    ])
    states = []; t = START_TIME
    for _ in range(steps):
        vel = vel + acc * dt
        pos = pos + vel * dt + 0.5 * acc * dt**2
        sv = np.array([pos[0],vel[0],pos[1],vel[1],pos[2],vel[2]])
        st = GroundTruthState(sv, timestamp=t); st.true_acc = acc.copy()
        states.append(st); t += timedelta(seconds=dt)
    return states


def run_simulation(master_lat, master_lon, master_alt_m,
                   master_heading_deg, master_speed_mps,
                   target_pos_ecef, target_vel_ecef, seed=None):
    if seed is not None: np.random.seed(seed)

    steps = int(DURATION / DT)
    truth = generate_ca_target_ecef(target_pos_ecef, target_vel_ecef, steps, DT)

    R_n2e = enu_to_ecef_matrix(master_lat, master_lon)
    h = np.radians(master_heading_deg)
    master_vel = R_n2e @ np.array([master_speed_mps*np.sin(h), master_speed_mps*np.cos(h), 0.])
    master_pos = geodetic_to_ecef(master_lat, master_lon, master_alt_m)

    wm_offsets = [np.array([0.,3000.,0.]),np.array([0.,-3000.,0.]),
                  np.array([3000.,3000.,0.]),np.array([3000.,-3000.,0.])]
    wm_pos = [master_pos + R_n2e@off for off in wm_offsets]

    cfgs=["all","radar_only","radar_eo","radar_esm","radar_only"]
    platforms = []
    for i, cfg in enumerate(cfgs):
        pos0 = master_pos if i==0 else wm_pos[i-1]
        plat = {"motion": ECEFPlatform(pos0, master_vel.copy(), dt=DT, phase=float(i)),
                "is_master": i==0,
                "datalink": None if i==0 else DatalinkNode(),
                "cached_dl": None}
        r,eo,esm = _make_sensors()
        plat["sensors"] = ({"all":[r,eo,esm],"radar_only":[r,None,None],
                             "radar_esm":[r,None,esm],"radar_eo":[r,eo,None]})[cfg]
        plat["last_meas"] = {idx: START_TIME-p for idx,p in SENSOR_PERIODS.items()}
        platforms.append(plat)

    T = len(truth)
    Data_Input = np.empty((T,), dtype=object)
    Data_Label  = np.zeros((T,6), dtype=np.float64)
    consec_invis = 0

    for t, state in enumerate(truth):
        t_s = float((state.timestamp - START_TIME).total_seconds())
        sv  = state.state_vector.flatten()
        Data_Label[t] = sv[[0,1,2,3,4,5]]

        uav_list = np.empty((len(platforms),), dtype=object)
        step_vis = False

        for u, plat in enumerate(platforms):
            p_pos, p_vel, p_acc = plat["motion"].step()
            plat_sv = np.array([p_pos[0],p_vel[0],p_pos[1],p_vel[1],p_pos[2],p_vel[2]])
            plat_state = GroundTruthState(plat_sv, timestamp=state.timestamp)

            sensor_out = {}; rec_ecef = None
            for idx, sensor in enumerate(plat["sensors"]):
                if sensor is None: continue
                if state.timestamp - plat["last_meas"][idx] >= SENSOR_PERIODS[idx] - timedelta(microseconds=10):
                    dets = sensor.measure([state], platform_state=plat_state)
                    obs = []
                    for d in dets:
                        if isinstance(d, TrueDetection):
                            az,el,r = _reorder(d.state_vector)
                            obs.append({"meas":[az,el,r]})
                            if idx==0 and rec_ecef is None:
                                rec_ecef = reconstruct_ecef_from_body_angles(az,el,r,p_pos,p_vel)
                    if obs and np.random.rand()>=0.15:
                        sensor_out[f"sensor_{idx}"] = obs
                    plat["last_meas"][idx] = state.timestamp

            if plat["is_master"]:
                dl_xyz, dl_age = None, 0.0; sens = sensor_out
            else:
                if rec_ecef is not None: plat["datalink"].ingest(t_s, rec_ecef)
                arr = plat["datalink"].pop(t_s)
                if arr is not None: plat["cached_dl"] = arr
                dl_xyz = plat["cached_dl"]["xyz"] if plat["cached_dl"] else None
                dl_age = plat["cached_dl"]["age"] if plat["cached_dl"] else -1.0
                sens = {}

            def _g(k, d):
                o = sens.get(k,[])
                return (o[0]["meas"][:d], 1.0) if o else ([0.]*d, 0.0)

            rm,rmk = _g("sensor_0",3); em,emk = _g("sensor_1",2); sm,smk = _g("sensor_2",2)
            if sensor_out: step_vis = True

            lat_p,lon_p,_ = ecef_to_geodetic(*p_pos)
            vu = ecef_to_enu_matrix(lat_p,lon_p) @ p_vel
            yaw=np.arctan2(vu[0],vu[1]); pitch=np.arctan2(vu[2],np.hypot(vu[0],vu[1]))

            ud = {"Pose":{"PositionWorld":p_pos.tolist(),"EulerRPY":[0.,float(pitch),float(yaw)],
                          "QuaternionWXYZ":[1.,0.,0.,0.]},
                  "IMU":{"Vector":[0.,0.,0.,float(p_acc[0]),float(p_acc[1]),float(p_acc[2])]},
                  "FastLink":{"Meas":dl_xyz.tolist() if dl_xyz is not None else [0.,0.,0.],
                              "Age":dl_age, "IsUpdate":1.0 if dl_xyz is not None else 0.0}}
            if plat["is_master"]:
                ud["Radar"]={"Meas":rm,"Mask":rmk}; ud["EO"]={"Meas":em,"Mask":emk}
                ud["ESM"]={"Meas":sm,"Mask":smk,"ModeCode":0.0}
                ud["SlowLink"]={"Meas":[0.,0.,0.],"Age":0.,"IsUpdate":0.}
            uav_list[u] = ud

        Data_Input[t] = {"GlobalTime":t_s,"UAV":uav_list}
        consec_invis = 0 if step_vis else consec_invis+1
        if consec_invis > 4: return None

    if consec_invis > 0: return None
    return {"Data_Input":Data_Input,"Data_Label":Data_Label,"T":DT}


def _random_scenario():
    lat=np.random.uniform(30.,50.); lon=np.random.uniform(100.,130.)
    alt=np.random.uniform(5000.,8000.); hdg=np.random.uniform(0.,360.)
    spd=np.random.uniform(220.,300.)
    master_pos = geodetic_to_ecef(lat,lon,alt)
    R_n2e = enu_to_ecef_matrix(lat,lon)
    h=np.radians(hdg)
    master_vel = R_n2e@np.array([spd*np.sin(h),spd*np.cos(h),0.])
    fwd,rgt,up = get_body_frame_axes(master_vel,master_pos)
    rng=np.random.uniform(8000.,25000.)
    az=np.random.uniform(-np.radians(45),np.radians(45))
    el=np.random.uniform(-np.radians(2),np.radians(2))
    tgt_pos = master_pos + rng*np.cos(el)*np.cos(az)*fwd + rng*np.cos(el)*np.sin(az)*rgt + rng*np.sin(el)*up
    tgt_lat,tgt_lon,_ = ecef_to_geodetic(*tgt_pos)
    th=np.radians(hdg+np.random.uniform(-108.,108.)); ts=np.random.uniform(200.,350.)
    tgt_vel = enu_to_ecef_matrix(tgt_lat,tgt_lon)@np.array([ts*np.sin(th),ts*np.cos(th),np.random.uniform(-30.,30.)])
    return lat,lon,alt,hdg,spd,tgt_pos,tgt_vel


if __name__ == "__main__":
    try:
        num_traj=500; save_dir="data/train_data/train_data_ca"
        os.makedirs(save_dir,exist_ok=True)
        print(f"🚀 开始生成 {num_traj} 条匀加速 (CA) 单目标轨迹 [ECEF 地心坐标系]...")
        count=0; attempts=0
        while count<num_traj:
            attempts+=1
            data=run_simulation(*_random_scenario())
            if data is None: continue
            scipy.io.savemat(os.path.join(save_dir,f"track_data_ca_{count:04d}.mat"),data)
            count+=1
            if count%50==0: print(f"  ✅ {count}/{num_traj}  (尝试 {attempts})")
        print(f"🎉 完成！数据保存于 {save_dir}/")
    except Exception: traceback.print_exc()
