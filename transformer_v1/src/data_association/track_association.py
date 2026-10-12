import numpy as np
from scipy.optimize import linear_sum_assignment
from src.utils.ecef_utils import get_body_frame_axes

def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def greedy_assignment(C, gate):
    """
    Greedy assignment for a cost matrix C.
    Returns list of [row, col, cost] for assigned pairs.
    """
    pairs = []
    if C.size == 0:
        return pairs
    
    Cwork = C.copy()
    while True:
        idx = np.argmin(Cwork)
        val = Cwork.flat[idx]
        if not np.isfinite(val) or val > gate:
            break
        r, c = np.unravel_index(idx, Cwork.shape)
        pairs.append([r, c, val])
        Cwork[r, :] = np.inf
        Cwork[:, c] = np.inf
    return pairs

def reject_ambiguous_rows(C, diff_threshold):
    C2 = C.copy()
    for i in range(C.shape[0]):
        vals = C[i, :]
        valid_vals = np.sort(vals[np.isfinite(vals)])
        if len(valid_vals) >= 2:
            if valid_vals[1] - valid_vals[0] < diff_threshold:
                C2[i, :] = np.inf
    return C2


def hungarian_assignment(cost_matrix, gate):
    """Return gated one-to-one assignments as ``(row, col, cost)`` tuples."""
    if cost_matrix.size == 0:
        return []
    finite_rows = np.any(np.isfinite(cost_matrix), axis=1)
    finite_cols = np.any(np.isfinite(cost_matrix), axis=0)
    if not np.any(finite_rows) or not np.any(finite_cols):
        return []

    row_ids = np.flatnonzero(finite_rows)
    col_ids = np.flatnonzero(finite_cols)
    reduced = cost_matrix[np.ix_(row_ids, col_ids)].copy()
    large = max(float(gate) * 1_000.0, 1e6)
    reduced[~np.isfinite(reduced)] = large
    rows, cols = linear_sum_assignment(reduced)
    pairs = []
    for row, col in zip(rows, cols):
        source_row = int(row_ids[row])
        source_col = int(col_ids[col])
        cost = float(cost_matrix[source_row, source_col])
        if np.isfinite(cost) and cost <= gate:
            pairs.append((source_row, source_col, cost))
    return pairs

def local_radar_meas_to_world(r, az, el, pUAV, euler):
    """Convert body-frame radar (range, az, el) to ECEF world coordinates."""
    # Assuming euler is used for body rotation in ecef_utils
    # In script_multitarget.py, we have `fwd, rgt, up` axes. But here we can use `euler` or just the `fwd, rgt, up`.
    # Let's use standard rpy matrix or just rely on velocity vector for axes if euler is not perfectly matching.
    # We will approximate this using `local_rpy2rot` from MATLAB or just `ecef_utils` if possible.
    
    roll, pitch, yaw = euler
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Rwb = Rz @ Ry @ Rx

    xBody = r * np.cos(el) * np.cos(az)
    yBody = r * np.cos(el) * np.sin(az)
    zBody = r * np.sin(el)
    pBody = np.array([xBody, yBody, zBody])
    
    return pUAV + Rwb @ pBody

def local_world_to_platform_az(pWorld, pUAV, euler):
    roll, pitch, yaw = euler
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Rwb = Rz @ Ry @ Rx
    
    pRelWorld = pWorld - pUAV
    pRelBody = Rwb.T @ pRelWorld
    return np.arctan2(pRelBody[1], pRelBody[0])

class TrackAssociation:
    def __init__(self):
        # Association parameters matching track_association_simple.m
        self.gateEO = 0.024
        self.gateESM = 0.075
        
        self.totalGateEO = 0.038
        self.totalGateESM = 0.095
        
        self.gateTopoEO = 0.060
        self.gateTopoESM = 0.120
        
        self.lambdaTopoEO = 0.22
        self.lambdaTopoESM = 0.08
        
        self.switchPenaltyEO = 0.003
        self.switchPenaltyESM = 0.004
        
        self.ambDiffEO = 0.00001
        self.ambDiffESM = 0.00001
        self.topoK = 3
        
        # Temporal memory (mapping radar tracking ID -> last associated EO/ESM tracking ID)
        self.prevEOForRadar = {}
        self.prevESMForRadar = {}
        self.prevRadarForPlatform = {}

    @staticmethod
    def _topology_vector(Y_frame, valid_idx, center_id, dim_indices, topo_k=3):
        center = Y_frame[center_id, dim_indices]
        distances = []
        for other_id in valid_idx:
            if other_id == center_id:
                continue
            other = Y_frame[other_id, dim_indices]
            delta = np.array([wrap_pi(other[i] - center[i]) for i in range(len(dim_indices))])
            distances.append(float(np.linalg.norm(delta)))
        distances.sort()
        result = np.full(topo_k, 0.12, dtype=float)
        result[:min(topo_k, len(distances))] = distances[:topo_k]
        return result

    def associate_same_platform(
        self,
        radar_meas,
        eo_meas,
        esm_meas,
        radar_valid,
        eo_valid,
        esm_valid,
        platform_id,
    ):
        """Associate EO/ESM local tracks to radar tracks without using truth IDs."""
        radar_ids = np.flatnonzero(radar_valid)
        eo_ids = np.flatnonzero(eo_valid)
        esm_ids = np.flatnonzero(esm_valid)

        eo_cost = np.full((len(radar_ids), len(eo_ids)), np.inf)
        for i, radar_id in enumerate(radar_ids):
            for j, eo_id in enumerate(eo_ids):
                angular = np.linalg.norm([
                    wrap_pi(eo_meas[eo_id, 0] - radar_meas[radar_id, 0]),
                    wrap_pi(eo_meas[eo_id, 1] - radar_meas[radar_id, 1]),
                ])
                if angular > self.gateEO:
                    continue
                shared_neighbors = min(self.topoK, len(radar_ids) - 1, len(eo_ids) - 1)
                if shared_neighbors > 0:
                    radar_topology = self._topology_vector(radar_meas, radar_ids, radar_id, [0, 1])
                    eo_topology = self._topology_vector(eo_meas, eo_ids, eo_id, [0, 1])
                    topology = (
                        np.linalg.norm(
                            radar_topology[:shared_neighbors] - eo_topology[:shared_neighbors]
                        )
                        / np.sqrt(shared_neighbors)
                    )
                else:
                    topology = 0.0
                if topology > self.gateTopoEO:
                    continue
                cost = angular + self.lambdaTopoEO * topology
                previous = self.prevEOForRadar.get((platform_id, int(radar_id)))
                if previous == int(eo_id):
                    cost *= 0.70
                elif previous is not None:
                    cost += self.switchPenaltyEO
                if cost <= self.totalGateEO:
                    eo_cost[i, j] = cost

        esm_cost = np.full((len(radar_ids), len(esm_ids)), np.inf)
        for i, radar_id in enumerate(radar_ids):
            for j, esm_id in enumerate(esm_ids):
                angular = abs(wrap_pi(esm_meas[esm_id, 0] - radar_meas[radar_id, 0]))
                if angular > self.gateESM:
                    continue
                shared_neighbors = min(self.topoK, len(radar_ids) - 1, len(esm_ids) - 1)
                if shared_neighbors > 0:
                    radar_topology = self._topology_vector(radar_meas, radar_ids, radar_id, [0])
                    esm_topology = self._topology_vector(esm_meas, esm_ids, esm_id, [0])
                    topology = (
                        np.linalg.norm(
                            radar_topology[:shared_neighbors] - esm_topology[:shared_neighbors]
                        )
                        / np.sqrt(shared_neighbors)
                    )
                else:
                    topology = 0.0
                if topology > self.gateTopoESM:
                    continue
                cost = angular + self.lambdaTopoESM * topology
                previous = self.prevESMForRadar.get((platform_id, int(radar_id)))
                if previous == int(esm_id):
                    cost *= 0.70
                elif previous is not None:
                    cost += self.switchPenaltyESM
                if cost <= self.totalGateESM:
                    esm_cost[i, j] = cost

        eo_pairs = hungarian_assignment(eo_cost, self.totalGateEO)
        esm_pairs = hungarian_assignment(esm_cost, self.totalGateESM)
        eo_by_radar = {int(radar_ids[i]): int(eo_ids[j]) for i, j, _ in eo_pairs}
        esm_by_radar = {int(radar_ids[i]): int(esm_ids[j]) for i, j, _ in esm_pairs}
        for radar_id, eo_id in eo_by_radar.items():
            self.prevEOForRadar[(platform_id, radar_id)] = eo_id
        for radar_id, esm_id in esm_by_radar.items():
            self.prevESMForRadar[(platform_id, radar_id)] = esm_id
        return eo_by_radar, esm_by_radar

    def previous_modalities(self, platform_id, radar_id):
        key = (platform_id, radar_id)
        return self.prevEOForRadar.get(key), self.prevESMForRadar.get(key)

    def associate_radar_platforms(
        self,
        reference_world,
        platform_world,
        reference_valid,
        platform_valid,
        platform_id,
        gate_m=1500.0,
    ):
        """Associate another platform's radar tracks to the reference radar tracks."""
        reference_ids = np.flatnonzero(reference_valid)
        platform_ids = np.flatnonzero(platform_valid)
        cost_matrix = np.full((len(reference_ids), len(platform_ids)), np.inf)
        for i, reference_id in enumerate(reference_ids):
            for j, local_id in enumerate(platform_ids):
                distance = float(np.linalg.norm(
                    reference_world[reference_id] - platform_world[local_id]
                ))
                previous = self.prevRadarForPlatform.get((platform_id, int(local_id)))
                if previous == int(reference_id):
                    distance *= 0.70
                elif previous is not None:
                    distance += 200.0
                if distance <= gate_m:
                    cost_matrix[i, j] = distance

        pairs = hungarian_assignment(cost_matrix, gate_m)
        mapping = {}
        for ref_row, local_col, _ in pairs:
            reference_id = int(reference_ids[ref_row])
            local_id = int(platform_ids[local_col])
            mapping[local_id] = reference_id
            self.prevRadarForPlatform[(platform_id, local_id)] = reference_id
        return mapping

    def previous_radar_mapping(self, platform_id, local_id):
        return self.prevRadarForPlatform.get((platform_id, local_id))
        
    def topology_vector(self, Y_frame, valid_idx, center_id, dim_indices):
        """
        Extract topology feature vector for spatial association disambiguation.
        Y_frame: [num_targets, D] 
        valid_idx: list of valid indices
        center_id: the index of the current target in Y_frame
        dim_indices: indices of dimensions to compare (e.g., [1, 2] for az, el)
        """
        center = Y_frame[center_id, dim_indices]
        dList = []
        for q in valid_idx:
            if q == center_id: continue
            other = Y_frame[q, dim_indices]
            dist = np.sqrt(np.sum([wrap_pi(other[m] - center[m])**2 for m in range(len(dim_indices))]))
            dList.append(dist)
        
        dList.sort()
        padValue = 0.12
        v = np.full(self.topoK, padValue)
        numUse = min(self.topoK, len(dList))
        if numUse > 0:
            v[:numUse] = dList[:numUse]
        return v

    def associate_frame(self, radar_meas, eo_meas, esm_meas, radar_valid, eo_valid, esm_valid, uav_pos_dict, uav_euler_dict):
        """
        Associate cross-sensor measurements in one frame.
        Assuming radar is from UAV 0, EO from UAV 0, ESM from UAV 2 (as in dataset generation)
        radar_meas: [N, 3]
        eo_meas: [N, 2]
        esm_meas: [N, 2] (only first is az)
        """
        radarIdx = np.where(radar_valid)[0]
        eoIdx = np.where(eo_valid)[0]
        esmIdx = np.where(esm_valid)[0]
        
        # 1. Radar-EO Association
        C_RE = np.full((len(radarIdx), len(eoIdx)), np.inf)
        for i, rid in enumerate(radarIdx):
            rAz, rEl = radar_meas[rid, 0], radar_meas[rid, 1]
            for j, eid in enumerate(eoIdx):
                eAz, eEl = eo_meas[eid, 0], eo_meas[eid, 1]
                
                angleCost = np.sqrt(wrap_pi(eAz - rAz)**2 + wrap_pi(eEl - rEl)**2)
                if angleCost > self.gateEO: continue
                
                rTopo = self.topology_vector(radar_meas, radarIdx, rid, [0, 1])
                eTopo = self.topology_vector(eo_meas, eoIdx, eid, [0, 1])
                topoCost = np.linalg.norm(rTopo - eTopo) / np.sqrt(self.topoK)
                
                if topoCost > self.gateTopoEO: continue
                
                cost = angleCost + self.lambdaTopoEO * topoCost
                
                if self.prevEOForRadar.get(rid) == eid:
                    cost *= 0.70
                elif rid in self.prevEOForRadar:
                    cost += self.switchPenaltyEO
                    
                if cost <= self.totalGateEO:
                    C_RE[i, j] = cost
                    
        C_RE = reject_ambiguous_rows(C_RE, self.ambDiffEO)
        pairsRE = greedy_assignment(C_RE, self.totalGateEO)
        
        # 2. Radar-ESM Association
        C_RM = np.full((len(radarIdx), len(esmIdx)), np.inf)
        for i, rid in enumerate(radarIdx):
            rRange, rAz, rEl = radar_meas[rid, 2], radar_meas[rid, 0], radar_meas[rid, 1]
            
            pRadarUAV = uav_pos_dict[0] # Radar is on UAV 0
            eRadarUAV = uav_euler_dict[0]
            
            pTargetWorld = local_radar_meas_to_world(rRange, rAz, rEl, pRadarUAV, eRadarUAV)
            
            for j, mid in enumerate(esmIdx):
                pESMUAV = uav_pos_dict[2] # ESM is on UAV 2
                eESMUAV = uav_euler_dict[2]
                
                mAzPred = local_world_to_platform_az(pTargetWorld, pESMUAV, eESMUAV)
                mAz = esm_meas[mid, 0]
                
                angleCost = abs(wrap_pi(mAz - mAzPred))
                if angleCost > self.gateESM: continue
                
                rTopo = self.topology_vector(radar_meas, radarIdx, rid, [0])
                mTopo = self.topology_vector(esm_meas, esmIdx, mid, [0])
                topoCost = np.linalg.norm(rTopo - mTopo) / np.sqrt(self.topoK)
                
                if topoCost > self.gateTopoESM: continue
                
                cost = angleCost + self.lambdaTopoESM * topoCost
                
                if self.prevESMForRadar.get(rid) == mid:
                    cost *= 0.70
                elif rid in self.prevESMForRadar:
                    cost += self.switchPenaltyESM
                    
                if cost <= self.totalGateESM:
                    C_RM[i, j] = cost
                    
        C_RM = reject_ambiguous_rows(C_RM, self.ambDiffESM)
        pairsRM = greedy_assignment(C_RM, self.totalGateESM)
        
        # Build Results
        associated_groups = []
        for i, rid in enumerate(radarIdx):
            eoID = None
            esmID = None
            
            idxPairEO = next((p for p in pairsRE if p[0] == i), None)
            if idxPairEO:
                eoID = eoIdx[idxPairEO[1]]
                self.prevEOForRadar[rid] = eoID
            
            idxPairESM = next((p for p in pairsRM if p[0] == i), None)
            if idxPairESM:
                esmID = esmIdx[idxPairESM[1]]
                self.prevESMForRadar[rid] = esmID
                
            associated_groups.append({
                'Radar_local_id': rid,
                'EO_local_id': eoID,
                'ESM_local_id': esmID
            })
            
        return associated_groups
