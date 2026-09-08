import numpy as np
from src.utils.ecef_utils import reconstruct_ecef_from_body_angles

class WingmanFusionEngine:
    def __init__(self):
        # Inverse variances (weights) based on generation noise parameters
        # Radar angle std=0.002 -> var=4e-6 -> inv=250000
        # EO angle std=0.002 -> var=4e-6 -> inv=250000
        # ESM angle std=0.005 -> var=2.5e-5 -> inv=40000
        self.w_radar_az = 1.0 / (0.002 ** 2)
        self.w_radar_el = 1.0 / (0.002 ** 2)
        
        self.w_eo_az = 1.0 / (0.002 ** 2)
        self.w_eo_el = 1.0 / (0.002 ** 2)
        
        self.w_esm_az = 1.0 / (0.005 ** 2)

    def fuse(self, radar_align, eo_align, esm_align, uav_pos, uav_vel, radar_valid, eo_valid, esm_valid):
        """
        Fuses Radar, EO, and ESM data into absolute ECEF XYZ coordinates.
        Inputs:
            radar_align: [K, 3] -> (r, az, el)
            eo_align: [K, 2] -> (az, el)
            esm_align: [K, 1] -> (az,)
            uav_pos: [K, 3] -> ECEF Position
            uav_vel: [K, 3] -> ECEF Velocity
            *_valid: [K] boolean masks
        Returns:
            ecef_xyz: [K, 3]
            valid_out: [K] boolean mask indicating if fusion was successful (needs radar for range)
        """
        K = radar_align.shape[0]
        ecef_xyz = np.full((K, 3), np.nan)
        valid_out = np.zeros(K, dtype=bool)

        for k in range(K):
            if not radar_valid[k] or not np.all(np.isfinite(radar_align[k])):
                # Without Radar, we have no range, cannot resolve 3D position locally
                continue
                
            r_radar, az_radar, el_radar = radar_align[k]
            
            az_fused = 0.0
            el_fused = 0.0
            sum_w_az = 0.0
            sum_w_el = 0.0

            # 1. Radar Contribution
            az_fused += az_radar * self.w_radar_az
            sum_w_az += self.w_radar_az
            el_fused += el_radar * self.w_radar_el
            sum_w_el += self.w_radar_el

            # 2. EO Contribution
            if eo_valid[k] and np.all(np.isfinite(eo_align[k])):
                az_eo, el_eo = eo_align[k]
                az_fused += az_eo * self.w_eo_az
                sum_w_az += self.w_eo_az
                el_fused += el_eo * self.w_eo_el
                sum_w_el += self.w_eo_el

            # 3. ESM Contribution
            if esm_valid[k] and np.isfinite(esm_align[k, 0]):
                az_esm = esm_align[k, 0]
                az_fused += az_esm * self.w_esm_az
                sum_w_az += self.w_esm_az

            # Normalize
            az_fused /= sum_w_az
            el_fused /= sum_w_el
            r_fused = r_radar  # Only Radar provides range

            # Convert to ECEF
            pWorld = reconstruct_ecef_from_body_angles(az_fused, el_fused, r_fused, uav_pos[k], uav_vel[k])
            ecef_xyz[k] = pWorld
            valid_out[k] = True

        return ecef_xyz, valid_out
