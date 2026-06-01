# ecef_utils.py — WGS84 ECEF 坐标系工具函数
import numpy as np

WGS84_A  = 6378137.0          # 半长轴 (m)
WGS84_E2 = 0.00669437999014   # 第一偏心率平方

def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    """大地坐标 (纬度/经度/高度) → ECEF (X, Y, Z)，单位 m"""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    X = (N + alt_m) * np.cos(lat) * np.cos(lon)
    Y = (N + alt_m) * np.cos(lat) * np.sin(lon)
    Z = (N * (1.0 - WGS84_E2) + alt_m) * np.sin(lat)
    return np.array([X, Y, Z], dtype=float)


def ecef_to_geodetic(x, y, z):
    """ECEF → 大地坐标 (lat_deg, lon_deg, alt_m)，Bowring 迭代法"""
    lon = np.arctan2(y, x)
    p   = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1.0 - WGS84_E2))
    for _ in range(10):
        N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        lat_new = np.arctan2(z + WGS84_E2 * N * np.sin(lat), p)
        if abs(lat_new - lat) < 1e-12:
            break
        lat = lat_new
    N   = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    cos_lat = np.cos(lat)
    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) / abs(np.sin(lat)) - N * (1.0 - WGS84_E2)
    return np.degrees(lat), np.degrees(lon), alt


def ecef_to_enu_matrix(lat_deg, lon_deg):
    """
    返回 3×3 旋转矩阵 R，使得 v_enu = R @ v_ecef
    ENU 轴：East, North, Up
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    R = np.array([
        [-np.sin(lon),               np.cos(lon),              0.0        ],
        [-np.sin(lat)*np.cos(lon),  -np.sin(lat)*np.sin(lon),  np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),   np.cos(lat)*np.sin(lon),  np.sin(lat)],
    ])
    return R  # v_enu = R @ v_ecef


def enu_to_ecef_matrix(lat_deg, lon_deg):
    """返回 R，使得 v_ecef = R @ v_enu"""
    return ecef_to_enu_matrix(lat_deg, lon_deg).T


def enu_vel_to_ecef(vE, vN, vU, lat_deg, lon_deg):
    """ENU 速度分量 → ECEF 速度矢量"""
    R = enu_to_ecef_matrix(lat_deg, lon_deg)
    return R @ np.array([vE, vN, vU], dtype=float)


def get_body_frame_axes(vel_ecef, pos_ecef):
    """
    由平台 ECEF 速度 + 位置，计算机体坐标系在 ECEF 中的三个单位轴：
    unit_fwd  — 前向（沿速度方向）
    unit_right — 右向（垂直于前向，在当地水平面内）
    unit_up   — 上向（垂直于前向，朝向当地天顶）

    使用当地地理"上"方向（ECEF 位置向量归一化）而非全局 Z 轴。
    """
    local_up = pos_ecef / np.linalg.norm(pos_ecef)

    speed = np.linalg.norm(vel_ecef)
    if speed < 1e-6:
        # 无速度：以任意方向为前向
        unit_fwd = np.array([1.0, 0.0, 0.0])
    else:
        unit_fwd = vel_ecef / speed

    unit_right = np.cross(local_up, unit_fwd)
    norm_r = np.linalg.norm(unit_right)
    if norm_r < 1e-6:
        # 速度与"上"方向平行（垂直起降场景），退化为任意右向
        arb = np.array([1.0, 0.0, 0.0]) if abs(local_up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        unit_right = np.cross(local_up, arb)
        unit_right /= np.linalg.norm(unit_right)
    else:
        unit_right /= norm_r

    unit_up = np.cross(unit_fwd, unit_right)
    unit_up /= np.linalg.norm(unit_up)

    return unit_fwd, unit_right, unit_up


def reconstruct_ecef_from_body_angles(az_rad, el_rad, r_m, platform_pos_ecef, platform_vel_ecef):
    """
    由雷达测量 (方位角 az, 俯仰角 el, 斜距 r) + 平台 ECEF 状态，
    反算目标 ECEF 坐标。
    机体坐标系：x=前, y=右, z=上（使用当地地理"上"）
    """
    unit_fwd, unit_right, unit_up = get_body_frame_axes(platform_vel_ecef, platform_pos_ecef)
    dx_b = r_m * np.cos(el_rad) * np.cos(az_rad)
    dy_b = r_m * np.cos(el_rad) * np.sin(az_rad)
    dz_b = r_m * np.sin(el_rad)
    return platform_pos_ecef + dx_b * unit_fwd + dy_b * unit_right + dz_b * unit_up
