import torch
import torch.nn as nn


def quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_invert(quaternion):
    return quaternion * quaternion.new_tensor([1, -1, -1, -1])

def quaternion_apply(quaternion, point):
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, f{point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def transform_points(quaternion, translation, points):
    quaternion = nn.functional.normalize(quaternion, dim=-1)
    transformed_points = points.unsqueeze(2) - translation.unsqueeze(1)
    transformed_points = quaternion_apply(quaternion.unsqueeze(1), transformed_points)
    return transformed_points

def sdfBox(quaternion, translation, dims, points):
    B,N,_ = points.shape
    _,K,_ = quaternion.shape
    dims = torch.abs(dims)
    transformed_points = transform_points(quaternion, translation, points)
    q_points = transformed_points.abs() - dims.unsqueeze(1).repeat(1,N,1,1)
    lengths = (q_points.max(torch.zeros_like(q_points))).norm(dim=-1)
    zeros_points = torch.zeros_like(lengths)
    xs = q_points[..., 0]
    ys = q_points[..., 1]
    zs = q_points[..., 2]
    filling = ys.max(zs).max(xs).min(zeros_points)
    return lengths + filling

def sdfExtrusion(sdf_2d, h, points):
    transformed_points = points  # [B,N,P,3]
    h = torch.abs(h)
    d = sdf_2d  # [B,N,P]
    z_diff = transformed_points[...,2].abs() - h.unsqueeze(dim=1)  # [B,N,K]
    a = d.max(z_diff).min(torch.zeros_like(z_diff))
    b = torch.cat((d.max(torch.zeros_like(z_diff)).unsqueeze(-1), (z_diff).max(torch.zeros_like(z_diff)).unsqueeze(-1)), -1)
    b = b.norm(dim=-1)
    
    return a + b
