import json
import typing as t
from pathlib import Path
import numpy as np
import torch
from typing import Tuple
from collections import defaultdict
import tqdm


def get_chamfer_distance_and_normal_consistency(
    gt_points: torch.Tensor,
    pred_points: torch.Tensor,
    gt_normals: torch.Tensor,
    pred_normals: torch.Tensor,
) -> Tuple[float, float]:
    gt_num_points = gt_points.shape[0]
    pred_num_points = pred_points.shape[0]

    points_gt_matrix = gt_points.unsqueeze(1).expand(
        [gt_points.shape[0], pred_num_points, gt_points.shape[-1]]
    )
    points_pred_matrix = pred_points.unsqueeze(0).expand(
        [gt_num_points, pred_points.shape[0], pred_points.shape[-1]]
    )

    distances = (points_gt_matrix - points_pred_matrix).pow(2).sum(dim=-1)
    match_pred_gt = distances.argmin(dim=0)
    match_gt_pred = distances.argmin(dim=1)

    dist_pred_gt = (pred_points - gt_points[match_pred_gt]).pow(2).sum(dim=-1).mean()
    dist_gt_pred = (gt_points - pred_points[match_gt_pred]).pow(2).sum(dim=-1).mean()

    normals_dot_pred_gt = (
        (pred_normals * gt_normals[match_pred_gt]).sum(dim=1).abs().mean()
    )

    normals_dot_gt_pred = (
        (gt_normals * pred_normals[match_gt_pred]).sum(dim=1).abs().mean()
    )
    chamfer_distance = dist_pred_gt + dist_gt_pred
    normal_consistency = (normals_dot_pred_gt + normals_dot_gt_pred) / 2

    return chamfer_distance.item(), normal_consistency.item()


def get_cd_nc_for_points(
    ground_truth_point_surface: Path,
    reconstructed_shapes_folder: Path,
    object_name: str,
) -> Tuple[float, float]:
    
    gt_file = (ground_truth_point_surface  / object_name).with_suffix(
        ".ply"
    )
    gt_pc_vertices, gt_pc_normals = read_point_normal_ply_file(gt_file.as_posix())
    
    # read preds
    pred_file = (reconstructed_shapes_folder  / object_name).with_suffix(".ply")
    
    try:
        pred_pc_vertices, pred_pc_normals = read_point_normal_ply_file(pred_file.as_posix())
    except:
        print('skipping')
        return 0,0
    
    
    num_points = min(pred_pc_vertices.shape[0],gt_pc_vertices.shape[0])
    if num_points == 0 :
        return 0,0

    return get_chamfer_distance_and_normal_consistency(
        to_tensor(gt_pc_vertices[:num_points]),
        to_tensor(pred_pc_vertices[:num_points]),
        to_tensor(gt_pc_normals[:num_points]),
        to_tensor(pred_pc_normals[:num_points]),
    )


def evaluate(
    valid_shape_names_file: str,
    reconstructed_shapes_folder: str,
    ground_truth_point_surface: str,
    out_folder: str,
):
    with open(valid_shape_names_file, "r") as f:
        eval_list = f.readlines()
    eval_list = [item.strip().split("/") for item in eval_list]
    print(f"Num objects: {len(eval_list)}")

    reconstructed_shapes_folder = Path(reconstructed_shapes_folder)
    ground_truth_point_surface = Path(ground_truth_point_surface)

    mean_metrics = defaultdict(float)
    total_entries = 0

    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)
    errors = []

    with tqdm.trange(len(eval_list)) as pbar:
        for idx in pbar:
            object_name = eval_list[idx][0]
            pbar.set_postfix_str(f"/{object_name}")


            points_cd, points_nc = get_cd_nc_for_points(
                ground_truth_point_surface,
                reconstructed_shapes_folder,
                object_name,
            )

            mean_metrics["chamfer_distance"] += points_cd
            mean_metrics["normal_consistency"] += points_nc
            
            if points_cd>0 and points_nc>0:
                total_entries += 1

    print(f"{len(errors)} error shapes")


    with open(out_folder / "errors.txt", "w") as f:
        f.write("\n".join(comp[0] + "/" + comp[1] for comp in errors))

    mean_metrics = {
        name: metric / total_entries for name, metric in mean_metrics.items()
    }
    with open(out_folder / "mean_metrics.json", "w") as f:
        json.dump(mean_metrics, f, indent=4)
    print(mean_metrics)


def to_tensor(data: np.ndarray) -> torch.Tensor:
    data = torch.from_numpy(data)
    return data

def get_edge_points(vertices: torch.Tensor, normals: torch.Tensor, sharp:float) -> np.ndarray:
    num_of_points = vertices.shape[0]
    points_mat_1 = vertices.view((num_of_points, 1, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    points_mat_2 = vertices.view((1, num_of_points, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    dist = (points_mat_1 - points_mat_2).pow(2).sum(dim=2)
    closest_index = (dist < 0.0001).int()

    normals_mat_1 = normals.view((num_of_points, 1, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    normals_mat_2 = normals.view((1, num_of_points, 3)).expand(
        [num_of_points, num_of_points, 3]
    )
    prod = (normals_mat_1 * normals_mat_2).sum(dim=2)
    
    all_edge_index = (prod.abs() < sharp).int()

    edge_index = (closest_index * all_edge_index).max(dim=1)[0]
    points = torch.cat((vertices, normals), dim=1)
    points = points[edge_index > 0.5].detach().cpu().numpy()

    np.random.shuffle(points)
    return points[:4096]

def get_simple_dataset_paths_from_config(
    processed_data_path: str, split_config_path: str
) -> t.List[str]:
    with open(split_config_path) as f:
        config = json.load(f)

    processed_data_path = Path(processed_data_path)
    renders = [(processed_data_path / path).as_posix() for path in config]

    return renders


def write_ply_point_normal(
    name: str, vertices: np.ndarray, normals: t.Optional[np.ndarray] = None
):
    fout = open(name, "w")
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(
                str(vertices[ii, 0])
                + " "
                + str(vertices[ii, 1])
                + " "
                + str(vertices[ii, 2])
                + " "
                + str(vertices[ii, 3])
                + " "
                + str(vertices[ii, 4])
                + " "
                + str(vertices[ii, 5])
                + "\n"
            )
    else:
        for ii in range(len(vertices)):
            fout.write(
                str(vertices[ii, 0])
                + " "
                + str(vertices[ii, 1])
                + " "
                + str(vertices[ii, 2])
                + " "
                + str(normals[ii, 0])
                + " "
                + str(normals[ii, 1])
                + " "
                + str(normals[ii, 2])
                + "\n"
            )
    fout.close()

def read_point_normal_ply_file(
    shape_file: str,
) -> t.Tuple[np.ndarray, np.ndarray]:
    file = open(shape_file, "r")
    lines = file.readlines()

    start = 0
    vertex_num = 0
    while True:
        line = lines[start].strip()
        if line == "end_header":
            start += 1
            break
        line = line.split()
        if line[0] == "element":
            if line[1] == "vertex":
                vertex_num = int(line[2])
        start += 1

    vertices = np.zeros([vertex_num, 3], np.float32)
    normals = np.zeros([vertex_num, 3], np.float32)
    for i in range(vertex_num):
        line = lines[i + start].split()
        vertices[i, 0] = float(line[0])  # X
        vertices[i, 1] = float(line[1])  # Y
        vertices[i, 2] = float(line[2])  # Z
        normals[i, 0] = float(line[3])  # normalX
        normals[i, 1] = float(line[4])  # normalY
        normals[i, 2] = float(line[5])  # normalZ
    return vertices, normals

