# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python projects/ppr/eval/compute_metrics.py --pred_prefix "" --fps 10 --skip 3
# python projects/ppr/eval/compute_metrics.py  --fps 30
import sys, os
import pdb
import json
import glob
import numpy as np
import argparse
import trimesh
import tqdm

sys.path.insert(0, os.getcwd())
from eval_utils import load_ama_intrinsics, ama_eval


cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)
from lab4d.utils.io import save_vid
from lab4d.utils.mesh_render_utils import PyRenderWrapper

parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument(
    "--testdir",
    default="logdir/ama-samba-fg-urdf-cse/export_0000/",
    help="path to the directory with results",
)
parser.add_argument(
    "--gt_seq", default="T_samba-1", help="name of the ground-truth sequqnce"
)
parser.add_argument(
    "--pred_prefix",
    default="/data3/gengshay/eval/humor/T_samba1",
    help="name of the pred sequqnce",
)
parser.add_argument("--skip", default=1, type=int, help="pred mesh has n times less")
parser.add_argument("--fps", default=30, type=int, help="fps of the video")
args = parser.parse_args()


def main():
    ama_path = "database/ama/"
    # gt
    gt_name, gt_cam_id = args.gt_seq.split("-")
    gt_cam_path = "%s/%s/calibration/Camera%s.Pmat.cal" % (ama_path, gt_name, gt_cam_id)
    intrinsics_gt, Gmat_gt = load_ama_intrinsics(gt_cam_path)

    # glob meshes
    gt_mesh_dir = "%s/%s/meshes/" % (ama_path, gt_name)
    gt_mesh_dict = {}
    for fidx, path in enumerate(sorted(glob.glob("%s/mesh_*.obj" % (gt_mesh_dir)))):
        if fidx % args.skip == 0:
            gt_mesh_dict[fidx] = trimesh.load(path, process=False)
            gt_mesh_dict[fidx].apply_transform(Gmat_gt)
    if len(gt_mesh_dict) == 0:
        print("no mesh found that matches %s*" % (args.testdir))
        return
    print("found %d groune-truth meshes" % (len(gt_mesh_dict)))

    # pred (from lab4d)
    camera_info = json.load(open("%s/camera.json" % (args.testdir), "r"))
    raw_size = camera_info["raw_size"]  # h,w

    # glob predicted meshes (from either lab4d or other methods)
    if args.pred_prefix == "":
        pred_prefix = "%s/mesh/fg-" % (args.testdir)  # use lab4d
        pred_mesh_paths = glob.glob("%s*.obj" % (pred_prefix))
        intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
        extrinsics = np.repeat(np.eye(4)[None], len(pred_mesh_paths), axis=0)
    else:
        pred_mesh_paths = glob.glob("%s*.obj" % (args.pred_prefix))
        pred_camera_paths = sorted(glob.glob("%s*.txt" % (args.pred_prefix)))
        cameras = np.stack([np.loadtxt(i) for i in pred_camera_paths], 0)
        intrinsics = cameras[:, 3]
        extrinsics = np.repeat(np.eye(4)[None], len(pred_mesh_paths), axis=0)
        extrinsics[:, :3] = cameras[:, :3]
    pred_mesh_dict = {}
    for fidx, mesh_path in enumerate(sorted(pred_mesh_paths)):
        fidx = int(mesh_path.split("/")[-1].split("-")[-1].split(".")[0])
        pred_mesh_dict[args.skip * fidx] = trimesh.load(mesh_path, process=False)
    assert len(pred_mesh_dict) == len(gt_mesh_dict)

    # evaluate
    # ama_eval(all_verts_gt, all_verts_gt, verbose=True)
    cd_avg, f010_avg, f005_avg, f002_avg, pred_cd_dict, gt_cd_dict = ama_eval(
        pred_mesh_dict, gt_mesh_dict, verbose=True
    )

    # render
    renderer_gt = PyRenderWrapper(raw_size)
    renderer_pred = PyRenderWrapper(raw_size)
    frames = []
    for fidx, mesh_obj in tqdm.tqdm(gt_mesh_dict.items()):
        renderer_gt.set_intrinsics(intrinsics_gt)
        color_gt = renderer_gt.render(mesh_obj, force_gray=True)[0]
        cd_gt = renderer_gt.render(gt_cd_dict[fidx])[0]

        renderer_pred.set_intrinsics(intrinsics[fidx // args.skip])
        color_pred = renderer_pred.render(pred_mesh_dict[fidx], force_gray=True)[0]
        cd_pred = renderer_pred.render(pred_cd_dict[fidx])[0]

        color = np.concatenate([color_gt, color_pred], axis=1)
        cd = np.concatenate([cd_gt, cd_pred], axis=1)
        final = np.concatenate([color, cd], axis=0)
        frames.append(final.astype(np.uint8))

    save_vid(
        "%s/render" % args.testdir,
        frames,
        suffix=".mp4",
        upsample_frame=-1,
        fps=args.fps,
    )
    print("saved to %s/render.mp4" % args.testdir)


if __name__ == "__main__":
    main()
