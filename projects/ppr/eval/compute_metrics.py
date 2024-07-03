# Copyright (c) 2023 Gengshan Yang, Carnegie Mellon University.
# python projects/ppr/eval/compute_metrics.py --pred_prefix "" --fps 10 --skip 3
# python projects/ppr/eval/compute_metrics.py
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
from lab4d.utils.pyrender_wrapper import PyRenderWrapper
from lab4d.utils.vis_utils import append_xz_plane

parser = argparse.ArgumentParser(description="script to render extraced meshes")
parser.add_argument(
    "--testdir",
    default="logdir/ama-samba-4v-fg-bob-2g-r120-cse/export_0000/",
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
        pred_prefix = "%s/fg/mesh/" % (args.testdir)  # use lab4d
        pred_mesh_paths = glob.glob("%s*.obj" % (pred_prefix))
        intrinsics = np.asarray(camera_info["intrinsics"], dtype=np.float32)
        # transform to view coord
        extrinsics = json.load(open("%s/fg/motion.json" % (args.testdir), "r"))
        extrinsics = np.asarray(extrinsics["field2cam"])

        if os.path.exists("%s/bg/motion.json" % (args.testdir)):
            extrinsics_bg = json.load(open("%s/bg/motion.json" % (args.testdir), "r"))
            extrinsics_bg = np.asarray(extrinsics_bg["field2cam"])

            # align bg floor with xz plane
            field2world_path = "%s/bg/field2world.json" % (args.testdir)
            field2world = np.asarray(json.load(open(field2world_path, "r")))
            world2field = np.linalg.inv(field2world)
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
        pred_mesh_dict[args.skip * fidx].apply_transform(extrinsics[fidx])
        # pred_mesh_dict[args.skip * fidx].apply_transform(np.linalg.inv(Gmat_gt))
    assert len(pred_mesh_dict) == len(gt_mesh_dict)

    if os.path.exists("%s/bg/motion.json" % (args.testdir)):
        pred_mesh_paths_bg = glob.glob("%s/bg/mesh/*.obj" % (args.testdir))
        pred_mesh_dict_bg = {}
        for fidx, mesh_path in enumerate(sorted(pred_mesh_paths_bg)):
            fidx = int(mesh_path.split("/")[-1].split("-")[-1].split(".")[0])
            pred_mesh_dict_bg[args.skip * fidx] = trimesh.load(mesh_path, process=False)
            pred_mesh_dict_bg[args.skip * fidx].apply_transform(extrinsics_bg[fidx])

    # evaluate
    # ama_eval(all_verts_gt, all_verts_gt, verbose=True)
    (
        cd_avg,
        f010_avg,
        f005_avg,
        f002_avg,
        pred_mesh_dict,
        pred_cd_dict,
        gt_cd_dict,
    ) = ama_eval(pred_mesh_dict, gt_mesh_dict, verbose=True)

    # render
    renderer_gt = PyRenderWrapper(raw_size)
    renderer_pred = PyRenderWrapper(raw_size)
    frames = []
    for fidx, mesh_obj in tqdm.tqdm(gt_mesh_dict.items(), desc=f"Rendering:"):
        # world_to_cam_pred = extrinsics_bg[fidx] @ world2field
        mesh_obj = append_xz_plane(mesh_obj, Gmat_gt)
        gt_cd_dict[fidx] = append_xz_plane(gt_cd_dict[fidx], Gmat_gt)
        pred_mesh_dict[fidx] = append_xz_plane(pred_mesh_dict[fidx], Gmat_gt)
        pred_cd_dict[fidx] = append_xz_plane(pred_cd_dict[fidx], Gmat_gt)
        # pred_mesh_dict[fidx] = trimesh.util.concatenate(
        #     [pred_mesh_dict[fidx], pred_mesh_dict_bg[fidx]]
        # )
        # mesh_obj.export("tmp/0.obj")
        # pred_mesh_dict[fidx].export("tmp/1.obj")
        # pdb.set_trace()

        # renderer_gt.set_camera_frontal(4, gl=True)
        renderer_gt.set_intrinsics(intrinsics_gt)
        renderer_gt.align_light_to_camera()
        color_gt = renderer_gt.render({"shape": mesh_obj})[0]
        cd_gt = renderer_gt.render({"shape": gt_cd_dict[fidx]})[0]

        # renderer_pred.set_camera_frontal(4, gl=True)
        renderer_pred.set_intrinsics(intrinsics[fidx // args.skip])
        renderer_pred.align_light_to_camera()
        color_pred = renderer_pred.render({"shape": pred_mesh_dict[fidx]})[0]
        cd_pred = renderer_pred.render({"shape": pred_cd_dict[fidx]})[0]

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
