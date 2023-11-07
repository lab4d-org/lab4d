# given image set A and refernce image b, find the best match a of b in A.
import configparser
import glob
import os
import sys
import pdb
import trimesh
import tqdm

import open3d as o3d
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0, os.getcwd())
from lab4d.utils.gpu_utils import gpu_map
from lab4d.utils.io import save_vid
from lab4d.utils.vis_utils import draw_cams
from lab4d.utils.geom_utils import K2mat
from preprocess.third_party.vcnplus.flowutils.flowlib import compute_color
from preprocess.scripts.extract_dinov2 import load_dino_model
from projects.csim.render_polycam import PolyCamRender, depth_to_canonical
from projects.csim.transform_bg_cams import transform_bg_cams


@torch.no_grad()
def extract_dino_feat(dinov2_model, rgb, size=16, upsample_ratio=2):
    """
    feat: (s,s, 384)
    """
    device = next(dinov2_model.parameters()).device
    h, w, _ = rgb.shape

    img = Image.fromarray(rgb)
    input_size = size * dinov2_model.patch_size
    transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    img = transform(img)[:3].unsqueeze(0).to(device)
    # TODO: use stride=4 to get high-res feature
    feat = dinov2_model.forward_features(img)["x_norm_patchtokens"]
    feat = feat.reshape(size, size, -1)

    feat = feat.permute(2, 0, 1)
    feat = F.interpolate(
        feat[None], size=(size * upsample_ratio, size * upsample_ratio), mode="bilinear"
    )[0]
    feat = feat.permute(1, 2, 0)
    return feat


def extract_dinov2_seq(dinov2_model, imglist):
    feats = []
    for rgb in tqdm.tqdm(imglist, "extracting features"):
        feat = extract_dino_feat(dinov2_model, rgb)
        feats.append(feat)
    feats = torch.stack(feats, 0)  # N, h,w, 16
    return feats


def data_pca(dinov2_model, imglist_all):
    feat_sampled = []
    imglist_all_perm = np.random.permutation(imglist_all)
    for impath in tqdm.tqdm(imglist_all_perm[:100], "sampling features for pca"):
        # print(impath)
        rgb = cv2.imread(impath)[:, :, ::-1]
        feat = extract_dino_feat(dinov2_model, rgb)
        feat = feat.reshape(-1, feat.shape[-1])
        rand_idx = np.random.permutation(len(feat))[:1000]
        feat_sampled.append(feat[rand_idx])
    feat_sampled = torch.cat(feat_sampled, 0)
    feat_sampled = feat_sampled.cpu().numpy()

    with threadpool_limits(limits=1):
        pca_vis = PCA(n_components=3)
        pca_vis.fit(feat_sampled)

        pca_save = PCA(n_components=16)
        pca_save.fit(feat_sampled)
    return pca_save, pca_vis


def visualize_feat(pca_vis, feat):
    h, w, _ = feat.shape
    feat = feat.reshape(-1, feat.shape[-1])
    feat_vis = pca_vis.transform(feat.cpu())
    feat_vis = (feat_vis - feat_vis.min()) / (feat_vis.max() - feat_vis.min())
    feat_vis = feat_vis * 255
    feat_vis = feat_vis.reshape(h, w, 3).astype(np.uint8)
    return feat_vis


def plot_matches(vis_img, corresp_map, ref_size):
    """
    vis_img: h1,w1,3
    best_corresp_map: h2,w2,2
    raw_size, 2, size of the original image
    """
    refh, refw = ref_size
    trgh = refh
    trgw = vis_img.shape[1] - refw
    h_ratio_ref = refh / corresp_map.shape[0]
    w_ratio_ref = refw / corresp_map.shape[1]
    h_ratio_trg = trgh / corresp_map.shape[0]
    w_ratio_trg = trgw / corresp_map.shape[1]

    u, v = np.meshgrid(range(refw), range(refh))
    u = u / u.max() - 0.5
    v = v / v.max() - 0.5
    colormap = compute_color(u, v)

    matches = []
    for idx in range(corresp_map.shape[1]):
        for idy in range(corresp_map.shape[0]):
            if corresp_map[idy, idx, 0] < 0:
                continue
            x1 = int(idx * w_ratio_ref)
            y1 = int(idy * h_ratio_ref)
            x2 = refw + int(corresp_map[idy, idx, 1] * w_ratio_trg)
            y2 = int(corresp_map[idy, idx, 0] * h_ratio_trg)

            matches.append((x1, y1, x2 - refw, y2))

            # assign color based on x1,y1
            color = colormap[y1, x1]
            cv2.line(vis_img, (x1, y1), (x2, y2), color, 1)
            # annotate the start and end point
            cv2.circle(vis_img, (x1, y1), 5, color, -1)
            cv2.circle(vis_img, (x2, y2), 5, color, -1)

    # cv2.imwrite("tmp/vis.jpg", vis_img)
    return vis_img, matches


def visualize_matches(
    feats_ref,
    feats_trg,
    best_indices,
    best_corresp_map,
    pca_vis,
    rgb_ref_list,
    rgb_trg_list,
    sil_ref_list,
):
    # visualize
    vis_imgs = []
    matches_all = []
    for idx, feat_ref in enumerate(tqdm.tqdm(feats_ref, "dimension reduction")):
        h, w, _ = feat_ref.shape
        refh, refw, _ = rgb_ref_list[idx].shape
        sil = sil_ref_list[idx].astype(np.uint8)
        trg_idx = best_indices[idx]
        # rgb_ref = cv2.resize(rgb_ref_list[idx], (w, h))
        # rgb_trg = cv2.resize(rgb_trg_list[trg_idx], (w, h))
        rgb_ref = rgb_ref_list[idx]
        rgb_trg = rgb_trg_list[trg_idx]
        rgb_trg = cv2.resize(
            rgb_trg, (refh * rgb_trg.shape[1] // rgb_trg.shape[0], refh)
        )
        rgb_ref[sil > 0] = 0

        feat_ref_vis = visualize_feat(pca_vis, feat_ref)
        sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_NEAREST)
        sil = cv2.dilate(sil, np.ones((5, 5), np.uint8), iterations=1)
        # rgb_ref[sil > 0] = 0
        feat_ref_vis[sil > 0] = 0

        feat_trg_vis = visualize_feat(pca_vis, feats_trg[trg_idx])

        # resize
        feat_ref_vis = cv2.resize(feat_ref_vis, (refw, refh))
        feat_trg_vis = cv2.resize(
            feat_trg_vis, (refh * rgb_trg.shape[1] // rgb_trg.shape[0], refh)
        )
        vis_img = np.concatenate([rgb_ref, rgb_trg], 1)
        # plot matches
        vis_img, matches = plot_matches(vis_img, best_corresp_map[idx], (refh, refw))
        matches_all.append(matches)
        vis_feat = np.concatenate([feat_ref_vis, feat_trg_vis], 1)
        vis = np.concatenate([vis_img, vis_feat], 0)
        vis_imgs.append(vis)
    return vis_imgs, matches_all


def cycle_consistency(corresp_map, corresp_map_inv):
    """
    Input:
        corresp_map: h1,w1,2
        corresp_map_inv: h2,w2,2
    Returns:
        consistency_score: h1,w1
    """
    h, w = corresp_map.shape[:2]
    corresp_map_list = corresp_map.reshape(-1, 2).long()
    ref_points = np.stack(np.meshgrid(range(w), range(h)), -1)
    ref_points = torch.tensor(ref_points, device=corresp_map.device)
    cycle_points = corresp_map_inv[corresp_map_list[:, 1], corresp_map_list[:, 0]]
    cycle_points = cycle_points.view(h, w, 2)
    distance = (cycle_points - ref_points).norm(2, -1)
    score = 1 - distance / np.sqrt(h * w)
    # cv2.imwrite("tmp/0.jpg", score.cpu().numpy()*255)
    return score


def find_correspondence(feats_ref, feats_trg, sil_ref_list, vis_threshold=0.6):
    best_indices = []
    best_corresp_map = []
    for idx, feat_ref in enumerate(tqdm.tqdm(feats_ref, "computing distance")):
        h, w, _ = feat_ref.shape
        sil = sil_ref_list[idx].astype(np.uint8)
        sil = cv2.resize(sil, (w, h), interpolation=cv2.INTER_NEAREST)
        sil = cv2.dilate(sil, np.ones((5, 5), np.uint8), iterations=1)
        dist_list = []
        corresp_maps = []
        feat_ref = F.normalize(feat_ref, 2, -1)
        for feat_trg in feats_trg:
            h_inv, w_inv, _ = feat_trg.shape
            feat_trg = F.normalize(feat_trg, 2, -1)
            # local: h1,w1,h2,w2,f
            score = feat_trg[:, :, None, None] * feat_ref[None, None]
            score = score.sum(-1)
            # for each pixel in h2,w2, find the index of h1,w1 with highest score
            score_map, corresp_map = score.view(-1, h, w).max(0)

            # h2,w2 => h2,w2,2
            corresp_map = torch.stack([corresp_map // w, corresp_map % w], -1).float()

            # consistency check: for each pixel in h1,w1, find the index of h2,w2 with highest score
            score_map_inv, corresp_map_inv = score.view(h_inv, w_inv, -1).max(-1)
            corresp_map_inv = torch.stack(
                [corresp_map_inv // w_inv, corresp_map_inv % w_inv], -1
            ).float()
            # check if the correspondence is consistent
            # if not, set the score to -1
            # pdb.set_trace()
            consistency_score = cycle_consistency(corresp_map, corresp_map_inv)

            score = score[sil <= 0].view(-1).median()
            corresp_map[score_map < vis_threshold] = -1
            corresp_map[consistency_score < 0.8] = -1
            corresp_maps.append(corresp_map)  # h1,w1,h2,w2
            # # global distance
            # dist = (
            #     F.normalize(feat_trg.mean(0).mean(0), 2, -1)
            #     * F.normalize(feat_ref.mean(0).mean(0), 2, -1)
            # ).sum(-1)
            dist_list.append(score)
        score, idx = torch.sort(torch.stack(dist_list))
        best_corresp_map.append(corresp_maps[idx[-1]])
        best_indices.append(idx[-1])
    return best_indices, best_corresp_map


def solve_pnp(x1y1, xyz, intrinsics):
    x1y1 = x1y1.astype(np.float32)[:, None]
    xyz = xyz[:, None]
    reproj, rtmat = robust_pnp(x1y1, xyz, intrinsics)
    return rtmat, reproj


def robust_pnp(x1y1, xyz, intrinsics):
    """
    align rot1 to rot2 using RANSAC
    """
    n_samples = len(x1y1)
    n_iters = 1000
    flags = cv2.SOLVEPNP_AP3P

    med_dists = []
    rtmats = []
    for i in range(n_iters):
        # sample 4 points and solve pnp
        idx = np.random.choice(n_samples, 4, replace=False)
        ret, rvec, trans = cv2.solvePnP(xyz[idx], x1y1[idx], intrinsics, 0, flags=flags)
        if ret == False or np.isnan(trans).any():
            continue
        rmat = cv2.Rodrigues(rvec)[0]
        trans = trans[:, 0]

        # compute reprojection error
        xyz_view = xyz @ rmat.T[None] + trans[None, None]
        xy_view = xyz_view @ intrinsics.T[None]
        xy_view = xy_view[..., :2] / xy_view[..., 2:]
        dist = np.linalg.norm(xy_view - x1y1, 2, -1)
        med_dists.append(np.median(dist))

        # save solution
        rtmat = np.eye(4)
        rtmat[:3, :3] = rmat
        rtmat[:3, 3] = trans
        rtmats.append(rtmat)

    # Convert rotation vectors back to rotation matrices
    best_idx = np.argmin(med_dists)
    best_rtmat = rtmats[best_idx]
    med_dist = med_dists[best_idx]

    # print(inliers)
    return med_dist, best_rtmat


def solve_procrustes(depth_ref, intrinsics_ref, xy_ref, xyz_corresp):
    from preprocess.libs.geometry import (
        compute_procrustes,
        compute_procrustes_median,
    )

    xyz_ref = depth_to_canonical(depth_ref, intrinsics_ref, np.eye(4))

    # trimesh.Trimesh(vertices=xyz_ref.reshape(-1, 3)[::100]).export(
    #     "tmp/xyz_ref.obj"
    # )
    xyz_ref = xyz_ref[xy_ref[..., 1], xy_ref[..., 0]]

    view2scene_r, view2scene_t, error = compute_procrustes_median(xyz_ref, xyz_corresp)
    view2scene = np.eye(4)
    view2scene[:3, :3] = view2scene_r
    view2scene[:3, 3] = view2scene_t
    # aligned = xyz_ref @ view2scene[:3, :3].T + view2scene[:3, 3]
    # trimesh.Trimesh(vertices=xyz_ref).export("tmp/xyz_ref.obj")
    # trimesh.Trimesh(vertices=xyz_corresp).export("tmp/xyz_corresp.obj")
    # trimesh.Trimesh(vertices=aligned).export("tmp/xyz_ref_aligned.obj")
    return (view2scene, error)


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def extract_dinov2(trg_path, ref_path, vis_threshold=0.6):
    seqname = ref_path.strip("/").split("/")[-1]
    config = configparser.RawConfigParser()
    seqname_global = seqname.rsplit("-", 1)[0]
    seqname_sub = int(seqname.rsplit("-", 1)[1])
    config.read("database/configs/%s.config" % seqname_global)
    intrinsics_ref = config.get("data_%d" % seqname_sub, "ks")
    intrinsics_ref = np.array([float(x) for x in intrinsics_ref.split(" ")])
    intrinsics_ref[0] = 800
    intrinsics_ref[1] = 800
    dinov2_model = load_dino_model()
    trg_list = sorted(glob.glob("%s/*.jpg" % trg_path))
    rgb_trg_list = [cv2.imread(impath)[:, :, ::-1] for impath in trg_list]
    rgb_trg_list = [np.transpose(rgb, (1, 0, 2))[:, ::-1] for rgb in rgb_trg_list]
    ref_list = sorted(glob.glob("%s/*.jpg" % ref_path))
    rgb_ref_list = [cv2.imread(impath)[:, :, ::-1] for impath in ref_list]
    sil_ref_list = [
        np.load(impath.replace("JPEGImages", "Annotations").replace(".jpg", ".npy"))
        for impath in ref_list
    ]
    sil_ref_list = [(sil > 0).astype(np.uint8) for sil in sil_ref_list]

    # extract features for trg and ref
    save_path_trg = "%s/dinov2_features.npy" % (trg_path)
    if os.path.exists(save_path_trg):
        print("found dino features at %s" % save_path_trg)
        feats_trg = np.load(save_path_trg)
        feats_trg = torch.from_numpy(feats_trg).to("cuda:0")
    else:
        feats_trg = extract_dinov2_seq(dinov2_model, rgb_trg_list)
        np.save(save_path_trg, feats_trg.cpu().numpy().astype(np.float16))
        print("dino features saved to %s" % save_path_trg)

    save_dir_ref = ref_path.replace("JPEGImages", "Relocalization")
    save_path_ref = "%s/dinov2_features.npy" % (save_dir_ref)
    if os.path.exists(save_path_ref):
        print("found dino features at %s" % save_path_ref)
        feats_ref = np.load(save_path_ref)
        feats_ref = torch.from_numpy(feats_ref).to("cuda:0")
    else:
        os.makedirs(save_dir_ref, exist_ok=True)
        feats_ref = extract_dinov2_seq(dinov2_model, rgb_ref_list)
        np.save(save_path_ref, feats_ref.cpu().numpy().astype(np.float16))
        print("dino features saved to %s" % save_path_ref)

    # find correspondence
    best_indices, best_corresp_map = find_correspondence(
        feats_ref, feats_trg, sil_ref_list, vis_threshold
    )

    # reduce dimension
    pca_save, pca_vis = data_pca(dinov2_model, trg_list + ref_list)

    vis_imgs, matches_all = visualize_matches(
        feats_ref,
        feats_trg,
        best_indices,
        best_corresp_map,
        pca_vis,
        rgb_ref_list,
        rgb_trg_list,
        sil_ref_list,
    )
    poly_path = trg_path.strip("/").rsplit("/", 2)[0]
    depth_list = sorted(glob.glob("%s/keyframes/depth/*.png" % poly_path))
    renderer_target = PolyCamRender(poly_path, image_size=rgb_trg_list[0].shape[:2])
    renderer_ref = PolyCamRender(poly_path, image_size=rgb_ref_list[0].shape[:2])
    trg_final_size = (
        rgb_ref_list[0].shape[0] * rgb_trg_list[0].shape[1] // rgb_trg_list[0].shape[0],
        rgb_ref_list[0].shape[0],
    )
    ref_final_size = rgb_ref_list[0].shape[:2][::-1]
    scene2views = []
    errors = []
    for i in tqdm.tqdm(range(len(vis_imgs)), "rendering"):
        # for each reference image
        sel_i = best_indices[i]
        rgb_target, depth_target = renderer_target.render(sel_i)
        xyz_target = depth_to_canonical(
            depth_target,
            renderer_target.intrinsics[sel_i],
            renderer_target.extrinsics[sel_i],
        )
        rgb_target = cv2.resize(rgb_target, trg_final_size)
        xyz_target = cv2.resize(xyz_target, trg_final_size)

        # find the corresponding 3d coordinate
        matches = matches_all[i]
        if len(matches) == 0:
            print("no matches found")
            scene2view = np.eye(4)
            error = 100
        else:
            xy_ref = np.array(matches)[:, :2]
            xy_trg = np.array(matches)[:, 2:]
            xyz_corresp = xyz_target[xy_trg[:, 1], xy_trg[:, 0]]

            # # solve pnp
            # Kmat_ref = K2mat(intrinsics_ref)
            # scene2view, error = solve_pnp(xy_ref, xyz_corresp, Kmat_ref)

            # solve procustes
            # find the scale factor from ref to target
            # outdir = "database/processed"
            # seqname = "Oct5at10-49AM-poly-0000"
            # depth_path = "%s/Depth/Full-Resolution/%s/%05d.npy" % (outdir, seqname, sel_i)
            # depth_target_zoe = np.load(depth_path).astype(np.float32)
            # depth_target_zoe = cv2.resize(depth_target_zoe, depth_target.shape[::-1])
            # scale_ref_to_trg = np.median(depth_target / depth_target_zoe)
            scale_ref_to_trg = 0.48  # experimentally found
            # depth_target_zoe = depth_target_zoe * scale_ref_to_trg
            # cv2.imwrite("tmp/0.jpg", depth_target * 50)
            # cv2.imwrite("tmp/1.jpg", depth_target_zoe * 50)
            depth_path = (
                ref_list[i].replace("JPEGImages", "Depth").replace(".jpg", ".npy")
            )
            depth_ref = np.load(depth_path).astype(np.float32)
            depth_ref = cv2.resize(depth_ref, ref_final_size)
            depth_ref = depth_ref * scale_ref_to_trg
            view2scene, error = solve_procrustes(
                depth_ref, intrinsics_ref, xy_ref, xyz_corresp
            )
            scene2view = np.linalg.inv(view2scene)

            # ###TODO register with icp

            # xyz_ref = depth_to_canonical(depth_ref, intrinsics_ref, np.eye(4))
            # xyz_ref = xyz_ref[sil_ref_list[i] == 0]
            # xyz_ref = xyz_ref.reshape(-1, 3)[::100]
            # xyz_ref = xyz_ref @ view2scene[:3, :3].T + view2scene[:3, 3]
            # # xyz_ref = xyz_target.reshape(-1, 3)[::100]
            # print("Apply point-to-point ICP")
            # threshold = 0.1
            # source = o3d.geometry.PointCloud()
            # source.points = o3d.utility.Vector3dVector(xyz_ref)

            # target = o3d.geometry.PointCloud()
            # target.points = o3d.utility.Vector3dVector(renderer_target.mesh.vertices)

            # pdb.set_trace()
            # voxel_size = 0.05  # means 5cm for this dataset
            # source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
            # target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

            # reg_p2p = execute_global_registration(
            #     source_down, target_down, source_fpfh, target_fpfh, voxel_size
            # )

            # reg_p2p = o3d.pipelines.registration.registration_icp(
            #     source,
            #     target,
            #     threshold,
            #     reg_p2p.transformation,
            #     o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            # )
            # view2scene = reg_p2p.transformation
            # print("Transformation is:")
            # print(reg_p2p.transformation)

            # trimesh.Trimesh(source.points).export("tmp/xyz_ref.obj")
            # trimesh.Trimesh(
            #     source.points @ view2scene[:3, :3].T + view2scene[:3, 3]
            # ).export("tmp/xyz_ref_aligned.obj")
            # scene2view = np.linalg.inv(view2scene)

        scene2views.append(scene2view)
        errors.append(error)
        print("localization error: %.4f" % error)

        # visualization
        rgb_ref, depth_ref = renderer_ref.render(
            None, intrinsics=intrinsics_ref, extrinsics=scene2view
        )

        xyz_ref = depth_to_canonical(depth_ref, intrinsics_ref, scene2view)
        rgb_ref = cv2.resize(rgb_ref, ref_final_size)
        xyz_ref = cv2.resize(xyz_ref, ref_final_size)

        xyz_vis_target = (
            (xyz_target - renderer_target.aabb[0])
            / (renderer_target.aabb[1] - renderer_target.aabb[0])
            * 256
        )
        xyz_ref_vis = (
            (xyz_ref - renderer_ref.aabb[0])
            / (renderer_ref.aabb[1] - renderer_ref.aabb[0])
            * 256
        )
        rgb_target = np.concatenate([rgb_target, xyz_vis_target], 0)
        rgb_ref = np.concatenate([rgb_ref, xyz_ref_vis], 0)
        vis_imgs[i] = np.concatenate([vis_imgs[i], rgb_ref, rgb_target], 1)

    mesh = draw_cams(scene2views)
    os.makedirs("tmp/dino", exist_ok=True)
    save_vid("tmp/dino/%s" % seqname, vis_imgs, fps=10)
    mesh.export("tmp/dino/%s.obj" % seqname)
    np.save("tmp/dino/extrinsics-%s.npy" % seqname, scene2views)
    np.save("tmp/dino/errors-%s.npy" % seqname, errors)
    print("video saved to tmp/dino/%s.mp4" % seqname)

    transform_bg_cams(seqname)


if __name__ == "__main__":
    trg_path = sys.argv[1]
    ref_path = sys.argv[2]
    extract_dinov2(trg_path, ref_path)
