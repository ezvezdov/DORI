# TODO: make dataset_wrapper package
# from os import path
# import sys
# sys.path.append(path.abspath('/home/ezvezdov/Programming/Work/Dataset-Wrapper'))
import dataset_wrapper as dw
import visualizer as vis

import numpy as np
import torch
import open3d as o3d

from sklearn.cluster import DBSCAN
from pytorch3d.ops.knn import knn_points

DATASET = dw.WAYMO_NAME
dataset_path ="/home/ezvezdov/Programming/Work/RCI/login3_home/datasets/waymo2/"

dawr = dw.DatasetWrapper(DATASET, dataset_path)


def remove_ground(points,x_max=35,x_min=-35,y_max=35,y_min=-35):
    '''
        ### Ground removal used in benchmarks for method validation
    '''
    return points[(points[..., 0] > x_min) & (points[..., 0] < x_max) &
                    (points[..., 1] > y_min) & (points[..., 1] < y_max) &
                    (points[..., 2] > 0.3) & (points[..., 2] < 2.)] 

if __name__ == '__main__':
    np.random.seed(10000)

    local_pts = []
    scene = 1
    from_frame = 5
    till_frame = 7

    

    
    
    items_list = [dawr.get_item(scene, frame) for frame in range(from_frame, till_frame)]
    local_pts = []

    for item in items_list:
        points = item["coordinates"]
        pose = item["transformation_matrix"][0:3,0:3] # cut-off intensity
        
        points = remove_ground(points)
        points = points @ pose.T
        local_pts.append(points)

    global_pts = np.concatenate(local_pts)

    ### DBSCAN clustering with geometrical feature only, can be extended to time domain to connect points from time + 1 and time - 1
    # For spatial-temporal clustering, use time axis as well which is scaled compared to eps
    # Example to connect 1 adjacent timeframe:
    # global_pts[:, 3] = global_pts[:, 3] * (eps - 0.01)
    dbscan = DBSCAN(eps=0.2)
    dbscan_ids = dbscan.fit_predict(global_pts[:, [0,1,2]])
    print(max(dbscan_ids))

    vis.visualize_pcd([global_pts])
    

    ### Chosing object point cloud
    for i in range(0,100):
        object_id = i
        object_pts = global_pts[dbscan_ids == object_id]

        print("object_pts", object_pts)

        vis.visualize_pcd([object_pts])


