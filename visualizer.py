import open3d as o3d

def visualize_pcd(list_of_objects):
    vis_list = []
    for obj in list_of_objects:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj)
        vis_list.append(pcd)

    o3d.visualization.draw_geometries(vis_list)