from __future__ import print_function
import urllib
import urllib.request
import bz2
import os
import numpy as np
import cv2 as cv
from scipy.sparse import lil_matrix
import open3d as o3d
# /**********************************************************/
# /** Retrieve Data from file and puts them into variables **/
# /**********************************************************/

class Bundle_Adjustment:
    def __init__(self, p_cloud):
        self.p_cloud = p_cloud
        self.BA_Process()


    def read_bal_data(self):
        n_cameras      = len(self.p_cloud.images_list)
        n_points       = len(self.p_cloud.landmarks)
        n_observations = sum([len(landmark.pos_2D) for landmark in self.p_cloud.landmarks ])

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i, image in enumerate(self.p_cloud.images_list):
            image.BA_id = i

        # second step
        j = 0
        for landmark in self.p_cloud.landmarks:
            for i, _ in enumerate(landmark.images):
                camera_indices[j] = landmark.images[i].BA_id
                point_indices[j]  = landmark.id
                points_2d[j]      = [landmark.pos_2D[i][0], landmark.pos_2D[i][1]]
                j+=1


        camera_params = np.empty((n_cameras, 10), dtype=np.float64)  
        for i, image in enumerate(self.p_cloud.images_list):
            rVec = cv.Rodrigues( image.absoluteTransformation["rotation"].T )[0].ravel()
            tVec = (- np.linalg.inv(image.absoluteTransformation["rotation"]) @ image.absoluteTransformation["translation"]).ravel()
            camera_params[i] = np.concatenate((np.ravel([rVec, tVec]), np.array([image.cameraMatrix[0][0], image.cameraMatrix[1][1], image.cameraMatrix[0][2], image.cameraMatrix[1][2]])), axis=0)
        camera_params_extrinsic = camera_params[:,  :6]      # rotation avector + translation vector`
        camera_params_intrinsic = camera_params[:, 6:10]     # focal_X, focal_y, c_x, c_y

        points_3d = np.empty((n_points, 3), dtype=np.float64) 
        for i, landmark in enumerate(self.p_cloud.landmarks):
            points_3d[i] = landmark.pos_3D

        return camera_params_extrinsic, camera_params_intrinsic, points_3d, camera_indices, point_indices, points_2d


    def project(self, points, camera_params_extrinsic, camera_params_intrinsic):
        rVecs                   = camera_params_extrinsic[:,  :3]
        tVecs                   = camera_params_extrinsic[:, 3:6] 
        camera_params_intrinsic = [np.matrix([[camera_param_intrinsic[0], 0, camera_param_intrinsic[2]], [0, camera_param_intrinsic[1], camera_param_intrinsic[3]], [0, 0, 1]]) for camera_param_intrinsic in camera_params_intrinsic]
        """Convert 3-D points to 2-D by projecting onto images."""
        project_points    = [cv.projectPoints(pts_3d, rVec, tVec, camera_params, distCoeffs=None)[0]  for pts_3d, rVec, tVec, camera_params in zip (points, rVecs, tVecs, camera_params_intrinsic)]
        project_points    = np.asarray(project_points).reshape(-1, 2)
        return project_points
        

    def fun(self, params, camera_params_intrinsic, n_cameras, n_points, camera_indices, point_indices, points_2d):
        """Compute residuals.
        `params` contains camera parameters and 3-D coordinates.
        """
        camera_params_extrinsic = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params_extrinsic[camera_indices], camera_params_intrinsic[camera_indices])

        return (points_proj - points_2d).ravel()


    def bundle_adjustment_sparsity(self, n_cameras, n_points, camera_indices, point_indices):
        m = camera_indices.size * 2    #(two parameters X and Y)
        n = n_cameras * 6 + n_points * 3
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 6 + s] = 1
            A[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return A


    def draw_pcloud(self, camera_params, pointcloud):
        vis = o3d.visualization.Visualizer()
        vis.create_window("Structure_from_Motion", 1280, 720)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(pointcloud).reshape(-1, 3))
        vis.add_geometry(pcd)

        for i in range (len(camera_params)):   
            Rt, _  = cv.Rodrigues(camera_params[i][ : 3])
            R      = Rt.T
            t      = -R @ camera_params[i][3 : 6]

            temp            = np.hstack((R,    t.reshape(-1, 1)))
            transformation  = np.vstack((temp, [0, 0, 0, 1]    ))

            mesh_img_i = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 0.5).transform(transformation)            
            vis.add_geometry(mesh_img_i)

        """  Launch Visualisation  """
        vis.run()

    
    def update_data(self, res_x, n_cameras):
        camera_params = res_x[:n_cameras*6].reshape(-1, 6)
        for i, image in enumerate(self.p_cloud.images_list):
            """ Retrieve Absolute Transformation of current image """
            Rt, _ = cv.Rodrigues(camera_params[i, :3])
            R     = Rt.T
            t     = - R @ camera_params[i, 3:6]
            image.setAbsolutePose(None, R, t)

        pointcloud = res_x[n_cameras*6:].reshape(-1, 3)
        for i, landmark in enumerate (self.p_cloud.landmarks):
            landmark.set_pos_3D(pointcloud[i])


    # /***************************************************************/
    # /** Retrieve data from the file and split them into variables **/
    # /***************************************************************/
    def BA_Process(self):
        camera_params_extrinsic, camera_params_intrinsic, points_3d, camera_indices, point_indices, points_2d = self.read_bal_data()

        n_cameras = camera_params_extrinsic.shape[0]
        n_points = points_3d.shape[0]

        n = 6 * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        print("n_cameras: {}".format(n_cameras))
        print("n_points: {}".format(n_points))
        print("Total number of parameters: {}".format(n))
        print("Total number of residuals: {}".format(m))


        # /***************************************************************/
        # /** Retrieve data from the file and split them into variables **/
        # /***************************************************************/
        import matplotlib.pyplot as plt
        x0 = np.hstack((camera_params_extrinsic.ravel(), points_3d.ravel()))
        camera_params = x0[:n_cameras*6].reshape(-1, 6)
        pointcloud = x0[n_cameras*6:].reshape(-1, 3)
        # draw_pcloud(camera_params, pointcloud)

        f0 = self.fun(x0, camera_params_intrinsic, n_cameras, n_points, camera_indices, point_indices, points_2d)   
        plt.plot(f0)
        print ("reprojection error mean before BA == ", np.linalg.norm(f0)/len(f0))
        # plt.show()


        """Constrcution d'une matrice binaire représentant la matric jacobienne pour least_square """
        A = self.bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
        import time
        from scipy.optimize import least_squares

        t0 = time.time()
        res = least_squares(self.fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e+1, method='trf',
                            args=(camera_params_intrinsic, n_cameras, n_points, camera_indices, point_indices, points_2d))
        t1 = time.time()
        print("Optimization took {0:.0f} seconds".format(t1 - t0))

        plt.plot(res.fun)
        print ("reprojection error mean after BA == ", np.linalg.norm(res.fun)/len(res.fun))
        # plt.show()

        camera_params = res.x[:n_cameras*6].reshape(-1, 6)
        pointcloud = res.x[n_cameras*6:].reshape(-1, 3)

        self.update_data(res.x, n_cameras)

        # self.draw_pcloud(camera_params, pointcloud)