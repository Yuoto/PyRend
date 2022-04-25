import sys
import numpy as np
import cv2


RENDERER_ROOT = r'D:\MultimediaIClab\AR\Rendering\PyRend'
sys.path.append(RENDERER_ROOT)
from utiles.transform import batch_skew, skew, cal_gradient2D
# from camera import Camera
import logging
from engine.poseEngine import PoseEngine

class GeneralPoseEngine(PoseEngine):

    def __init__(self, K, is_rigid, est_derivate):
        self.fx = K[0,0]
        self.fy = K[1,1]
        self.cx = K[0,2]
        self.cy = K[1,2]
        self.is_rigid = is_rigid
        # self.tikhonov_alpha = 1e9
        self.tikhonov_alpha = 0
        self.lambda_damp = 5
        self.tikhonov_upbnd = 1e13
        self.stepScale = 1
        self.step_size = 1.
        # self.step_size = 0.1
        self.est_derivate = est_derivate

    def pres_ppi(self, gradient_map, uv, weights=None):
        """
        This version implements the feature loss

        Can vary from different energy function declarations
        (photometric loss): this corresponds to ∂res/∂I * ∂I/∂pi
        (feature loss): this corresponds to ∂res/∂f * ∂f/∂pi
        (2d geo loss): this corresponds to ∂res/∂pi

        [optional] Importance weights \in [0,1], scalar for each residual term

        :param gradient_map: shape (h,w,2) numpy array
        :param uv: shape (M_proj,2) numpy array, matched sampling indices
        :param weights: shape (h,w,1) numpy array, importance weights \in [0,1]
        :return: Jgs: shape (M_proj,d,2) numpy array, ∂res/∂pi
        """
        h, w, d = gradient_map.shape
        # mipMapS = int(np.sqrt(u.shape[0]))

        # [optional] Importance weights
        if weights is None:
            weights = np.ones_like(gradient_map[:, :, 0])[..., np.newaxis]
        gradient_map *= weights

        # Jg_x = np.zeros((h, w, d))
        # Jg_y = np.zeros((h, w, d))
        # Jg_x[:, 1: w - 1] = (feat[:, 2:w] - feat[:, 0:w - 2]) / 2
        # Jg_y[1: h - 1, :] = (feat[2:h, :] - feat[0:h - 2, :]) / 2

        # Sample the valid gradient pixels
        Jgs_x = cv2.remap(gradient_map[:, :, 0],
                          map1=uv[:, 0].astype(np.float32),
                          map2=uv[:, 1].astype(np.float32),
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1.)

        Jgs_y = cv2.remap(gradient_map[:, :, 1],
                          map1=uv[:, 0].astype(np.float32),
                          map2=uv[:, 1].astype(np.float32),
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=1.)

        Jgs = np.stack((Jgs_x, Jgs_y), axis=2)

        return Jgs

    def ppi_pc(self, pc, pyramid_level=0):
        """
        Compute the gradient of 2D image coordinates with respect to 3D camera coordinates
        :param pc: shape (M_proj,3,1) numpy array, M_proj (x,y,z) camera coordinate points
        :return: Jpi: shape (M_proj,2,3) numpy array, ∂pi/∂w
        """
        pyramid_scale = 2 ** (-pyramid_level)

        inv_d = 1. / pc[:, 2]
        num = inv_d.shape[0]
        O = np.zeros((num, 1))
        pc_x = pc[:, 0] * inv_d
        pc_y = pc[:, 1] * inv_d

        Jw_x = self.fx*pyramid_scale * np.concatenate((inv_d, O, -inv_d * pc_x), axis=1)
        Jw_y = self.fy*pyramid_scale * np.concatenate((O, inv_d, -inv_d * pc_y), axis=1)

        Jpi = np.stack((Jw_x, Jw_y), axis=1)

        return Jpi

    def pc_pw(self, Rp, num):
        """
        Compute the gradient of Xc with respect to the Xw
        :param Rp: [3, 3]  numpy array, s.t. Xc = RpXw
        :return: [N, 3, 3]  numpy array, ∂Xc/∂Xw
        """

        return Rp.reshape(1,3,3).repeat(num, axis=0)


    def pw_pxir(self, pc, pm=None, R=None, jl=None):
        """
        Compute the gradient of a SO3 group with respect to the rotational lie algebra [rx,ry,rz] R^3
        :param pc: shape (N,3,1)  numpy array, N world space points
        :return: shape (N,3,3)  numpy array, ∂w/∂ξi
        """

        if self.est_derivate:
            return -batch_skew(pc)
            # return -batch_skew(pc) @ jl[np.newaxis, :]
        else:
            return self.pw_pR(pm) @ self.pR_pxir(R)

    def pR_pxir(self, R):
        """
       Compute the gradient of R with respect to the rotational lie algebra in R^3
       :param R: shape (N,3,3)  numpy array, N Rotation matrices
       :return: shape (N,9,3)  numpy array, ∂R/∂xir
       """
        r = cv2.Rodrigues(R.squeeze())[0]
        out2 = cv2.Rodrigues(r)[1].T[np.newaxis,:]

        # ==== from dodeca
        # out3 = np.zeros((9,3))
        # r = r.squeeze()
        # rx, ry, rz = r[0], r[1], r[2]
        #
        # l2 = rx * rx + ry * ry + rz * rz
        # l = np.sqrt(l2)
        # l3 = l2 * l
        # l4 = l2 * l2
        # rx2ry2 = rx * rx + ry * ry
        # rx2rz2 = rx * rx + rz * rz
        # ry2rz2 = ry * ry + rz * rz
        # cl = np.cos(l)
        # sl = np.sin(l)
        # cl1 = cl - 1
        # rxsl = rx * sl
        # rysl = ry * sl
        # rzsl = rz * sl
        # rxcl = rx * cl
        # rycl = ry * cl
        # rzcl = rz * cl
        # sl_l = sl / l
        # rx_l2 = rx / l2
        # ry_l2 = ry / l2
        # rz_l2 = rz / l2
        # rx_l3 = rx / l3
        # ry_l3 = ry / l3
        # rz_l3 = rz / l3
        # cl1_l4 = cl1 / l4
        # rxcl1_l4 = rx * cl1_l4
        # rycl1_l4 = ry * cl1_l4
        # rzcl1_l4 = rz * cl1_l4
        # out3[0, 0] = -(sl * ry2rz2 * rx_l3) - (2 * ry2rz2 * rxcl1_l4)
        # out3[0, 1] = (2 * cl1 * ry_l2) - (sl * ry2rz2 * ry_l3)- (2 * ry2rz2 * rycl1_l4)
        # out3[0, 2] = (2 * cl1 * rz_l2) - (sl * ry2rz2 * rz_l3) - (2 * ry2rz2 * rzcl1_l4)
        # out3[1, 0] = (rzsl * rx_l3) - (rzcl * rx_l2) - (cl1 * ry_l2)+ (rx * rysl * rx_l3) + (2 * rx * ry * rxcl1_l4)
        # out3[1, 1] = (rzsl * ry_l3) - (rzcl * ry_l2) - (cl1 * rx_l2)+ (rx * rysl * ry_l3) + (2 * rx * ry * rycl1_l4)
        # out3[1, 2] = (rzsl * rz_l3) - (rzcl * rz_l2) - sl_l+ (rx * rysl * rz_l3) + (2 * rx * ry * rzcl1_l4)
        # out3[2, 0] = (rycl * rx_l2) - (cl1 * rz_l2) - (rysl * rx_l3)+ (rx * rzsl * rx_l3) + (2 * rx * rz * rxcl1_l4)
        # out3[2, 1] = sl_l + (rycl * ry_l2) - (rysl * ry_l3)+ (rx * rzsl * ry_l3) + (2 * rx * rz * rycl1_l4)
        # out3[2, 2] = (rycl * rz_l2) - (cl1 * rx_l2) - (rysl * rz_l3)+ (rx * rzsl * rz_l3) + (2 * rx * rz * rzcl1_l4)
        # out3[3, 0] = (rzcl * rx_l2) - (cl1 * ry_l2) - (rzsl * rx_l3)+ (rx * rysl * rx_l3) + (2 * rx * ry * rxcl1_l4)
        # out3[3, 1] = (rzcl * ry_l2) - (cl1 * rx_l2) - (rzsl * ry_l3)+ (rx * rysl * ry_l3) + (2 * rx * ry * rycl1_l4)
        # out3[3, 2] = sl_l + (rzcl * rz_l2) - (rzsl * rz_l3)+ (rx * rysl * rz_l3) + (2 * rx * ry * rzcl1_l4)
        # out3[4, 0] = (2 * cl1 * rx_l2) - (sl * rx2rz2 * rx_l3)- (2 * rx2rz2 * rxcl1_l4)
        # out3[4, 1] = -(sl * rx2rz2 * ry_l3)- (2 * rx2rz2 * rycl1_l4)
        # out3[4, 2] = (2 * cl1 * rz_l2) - (sl * rx2rz2 * rz_l3)- (2 * rx2rz2 * rzcl1_l4)
        # out3[5, 0] = (rxsl * rx_l3) - (rxcl * rx_l2) - sl_l+ (ry * rzsl * rx_l3) + (2 * ry * rz * rxcl1_l4)
        # out3[5, 1] = (rxsl * ry_l3) - (rxcl * ry_l2) - (cl1 * rz_l2)+ (ry * rzsl * ry_l3) + (2 * ry * rz * rycl1_l4)
        # out3[5, 2] = (rxsl * rz_l3) - (rxcl * rz_l2) - (cl1 * ry_l2)+ (ry * rzsl * rz_l3) + (2 * ry * rz * rzcl1_l4)
        # out3[6, 0] = (rysl * rx_l3) - (rycl * rx_l2) - (cl1 * rz_l2)+ (rx * rzsl * rx_l3) + (2 * rx * rz * rxcl1_l4)
        # out3[6, 1] = (rysl * ry_l3) - (rycl * ry_l2) - sl_l+ (rx * rzsl * ry_l3) + (2 * rx * rz * rycl1_l4)
        # out3[6, 2] = (rysl * rz_l3) - (rycl * rz_l2) - (cl1 * rx_l2)+ (rx * rzsl * rz_l3) + (2 * rx * rz * rzcl1_l4)
        # out3[7, 0] = sl_l + (rxcl * rx_l2) - (rxsl * rx_l3)+ (ry * rzsl * rx_l3) + (2 * ry * rz * rxcl1_l4)
        # out3[7, 1] = (rxcl * ry_l2) - (cl1 * rz_l2) - (rxsl * ry_l3)+ (ry * rzsl * ry_l3) + (2 * ry * rz * rycl1_l4)
        # out3[7, 2] = (rxcl * rz_l2) - (cl1 * ry_l2) - (rxsl * rz_l3)+ (ry * rzsl * rz_l3) + (2 * ry * rz * rzcl1_l4)
        # out3[8, 0] = (2 * cl1 * rx_l2) - (sl * rx2ry2 * rx_l3)- (2 * rx2ry2 * rxcl1_l4)
        # out3[8, 1] = (2 * cl1 * ry_l2) - (sl * rx2ry2 * ry_l3)- (2 * rx2ry2 * rycl1_l4)
        # out3[8, 2] = -(sl * rx2ry2 * rz_l3) - (2 * rx2ry2 * rzcl1_l4)
        # assert np.allclose(out3, out2)

        return out2

    def pw_pR(self, pm):
        """
        Compute the gradient of pc with respect to the rotational lie group SO3 in R^9
         __________________________
        |xm ym zm 0  0  0  0  0  0 |
        |0  0  0  xm ym zm 0  0  0 |
        |0  0  0  0  0  0  xm ym zm| _3x9
        :param pm: shape (N,3,1)  numpy array, N world space points
        :return: shape (N,3,9)  numpy array, ∂w/∂R
        """
        N = pm.shape[0]
        out = np.zeros((N,3,9))
        pm_squeeze = pm.squeeze()
        out[:, 0, :3] = pm_squeeze
        out[:, 1, 3:6] = pm_squeeze
        out[:, 2, 6:9] = pm_squeeze

        return out

    def pw_pxit(self, num):
        """
         Compute the gradient of a SO3 group with respect to the rotational lie algebra [tx,ty,tz] R^3
        Return constant scalar 1
        """


        return np.eye(3).reshape(1, 3, 3).repeat(num, axis=0)
        # return 1




    # ==== PEs for Computing Gradient
    def compute_jacob_2d_geo(self, feat, res, offset, pc, R, pw):
        pass

    def compute_jacob_depth(self):
        pass

    def compute_jacob_feat(self, feat, pc_opt, ps_proj, match_mask, proj_mask, pyramid_idx, bilinear_sampling, pm_opt, R, jl):
        """

        :param feat: shape (h,w,d) numpy array, feature map (for d=3, can be an image)
        :param pc_opt: shape (N,3,1) numpy array, N to be optimized camera space coordinates (x,y,z)
        :param ps_proj: shape (M_proj,2) numpy array, M_proj valid image space coordinates (u,v)
        :param match_mask: shape (N) numpy array, N \in [0,1] with M ones (filter out nodes that don't contrib to residual)
        :param proj_mask: shape (M) numpy array, M \in [0,1] with M_val ones (filter out nodes that is out of screen)
        :return:
            Jxir: shape (NxD,3) numpy array, ∂feat/∂ξi
            Jxit: shape (NxD,3) numpy array, ∂feat/∂ti
        """
        assert len(feat.shape) == 3, "feat: shape (h,w,d) numpy array, feature map (for d=3, can be an image)"
        assert len(pc_opt.shape) == 3, "pc_opt: shape (N,3,1) numpy array, N to be optimized camera space coordinates"
        assert len(ps_proj.shape) == 2, "ps_proj: shape (M_proj,2) numpy array, M_proj valid image space coordinates (u,v)"
        assert len(match_mask.shape) == 1, "match_mask: shape (N) numpy array, N \in [0,1] with M ones"

        # pc_match: shape (M,3,1) numpy array, matched camera coordinate for data term
        pc_match = pc_opt[match_mask]
        pc_proj = pc_match[proj_mask]


        if pm_opt is not None:
            assert len(
                pm_opt.shape) == 3, "pm_opt: shape (N,3,1) numpy array, N to be optimized model space coordinates"
            pm_match = pm_opt[match_mask]
            pm_proj = pm_match[proj_mask]
        else:
            pm_proj = None

        h, w, d = feat.shape
        N = pc_opt.shape[0]
        M = pc_match.shape[0]
        M_proj = ps_proj.shape[0]

        assert proj_mask.shape[0] == M, "number of projection mask, should be the same as pc_match (i.e. M)"
        assert pc_proj.shape[0] == M_proj, "number of projected pc, should be the same as ps_proj (i.e. M_proj)"

        # Jpres_pw: shape (M_proj,d,2)@(M_proj,2,3)=(M_proj,d,3) numpy array:
        Jpres_pw = self.pres_ppi(feat, ps_proj,bilinear_sampling=bilinear_sampling) @ self.ppi_pw(pc_proj, pyramid_idx)

        if self.is_rigid:
            # [Rigid]
            # Jxir: (M_proj,d,3)@(M_proj,3,3)=(M_proj,d,3)  reshape -> (M_proj x d, 3) numpy array
            # Jxir = (Jpres_pw @ self.pw_pxir(pc_proj)).reshape(M_proj * d, 3)
            Jxir = (Jpres_pw @ self.pw_pxir(pc_proj, pm=pm_proj, R=R, jl=jl)).reshape(M_proj * d, 3)
            Jxit = Jpres_pw.reshape(M_proj * d, 3)

        else:
            # [Non-rigid]
            # Jpres_pw: shape (M_proj,N,d,3) numpy array
            # Jxir: (M_proj,N,d,3)@(N,3,3)=(M_proj,N,d,3) swap & reshape -> (Mxd, Nx3) numpy array
            # Jxit: (M_proj,N,d,3) swap & reshape -> (Mxd, Nx3) numpy array

            # TODO: let the gradients to be zero for dim = 1

            Jpres_pw = Jpres_pw[:, np.newaxis, :, :].repeat(N, axis=1)
            Jxir = (Jpres_pw @ self.pw_pxir(pc_opt)).swapaxes(1, 2).reshape(M_proj * d, N * 3)
            Jxit = Jpres_pw.swapaxes(1, 2).reshape(M_proj * d, N * 3)

        return Jxir, Jxit

    def compute_jacob_reg(self):
        pass

    def compute_jacob_all(self):
        """
        Total jacobian with balanced weights
        :return:
        """
        pass


    # ==== PEs for transformations
    def project(self, h, w, pc, offset, pyramid_idx):
        """
        Project the camera coordinates into image plane, but with offset shifted!
        Also, generating a mask for valid pixels (i.e. valid camera coordinates that do not lie out of the cropped image size)
        :param h: scalar, feature height
        :param w: scalar, feature width
        :param pc: shape (M,3,1) numpy array, M camera space coordinates
        :param offset: shape (2) numpy array, (x,y) valid region offset
        :return:
            uv: shape (M,2) numpy array (dtype:float), sampling indices
            mask: shape (M) numpy array (dtype:boolean), M \in [True, False] which filters out invalid pixels (out of frame)
        """
        pyramid_scale = 2**(-pyramid_idx)

        u = ((pc[:, 0, 0] / pc[:, 2, 0]) * self.fx*pyramid_scale + self.cx*pyramid_scale - offset[0])
        v = ((pc[:, 1, 0] / pc[:, 2, 0]) * self.fy + self.cy*pyramid_scale - offset[1])

        mask = self.checkValidPix(h, w, u, v)
        uv = np.stack((u, v), axis=1)
        return uv, mask

    def checkValidPix(self, h, w, u, v):
        mask_u = np.logical_and(u >= 0, u < w)
        mask_v = np.logical_and(v >= 0, v < h)
        mask = np.logical_and(mask_u, mask_v)
        return mask

    def world2cam(self, pw, R, t):
        """
        Transform N points from world space to camera space
        :param pw: shape (N,3,1)  numpy array, N world space points
        :param R: shape (N,3,3) numpy array, N rotation matrices
        :param t: shape (N,3,1) numpy array, N translation vectors
        :return: shape (N,3,1)  numpy array, N camera space points
        """
        assert len(pw.shape) == 3, "pw: shape (N,3,1) numpy array, N world space coordinates"
        assert len(R.shape) == 3, "R: shape (N,3,3) numpy array, N rotation matrices"
        assert len(t.shape) == 3, "t: shape (N,3,1) numpy array, N translation vectors"

        return R @ pw + t

    def backproject(self, depth):
        """
        Return the flattened camera coordinate given depth map & camera info
        :param depth:  m x n depth map
        :return: shape (N,3,1)  numpy array, N camera space points
        """
        pc = self.camera.backproject(depth, depth!=0)[...,np.newaxis]

        return pc

    def backproject_points(self, depth, coords, pyramid_idx):
        """

        :param depth: N x 1 depth value
        :param coords: N x 2 pixel coordinate
        :return: pc: N x 3 camera coordinate
        """
        assert  depth.shape[0] == coords.shape[0]
        pyramid_scale = 2 ** (-pyramid_idx)
        constant_x = 1.0 / (self.camera.focal[0]*pyramid_scale)
        constant_y = 1.0 / (self.camera.focal[1]*pyramid_scale)

        pc = np.zeros((len(coords), 3))
        pc[:, 0] = (coords[:, 0] - self.camera.center[0]*pyramid_scale) * depth.squeeze() * constant_x
        pc[:, 1] = (coords[:, 1] - self.camera.center[1]*pyramid_scale) * depth.squeeze() * constant_y
        pc[:, 2] = depth.squeeze()

        return pc

    # ==== PEs for Linear solver
    def solve(self, a, b, method='QR', verbose=True):
        """
        Solver for linear systems
        """
        if verbose:
            print('Cond A: {:e}'.format(np.linalg.cond(a)))

        # x1 = np.linalg.solve(a, b)
        l = np.linalg.cholesky(a)
        y = np.linalg.solve(l,b)
        x = np.linalg.solve(l.T,y)
        # assert np.allclose(x1, x)

        return x

    def dp_iteration(self,
                     iter_func,
                     loss_func,
                     update_x_func,
                     pyramid_idx_list,
                     linesearch=True,
                     verbose=True,
                     **kargs):

        loss_history = []
        dp_history = []

        for py_idx, n in enumerate(range(len(pyramid_idx_list))):
            kargs['pyramid_idx'] = pyramid_idx_list[py_idx]

            # evaluate at dp=zero
            dp, loss, jacobian_data = iter_func(**kargs)
            loss_history.append(loss)
            dp_history.append(dp)

            if linesearch:
                self.step_size, loss = self.backtracking_line_search(f=loss_func,
                                                      dp=dp,
                                                      update_x_func=update_x_func,
                                                      f_x=loss,
                                                      df_x=jacobian_data,
                                                      c=0.5,
                                                      gamma=0.5,
                                                      eps=1e-8,
                                                      verbose=verbose,
                                                       **kargs
                                                      )
            if verbose:
                print('GN iteration: {:d}, loss: {:f}'.format(n, loss))

            update_x_func(dp*self.step_size)
            print('d')

        # choose the one with lowest loss
        #TODO: check if this is required
        best_idx = np.argmin(loss_history)
        best_dp = dp_history[best_idx]

        return best_dp*self.step_size


    def backtracking_line_search(self, f, dp, update_x_func, f_x=None, df_x=None,
                               c=0.5, gamma=0.5, eps=1e-6, verbose=True, **kargs):
        """
        Backtracking linesearch
        f: function
        x: current point
        dp: direction of search
        f_x = f(x) (Optional)
        args: optional arguments to f (optional)
        c, gamma: backtracking parameters
        eps: (Optional) quit if norm of step produced is less than this
        verbose: (Optional) Print lots of info about progress

        Reference: Nocedal and Wright 2/e (2006), p. 37

        Usage notes:
        -----------
        Recommended for Newton methods; less appropriate for quasi-Newton or conjugate gradients
        """

        step = 1.0
        iter_num = 0
        len_dp = np.linalg.norm(dp)

        if f_x is None:
            df_x, f_x = f(**kargs)
            df_x = df_x.sum(axis=0)
            f_x = f_x.sum()

        # # test correctness
        # f_x_dp1 = [f(x * dp/10, **kargs)[1].sum() for x in range(20)]

        assert df_x.T.shape == dp.shape
        assert 0 < c < 1, 'Invalid value of c in backtracking linesearch'
        assert 0 < gamma < 1, 'Invalid value of gamma in backtracking linesearch'

        m = df_x @ dp

        assert m.shape == (1, 1) or m.shape == ()

        if m > 0:
            print('Stop linesearch. Attempted to linesearch uphill')
            return step, f_x


        # Loop until Armijo condition is satisfied
        update_x_func(step * dp)
        f_x_dp = f(**kargs)[1].sum()
        update_x_func(step * -dp)
        f_x_criterion = f_x + c * step * m

        while f_x_dp > f_x_criterion:

            step *= gamma # decrease step
            iter_num += 1

            update_x_func(step * dp)
            f_x_dp = f(**kargs)[1].sum()
            update_x_func(step * -dp)
            f_x_criterion = f_x + c * step * m

            # # linesearch failed
            # if f_x > best_loss:
            #     step=1.
            #     break

            if verbose:
                print('linesearch iteration: {:d}, step: {:4.3f}, f_x_dp: {:1f}, f_x_criterion: {:1f}'.format(iter_num,
                                                                                                              step,
                                                                                                              f_x_dp,
                                                                                                              f_x_criterion))
                if step * len_dp < eps:
                    print('Step is too small, stop')
                    break

        if verbose:
            print('linesearch done')
        return step, f_x_dp


    def line_search(self, total_loss, hessian, b, loss_func, dp, method='QR', **kargs):
        """
        For each line search iteration, find the best lambda
        :param total_loss:
        :param hessian:
        :param b:
        :param loss_func: (callable)
        :param method:
        :param **kargs:
        :return:
        """
        # best_loss = total_loss
        # dp_best = dp
        #
        #
        # for n in range(self.iterationNum):
        #
        #     hessian_total_reg_up = hessian + self.tikhonov_alpha * np.eye(hessian.shape[0])
        #     dp_up = self.solve(hessian_total_reg_up, b, method=method).squeeze()
        #     res_up, _ = loss_func(dp=dp_up, **kargs)
        #     loss_up = res_up.sum()
        #
        #     hessian_total_reg_down = hessian + self.tikhonov_alpha / self.lambda_damp * np.eye(hessian.shape[0])
        #     dp_down = self.solve(hessian_total_reg_down, b, method=method).squeeze()
        #     res_down, _ = loss_func(dp=dp_down, **kargs)
        #     loss_down = res_down.sum()
        #
        #
        #     # if both failed, then increase the tikhonov regularization (more like gradient descent)
        #     if loss_up >= best_loss and loss_down >= best_loss:
        #         if self.tikhonov_alpha >= self.tikhonov_upbnd:
        #             logging.info('[DPR] Iteration stopped: max lambda met')
        #             break
        #
        #         self.tikhonov_alpha *= self.lambda_damp
        #         # for every up-hill, increase the damping term of lambda
        #         #self.lambda_damp *= 2
        #         logging.info('Up scale lambda. No step taken')
        #
        #
        #     # if the Hessian approximation is in the trust-region, then decrease the damping term of lambda (more like Newton's)
        #     elif loss_down < loss_up:
        #         self.tikhonov_alpha /= self.lambda_damp
        #         #self.lambda_damp /= 3.
        #         logging.info('Down scale lambda & step')
        #         # for every up-hill, increase lambda
        #         best_loss = loss_down
        #         dp_best = dp_down
        #
        #     # if the Hessian approximation is not in the trust-region but better than previous loss, then freeze lambda
        #     else:
        #         best_loss = loss_up
        #         logging.info('Freeze lambda & step')
        #         dp_best = dp_up
        #
        #     if __debug__:
        #         res_down, _ = loss_func(dp=dp_best, **kargs)
        #         pass

        # __findBestDamp(self, dp, totalLoss, res, Ja, maskOn):
        # Ermijo criterion E(p+dp) <= E(p) + alpha*t*▽E(p)^T*dp
        iter_num = 0
        E_dp = 0.


        # 1. compute current loss E(p) with d=0
        E_p = totalLoss
        # experimental scale
        alpha = 0.2
        # self.stepScale *= self.stepSizeDamp

        # 2. compute local function gradient
        g = self.compute_local_func_grad(res, Ja, dp)

        # 3. line searching for best step size
        while True:
            
            # Compute loss
            res, _ = loss_func(dp=dp, **kargs)
            E_dp = res.sum()


            # if the damping size is still too large
            b = alpha * g.T.dot(dp)[0]
            E_originLoss = E_p + b
            xtol = np.linalg.norm(dp[3:6]) < self.T_tol
            if xtol:
                self.terminate = True

            if E_dp > E_originLoss and not self.terminate:
                # if True:
                self.__updateFAPose(-dp)
                Epdp = 0.
                # damp the delta pose
                dp *= self.stepSizeDamp
                self.stepScale *= self.stepSizeDamp

                iter_num += 1
                logging.debug('Now LS iter num: {:d}'.format(iter_num))

                continue
            # if the step size is suitable
            else:
                break

        return dp_best





def main():


    H, W, D = 200, 150, 10
    SCR_WIDTH = 1280
    SCR_HEIGHT = 1024
    flirIntrinsic = np.array([
        [2.58397003e+03, 0., 6.80406665e+02],
        [0., 2.59629026e+03, 4.96856500e+02],
        [0., 0., 1.]], dtype=np.float32)
    flirDistortion = np.array(
        [-4.0397944508415173e-01, 6.2053493009322680e-01, 2.5890572913194530e-03, -1.9067252961622230e-03,
         -1.3223649399122224e+00])

    intrinsic = np.zeros((4, 4), dtype=np.float32)
    intrinsic[:3, :3] = flirIntrinsic
    intrinsic[3, 3] = 1
    focal = (0, 0)
    center = (0, 0)
    mcam1 = Camera([SCR_WIDTH, SCR_HEIGHT], focal, center, near=0.001, far=1000)
    mcam1.setIntrinsic(intrinsic)
    mcam1.setDistCoeff(flirDistortion)

    # =====================================
    # Non-rigid params
    # =====================================
    # gpe = GeneralPoseEngine(camera=mcam1, is_rigid=False)
    # M = 250  # matched control points
    # N = 1000  # number of optimized nodes
    # pw = np.random.rand(N, 3, 1)
    # R = np.random.rand(N, 3, 3)
    # t = np.random.rand(N, 3, 1)
    # feat = np.random.rand(H, W, D)
    # res = np.random.rand(M * D, 1)
    # offset = np.random.rand(M, 2)
    # match_mask = np.zeros(N,dtype=bool)
    # match_mask[:M] = True

    # =====================================
    # Rigid params
    # =====================================
    gpe = GeneralPoseEngine(camera=mcam1, is_rigid=True)
    M = 250  # matched control points
    N = 1  # number of optimized nodes
    # recipe for pose engine
    pw = np.random.rand(M, 3, 1)
    R = np.random.rand(N, 3, 3)
    t = np.random.rand(N, 3, 1)
    feat = np.random.rand(H, W, D)
    res = np.random.rand(M * D, 1)
    offset = np.random.rand(M, 2)
    match_mask = np.ones(M,dtype=bool)

    pc_opt = gpe.world2cam(pw, R, t)
    pc_match = pc_opt[match_mask]

    # ps_opt: shape (M,2) numpy array, sampling indices
    ps_opt, proj_mask = gpe.project(H, W, pc_match, offset)
    ps_match = ps_opt[proj_mask]
    Jxir, Jxit = gpe.compute_jacob_feat(feat, offset, pc_opt, ps_proj, match_mask)

    # J_data: (Mxd, Nx6) numpy array
    jacobian_data = np.concatenate((Jxir, Jxit), axis=1)

    a, b = gpe.form_A_b(jacobian_data, res)
    x = gpe.solve(a, b)



if __name__ == '__main__':
    main()
