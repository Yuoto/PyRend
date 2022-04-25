import abc
from abc import abstractmethod


class PoseEngine(abc.ABC):

    # @property
    # @abstractmethod
    # def memory_for_nodes(self):
    #     pass

    # ==== PEs for Computing Gradient
    @abstractmethod
    def compute_jacob_2d_geo(self):
        pass

    @abstractmethod
    def compute_jacob_depth(self):
        pass

    @abstractmethod
    def compute_jacob_feat(self):
        pass

    @abstractmethod
    def compute_jacob_reg(self):
        pass

    @abstractmethod
    def compute_jacob_all(self):
        """
        Total jacobian with balanced weights
        :return:
        """


    @abstractmethod
    def pres_ppi(self, *args):
        """
        Can vary from different energy function declarations
        (photometric loss): this corresponds to ∂res/∂I * ∂I/∂pi
        (feature loss): this corresponds to ∂res/∂f * ∂f/∂pi
        (2d geo loss): this corresponds to ∂res/∂pi

        [optional] Importance weights \in [0,1], scalar for each residual term
        """

    @abstractmethod
    def ppi_pc(self, fx, fy, px, py, pz):
        """
         Compute the gradient of 2D image coordinates with respect to 3D camera coordinates
         Return 2x3  matrix
        """

    @abstractmethod
    def pc_pw(self, Rp, num):
        """
        Compute the gradient of Xc with respect to the Xw
        :param Rp: [3, 3]  numpy array, s.t. Xc = RpXw
        :return: [N, 3, 3]  numpy array, ∂Xc/∂Xw
        """

    @abstractmethod
    def pw_pxir(self, R, pm):
        """
         Compute the gradient of a SO3 group with respect to the rotational lie algebra [rx,ry,rz] R^3
        Return 3x3 skew-symmetric matrix
        """

    @abstractmethod
    def pw_pxit(self):
        """
         Compute the gradient of a SO3 group with respect to the rotational lie algebra [tx,ty,tz] R^3
        Return 3x3 Identity
        """

    # ==== PEs for transformations
    @abstractmethod
    def world2cam(self):
        pass

    # ==== PEs for Linear solver
    @abstractmethod
    def solve(self, A, b):
        """
        Solver for linear systems
        """