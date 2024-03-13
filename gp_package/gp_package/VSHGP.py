import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gp
import numpy as np

from gpflow.config import default_jitter
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.models import BayesianModel
from gpflow.models.training_mixins import InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

from copy import deepcopy
from tqdm import trange

tf.config.run_functions_eagerly(True)

class VSHGP(BayesianModel, InternalDataTrainingLossMixin):
    """
    Variational Sparse Hierarchical Gaussian Process (VSHGP) model.
    ----------------
    This class implements the GP model presented in [1] from scratch using GPflow and TensorFlow. This class uses variational approximation methods via inducing points, allowing scalability to large datasets. In addition, the model is designed to handle heteroscedastic noise (i.e., non-constant noise along the input space), which is a common feature of real-world data, by placing an additional GP on the log noise variance.
    
    Attributes:
    ----------------
    data: tuple:
        A tuple containing the input and output data. The input data is a 2D array of shape (n x d), where n is the number of data points and d is the number of input dimensions. The output data is a 2D array of shape (n x 1), where n is the number of data points.
    kernel_f: gpflow.kernels:
        The kernel function for the latent function f_m.
    kernel_g: gpflow.kernels:
        The kernel function for the noise variance function g_u.
    inducing_points_f: np.ndarray:
        The inducing points for the latent function f_m. The array has shape (m x d), where m is the number of inducing points and d is the number of input dimensions.
    inducing_points_g: np.ndarray:
        The inducing points for the noise variance function g_u. The array has shape (u x d), where u is the number of inducing points and d is the number of input dimensions.
    
    Notes:
    ----------------
    Model optimization is done by maximising the evidence lower bound (ELBO). The `fit` method is used to train the model using the Adam optimizer. It uses TensorFlow's automatic differentiation feature to compute the gradients of the loss function with respect to the model's trainable variables.
    
    
    References:
    ----------------
    ..[1] Liu, H., Ong, Y. S., and Cai, J. (2018). Large-scale Heteroscedastic Regression via Gaussian Process. IEEE Transactions on Neural Networks and Learning Systems, 32(2):708-721.
    """
    def __init__(self,
        data,
        kernel_f,
        kernel_g,
        inducing_points_f,
        inducing_points_g):

        super().__init__()
        
        # Convert all data to tensor:
        X, Y = data
        self.X = data_input_to_tensor(X)
        self.Y = data_input_to_tensor(Y)
        
        # Copy kernels
        self.Kf = deepcopy(kernel_f)
        self.Kg = deepcopy(kernel_g)
        
        # Inducing_points
        self.inducing_f = inducingpoint_wrapper(inducing_points_f)
        self.inducing_g = inducingpoint_wrapper(inducing_points_g)
        
        # Handy constants
        self.num_inducing_f = self.inducing_f.shape[0] # number of inducing points for f_m
        self.num_inducing_g = self.inducing_g.shape[0] # number of inducing points for g_u
        self.num_data = self.X.shape[0]                # training size
        
        self.mu0 = gp.Parameter(tf.math.log(1e-4))     
        self.diaglambda = gp.Parameter(0.5 * tf.ones([self.num_data, 1]), transform = tfp.bijectors.Softplus())
        
    def maximum_log_likelihood_objective(self):
        """
        The objective maximum likelihood estimate. In this case, we maximise the evidence lower bound (ELBO).
        """
        
        return self.elbo()
    
    def elbo(self):
        """
        The evidence lower bound (ELBO) for the VSHGP.
        """
        
        # Covariance matrices
        knn_f_diag = self.Kf(self.X, full_cov = False)[:, None]             # n x 1
        knn_g_diag = self.Kg(self.X, full_cov = False)[:, None]             # n x 1
        kmm_f = Kuu(self.inducing_f, self.Kf, jitter = default_jitter())    # m x m
        kuu_g = Kuu(self.inducing_g, self.Kg, jitter = default_jitter())    # u x u
        knm_f = tf.transpose(Kuf(self.inducing_f, self.Kf, self.X))         # n x m
        knu_g = tf.transpose(Kuf(self.inducing_g, self.Kg, self.X))         # n x u
        
        # Cholesky decompositions
        # f decompositions
        Lmm_f = tf.linalg.cholesky(kmm_f)                                                               # m x m - lower triangular matrix
        invLmm_f = tf.linalg.triangular_solve(Lmm_f, tf.eye(self.num_inducing_f, dtype = tf.float64))   # m x m
        Qnn_f_half = tf.matmul(knm_f, invLmm_f, transpose_b = True)                                     # n x m
        diagQnn_f = tf.reduce_sum(tf.multiply(Qnn_f_half, Qnn_f_half), axis = 1)[:, None]               # n x 1
        
        # g decompositions
        Luu_g = tf.linalg.cholesky(kuu_g) # u x u
        invLuu_g = tf.linalg.triangular_solve(Luu_g, tf.eye(self.num_inducing_g, dtype = tf.float64))   # u x u
        Qnn_g_half = tf.matmul(knu_g, invLuu_g, transpose_b = True)                                     # n x u - this should be omega
        diagQnn_g = tf.reduce_sum(tf.multiply(Qnn_g_half, Qnn_g_half), axis = 1)[:, None]               # n x 1
        # Onu_g = tf.matmul(tf.matmul(knu_g, invLuu_g, transpose_b = True), invLuu_g)                   # n x u
        
        # Variational parameters
        # diagslamda -> square root of the diagonal elements of lambda
        diagslambda = tf.sqrt(self.diaglambda)
        
        # Calculation of mu_u and Sigma_u
        mu_u = tf.matmul(knu_g, (self.diaglambda - 0.5), transpose_a = True) + self.mu0 * tf.ones([self.num_inducing_g, 1], dtype = tf.float64)   # u x 1
        
        # stable implementation
        temp = tf.matmul(invLuu_g, tf.multiply(tf.transpose(knu_g), tf.tile(tf.transpose(diagslambda), tf.constant([self.num_inducing_g, 1]))))
        AA = tf.eye(self.num_inducing_g, dtype = tf.float64) + tf.matmul(temp, temp, transpose_b = True)                        # u x u
        invL_AA = tf.linalg.triangular_solve(tf.linalg.cholesky(AA), tf.eye(self.num_inducing_g, dtype = tf.float64))           # u x u
        invL_KLambda = tf.matmul(invLuu_g, invL_AA, transpose_a = True, transpose_b = True)                                     # u x u - an upper triangular
        temp = tf.matmul(kuu_g, invL_KLambda)                                                               # u x u
        Sigma_u = tf.matmul(temp, temp, transpose_b = True)                                                 # u x u
        
        # tricks for stable Cholesky decomposition
        s2 = tf.math.abs(tf.math.reduce_mean(tf.linalg.diag_part(Sigma_u)))
        L_Sigma_u = tf.linalg.cholesky(Sigma_u + tf.eye(self.num_inducing_g, dtype = tf.float64) * s2 * default_jitter())
        
        # calculation of mu_g and Sigma_g
        # \int p(g|g_u) q(g_u) dg_u = N(g|mu_g, Sigma_g) - under eq 10b in the paper
        mu_g = tf.matmul(Qnn_g_half, tf.matmul(Qnn_g_half, (self.diaglambda - 0.5), transpose_a = True)) + self.mu0 * tf.ones([self.num_data, 1], dtype = tf.float64) # n x 1
        temp = tf.matmul(knu_g, invL_KLambda)                                                                                           # n x u
        diagSigma_g = knn_g_diag - diagQnn_g + tf.reduce_sum(tf.multiply(temp, temp), axis = 1)[:, None]                                         # n x 1
        
        # Calculation of R_g
        diagR_g = tf.math.exp(mu_g - 0.5 * diagSigma_g) # n x 1, the diagonal elements of R_g, R_g = diag(diagR_g)
        diagsR_g = tf.sqrt(diagR_g)                     # n x 1
        diagInvR_g = 1. / diagR_g                       # n x 1, the diagonal elements of R_g, invR_g = diag(diagInvR_g)
        
        # Precomputations
        invRgKnmf = tf.multiply(tf.tile(diagInvR_g, tf.constant([1, self.num_inducing_f])), knm_f) # n x m

        # Calculation of K_{Lambda}^{-1} # this whole section needs a revamp
        BB = tf.eye(self.num_inducing_f, dtype = tf.float64) + tf.matmul(tf.matmul(invLmm_f, (tf.matmul(knm_f, invRgKnmf, transpose_a = True))), invLmm_f, transpose_b = True)  # m x m
        L_BB = tf.linalg.cholesky(BB)                                                                                               # m x m, lower triangular matrix
        
        # L_BB tends to go to NaNs, this needs careful treatment, as it seems to be very unstable.
        
        invL_BB = tf.linalg.triangular_solve(L_BB, tf.eye(self.num_inducing_f, dtype = tf.float64))                                 # m x m, lower triangular matrix
        L_KR = tf.matmul(Lmm_f, L_BB)                                                                                               # m x m, lower triangular matrix
        invL_KR = tf.matmul(invLmm_f, invL_BB, transpose_a = True, transpose_b = True)                                              # m x m, upper triangular matrix                                                             # m x m

        invRgKnmfinvKR_half = tf.matmul(invRgKnmf, invL_KR)
        
        # Compute ELBO
        # bound = F1 + F2 + F3 + F4
        # F1 = logN(y|0, Q_nn^f + Rg)
        # term1 = - 0.5 * log|Qnn_f + R_g| => scalar
        term1 = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lmm_f))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_KR))) - tf.reduce_sum(tf.math.log(diagsR_g)) # scalar
        term2 = -0.5 * tf.matmul(tf.math.pow(tf.transpose(self.Y), 2), diagInvR_g) # scalar
        term2 = tf.reduce_sum(term2) # the loss function is expecting shape as [], not [1, 1]
        A0 = tf.matmul(self.Y, invRgKnmfinvKR_half, transpose_a = True) # 1 x m
        term3 = 0.5 * tf.matmul(A0, A0, transpose_b = True)
        term3 = tf.reduce_sum(term3)
        F1 = tf.cast(-0.5 * self.num_data * tf.math.log(2 * np.pi), tf.float64) + term1 + term2 + term3
        
        # F2 = - 0.25 * Tr(Sigma_g)
        F2 = -0.25 * tf.reduce_sum(diagSigma_g)

        # F3 = - 0.5 * Tr(R_g^{-1}*(K_nn^f - Q_nn^f))
        F3 = -0.5 * tf.reduce_sum(tf.multiply((knn_f_diag - diagQnn_f), diagInvR_g))
        
        # F4 = -KL{N(g|mu_u, Sigma_u) || N(g_u|0, K_uu^g)}
        # (u x u) * (u x 1)
        temp = tf.matmul(invLuu_g,  (mu_u - self.mu0))   # u x 1 
        invL1L0 = tf.matmul(invLuu_g , L_Sigma_u)   # u x u
        F4 = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Luu_g))) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_Sigma_u))) + 0.5 * self.num_inducing_g - 0.5 * tf.reduce_sum(tf.multiply(invL1L0, invL1L0)) - 0.5 * tf.matmul(temp, temp, transpose_a = True)        
        F4 = tf.reduce_sum(F4)
                
        return F1 + F2 + F3 + F4
        
    def fit(self, params, compile: bool = False, progress_bar = True):
        """
        Fit the VSHGP using an Adam optimizer.
        Args:
            :params: dict: A dictionary containing the fitting parameters:
                :learning_rate - the Adam learning rate
                :no_steps - the number of optimization steps.
            :compile: bool, optional: Whether to compile the function for speed. Defaults to Flase.
        """
        self.objective_evals = []
        opt = tf.optimizers.Adam(learning_rate= params["learning_rate"])
        
        # compile the function for speed
        if compile:
            objective = tf.function(self.training_loss)
        else:
            objective = self.training_loss
            
        if progress_bar:
            pbar = trange(params["no_steps"])
        else: 
            pbar = range(params["no_steps"])
            
        for i in pbar:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                loss = objective()
                self.objective_evals.append(loss.numpy())
            
            grads = tape.gradient(loss, self.trainable_variables)
            opt.apply_gradients(zip(grads, self.trainable_variables))
            if i % params["log_interval"] == 0:
                if progress_bar:
                    pbar.set_postfix(loss = loss.numpy())
                    
    def predict_f(self, xtest):
        """
        Predict the mean and variance of the latent functions.
        """
        knn_g_diag = self.Kg(self.X, full_cov = False)[:, None]             # n x 1
        kmm_f = Kuu(self.inducing_f, self.Kf, jitter = default_jitter())    # m x m
        kuu_g = Kuu(self.inducing_g, self.Kg, jitter = default_jitter())    # u x u
        knm_f = tf.transpose(Kuf(self.inducing_f, self.Kf, self.X))         # n x m
        knu_g = tf.transpose(Kuf(self.inducing_g, self.Kg, self.X))         # n x u
        
        Ksm_f = tf.transpose(Kuf(self.inducing_f, self.Kf, xtest))          # s x m
        Ksu_g = tf.transpose(Kuf(self.inducing_g, self.Kg, xtest))          # s x u
        Kss_f_diag = self.Kf(xtest, full_cov = False)[:, None]              # s x 1
        Kss_g_diag = self.Kg(xtest, full_cov = False)[:, None]              # s x 1
        
        # --------------------------------------------------------------------
        #                       Re-computations
        # --------------------------------------------------------------------
        
        # Cholesky decompositions
        # f decompositions
        Lmm_f = tf.linalg.cholesky(kmm_f)                                                               # m x m - lower triangular matrix
        invLmm_f = tf.linalg.triangular_solve(Lmm_f, tf.eye(self.num_inducing_f, dtype = tf.float64))   # m x m
        
        # g decompositions
        Luu_g = tf.linalg.cholesky(kuu_g)                                                               # u x u - lower triangular
        invLuu_g = tf.linalg.triangular_solve(Luu_g, tf.eye(self.num_inducing_g, dtype = tf.float64))   # u x u - lower triangular
        invKuu_g = tf.linalg.inv(kuu_g)                                                                 
        
        Qnn_g_half = tf.matmul(knu_g, invLuu_g, transpose_b = True)                                     # n x u - this should be omega
        diagQnn_g = tf.reduce_sum(tf.multiply(Qnn_g_half, Qnn_g_half), axis = 1)[:, None]               # n x 1
        
        # Variational parameters
        diagslambda = tf.sqrt(self.diaglambda)
        
        # Calculation of mu_u and Sigma_u
        mu_u = tf.matmul(knu_g, (self.diaglambda - 0.5), transpose_a = True) + self.mu0 * tf.ones([self.num_inducing_g, 1], dtype = tf.float64)   # u x 1
        
        # stable implementation - refer to the MATLAB code for details
        temp = tf.matmul(invLuu_g, tf.multiply(tf.transpose(knu_g), tf.tile(tf.transpose(diagslambda), tf.constant([self.num_inducing_g, 1]))))
        AA = tf.eye(self.num_inducing_g, dtype = tf.float64) + tf.matmul(temp, temp, transpose_b = True)                        # u x u
        invL_AA = tf.linalg.triangular_solve(tf.linalg.cholesky(AA), tf.eye(self.num_inducing_g, dtype = tf.float64))           # u x u
        invL_KLambda = tf.matmul(invLuu_g, invL_AA, transpose_a = True, transpose_b = True)                                     # u x u - an upper triangular
        temp = tf.matmul(kuu_g, invL_KLambda)                                                               # u x u
        Sigma_u = tf.matmul(temp, temp, transpose_b = True)                                                 # u x u
        
        # tricks for stable Cholesky decomposition
        s2 = tf.math.abs(tf.math.reduce_mean(tf.linalg.diag_part(Sigma_u)))
        L_Sigma_u = tf.linalg.cholesky(Sigma_u + tf.eye(self.num_inducing_g, dtype = tf.float64) * s2 * default_jitter())
        
        # calculation of mu_g and Sigma_g
        mu_g = tf.matmul(Qnn_g_half, tf.matmul(Qnn_g_half, (self.diaglambda - 0.5), transpose_a = True)) + self.mu0 * tf.ones([self.num_data, 1], dtype = tf.float64) # n x 1
        temp = tf.matmul(knu_g, invL_KLambda)                                                                                           # n x u
        diagSigma_g = knn_g_diag - diagQnn_g + tf.reduce_sum(tf.multiply(temp, temp), axis = 1)[:, None]                                # n x 1
        
        # Calculation of R_g
        diagR_g = tf.math.exp(mu_g - 0.5 * diagSigma_g)      # n x 1, the diagonal elements of R_g, R_g = diag(diagR_g)
        diagInvR_g = 1. / diagR_g                            # n x 1, the diagonal elements of R_g, invR_g = diag(diagInvR_g)
        
        # Precalculations
        invRgKnmf = tf.multiply(tf.tile(diagInvR_g, tf.constant([1, self.num_inducing_f])), knm_f) # n x m
        
        # precomputations matrices
        BB = tf.eye(self.num_inducing_f, dtype = tf.float64) + tf.matmul(tf.matmul(invLmm_f, (tf.matmul(knm_f, invRgKnmf, transpose_a = True))), invLmm_f, transpose_b = True)  # m x m
        L_BB = tf.linalg.cholesky(BB)                                                                                               # m x m, lower triangular matrix
        invL_BB = tf.linalg.triangular_solve(L_BB, tf.eye(self.num_inducing_f, dtype = tf.float64))                                 # m x m, lower triangular matrix
        invL_KR = tf.matmul(invLmm_f, invL_BB, transpose_a = True, transpose_b = True)                                              # m x m, upper triangular matrix

        invRgKnmfinvKR_half = tf.matmul(invRgKnmf, invL_KR)
        
        # predictive mean
        mu_star = tf.matmul(tf.matmul(Ksm_f, tf.matmul(invL_KR, invRgKnmfinvKR_half, transpose_b = True)), self.Y)
        
        # predictive self-variance
        term1 = tf.matmul(Ksm_f, invLmm_f, transpose_b = True)
        term2 = tf.matmul(Ksm_f, invL_KR)
        var_fs = Kss_f_diag - tf.reduce_sum(tf.multiply(term1, term1), axis = 1)[:, None] + tf.reduce_sum(tf.multiply(term2, term2), axis = 1)[:, None]
        
        mu_gs = tf.matmul(tf.matmul(Ksu_g, invKuu_g), (mu_u - self.mu0)) + self.mu0                    # s x 1
        term1 = tf.matmul(Ksu_g, invLuu_g, transpose_b = True)                                         # s x u
        term2 = tf.matmul(tf.matmul(Ksu_g, invKuu_g), L_Sigma_u)                                       # s x u
        var_gs = Kss_g_diag - tf.reduce_sum(tf.multiply(term1, term1), axis = 1)[:, None] + tf.reduce_sum(tf.multiply(term2, term2), axis = 1)[:, None]
        
        var_star = var_fs + tf.math.exp(mu_gs + 0.5 * var_gs) # s x 1
        
        return mu_star, var_star