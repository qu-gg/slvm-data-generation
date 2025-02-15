from __future__ import division
from warnings import warn
import numpy as np
from scipy.sparse import csr_matrix


from pybasicbayes.util.general import objarray
from pylds.lds_messages_interface import info_E_step, info_sample, kalman_info_filter, kalman_filter, E_step

# TODO on instantiating, maybe gaussian states should be resampled
# TODO make niter an __init__ arg instead of a method arg

###########
#  bases  #
###########

class _LDSStates(object):
    def __init__(self, model, T=None, data=None, inputs=None, stateseq=None,
                 initialize_from_prior=False,
                 initialize_to_noise=True):
        self.model = model

        self.T = T if T is not None else data.shape[0]
        self.data = data
        self.inputs = np.zeros((self.T, 0)) if inputs is None else inputs

        self._normalizer = None

        if stateseq is not None:
            self.gaussian_states = stateseq
        elif initialize_from_prior:
            self.generate_states()
        elif initialize_to_noise:
            self.gaussian_states = np.random.normal(size=(self.T, self.D_latent))
        elif data is not None:
            self.resample()
        else:
            raise Exception("Invalid options. Must specify how states are initialized.")

    ### Basics

    def log_likelihood(self):
        if self._normalizer is None:
            self._normalizer, _, _ = kalman_info_filter(*self.info_params)

            # self._normalizer += self._info_extra_loglike_terms(
            #     *self.extra_info_params,
            #     isdiag=self.diagonal_noise)

        return self._normalizer

    def generate_states(self):
        # Generate from the prior and raise exception if unstable
        T, n = self.T, self.D_latent

        gss = np.empty((T,n),dtype='double')
        gss[0] = np.random.multivariate_normal(self.mu_init, self.sigma_init)

        for t in range(1,T):
            gss[t] = self.dynamics_distn.\
                rvs(x=np.hstack((gss[t-1][None,:], self.inputs[t-1][None,:])),
                    return_xy=False)
            assert np.all(np.isfinite(gss[t])), "LDS appears to be unstable!"

        self.gaussian_states = gss

    def generate_obs(self):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        ed = self.emission_distn
        gss = self.gaussian_states
        data = np.empty((T,p),dtype='double')

        for t in range(self.T):
            data[t] = \
                ed.rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                       return_xy=False)

        return data

    def sample_predictions(self, Tpred, inputs=None, states_noise=False, obs_noise=False):
        inputs = np.zeros((Tpred, self.D_input)) if inputs is None else inputs
        _, filtered_mus, filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        init_mu = self.A.dot(filtered_mus[-1]) + self.B.dot(self.inputs[-1])
        init_sigma = self.sigma_states + self.A.dot(
            filtered_sigmas[-1]).dot(self.A.T)

        randseq = np.zeros((Tpred - 1, self.D_latent))
        if states_noise:
            L = np.linalg.cholesky(self.sigma_states)
            randseq += np.random.randn(Tpred - 1, self.D_latent).dot(L.T)

        states = np.empty((Tpred, self.D_latent))
        if states_noise:
            states[0] = np.random.multivariate_normal(init_mu, init_sigma)
        else:
            states[0] = init_mu
            
        for t in range(1, Tpred):
            states[t] = self.A.dot(states[t - 1]) + \
                        self.B.dot(inputs[t - 1]) + \
                        randseq[t - 1]

        obs = states.dot(self.C.T) + inputs.dot(self.D.T)
        if obs_noise:
            L = np.linalg.cholesky(self.sigma_obs)
            obs += np.random.randn(Tpred, self.D_emission).dot(L.T)

        return obs

    ## convenience properties

    @property
    def D_latent(self):
        return self.dynamics_distn.D_out

    @property
    def D_input(self):
        return self.dynamics_distn.D_in - self.dynamics_distn.D_out

    @property
    def D_emission(self):
        return self.emission_distn.D_out

    @property
    def dynamics_distn(self):
        return self.model.dynamics_distn

    @property
    def emission_distn(self):
        return self.model.emission_distn

    @property
    def diagonal_noise(self):
        return self.model.diagonal_noise

    @property
    def mu_init(self):
        return self.model.mu_init

    @property
    def sigma_init(self):
        return self.model.sigma_init

    @property
    def A(self):
        return self.dynamics_distn.A[:, :self.D_latent]

    @property
    def B(self):
        return self.dynamics_distn.A[:, self.D_latent:]

    @property
    def sigma_states(self):
        return self.dynamics_distn.sigma

    @property
    def C(self):
        return self.emission_distn.A[:,:self.D_latent]

    @property
    def D(self):
        return self.emission_distn.A[:, self.D_latent:]

    @property
    def sigma_obs(self):
        return self.emission_distn.sigma

    @property
    def _kwargs(self):
        return dict(super(_LDSStates, self)._kwargs,
                    gaussian_states=self.gaussian_states)

    @property
    def info_init_params(self):
        J_init = np.linalg.inv(self.sigma_init)
        h_init = np.linalg.solve(self.sigma_init, self.mu_init)

        log_Z_init = -1. / 2 * h_init.dot(np.linalg.solve(J_init, h_init))
        log_Z_init += 1. / 2 * np.linalg.slogdet(J_init)[1]
        log_Z_init -= self.D_latent / 2. * np.log(2 * np.pi)

        return J_init, h_init, log_Z_init

    @property
    def info_dynamics_params(self):
        A = self.A
        B = self.B
        Q = self.sigma_states

        # Get the pairwise potentials
        # TODO: Check for diagonal before inverting
        J_pair_22 = np.linalg.inv(Q)
        J_pair_21 = -J_pair_22.dot(A)
        J_pair_11 = A.T.dot(-J_pair_21)

        # Check if diagonal and avoid inverting D_obs x D_obs matrix
        mBTQiA = B.T.dot(J_pair_21)
        BTQi = B.T.dot(J_pair_22)
        h_pair_1 = self.inputs[:-1].dot(mBTQiA)
        h_pair_2 = self.inputs[:-1].dot(BTQi)

        log_Z_pair = -1. / 2 * np.linalg.slogdet(Q)[1]
        log_Z_pair -= self.D_latent / 2. * np.log(2 * np.pi)
        hJh_pair = B.T.dot(np.linalg.solve(Q, B))
        log_Z_pair -= 1. / 2 * np.einsum('ij,ti,tj->t', hJh_pair, self.inputs[:-1], self.inputs[:-1])

        return J_pair_11, J_pair_21, J_pair_22, h_pair_1, h_pair_2, log_Z_pair

    @property
    def info_emission_params(self):
        C = self.C
        centered_data = self.data - self.inputs.dot(self.D.T)

        # Observations
        log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
        if self.diagonal_noise:
            # Use the fact that the diagonal regression prior is factorized
            rsq = self.emission_distn.sigmasq_flat
            RinvC = (1/rsq)[:,None] * C
            J_node = C.T.dot(RinvC)
            h_node = centered_data.dot(RinvC)

            log_Z_node -= 1./2 * np.sum(np.log(rsq))
            log_Z_node -= 1./2 * np.sum(centered_data**2 * 1./rsq, axis=1)

        else:
            Rinv = np.linalg.inv(self.sigma_obs)
            RinvC = Rinv.dot(C)

            J_node = C.T.dot(RinvC)
            h_node = centered_data.dot(RinvC)

            log_Z_node += 1./2 * np.linalg.slogdet(Rinv)[1]
            log_Z_node -= 1./2 * np.einsum('ij,ti,tj->t', Rinv,
                                           centered_data, centered_data)

        return J_node, h_node, log_Z_node

    @property
    def info_params(self):
        return self.info_init_params + self.info_dynamics_params + self.info_emission_params

    def info_filter(self):
        self._normalizer, filtered_Js, filtered_hs = \
            kalman_info_filter(*self.info_params)

        return filtered_Js, filtered_hs

    def kalman_filter(self):
        self._normalizer, filtered_mus, filtered_sigmas = kalman_filter(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        # Update the normalization constant
        # self._gaussian_normalizer += self._info_extra_loglike_terms(
        #     *self.extra_info_params,
        #     isdiag=self.diagonal_noise)
        return filtered_mus, filtered_sigmas

    def smooth(self):
        # Use the info E step because it can take advantage of diagonal noise
        # The standard E step could but we have not implemented it
        self.info_E_step()
        return self.smoothed_mus.dot(self.C.T) + self.inputs.dot(self.D.T)

    ### Expectations
    def E_step(self):
        return self.info_E_step()

    def std_E_step(self):
        self._normalizer, self.smoothed_mus, self.smoothed_sigmas, \
        E_xtp1_xtT = E_step(
            self.mu_init, self.sigma_init,
            self.A, self.B, self.sigma_states,
            self.C, self.D, self.sigma_obs,
            self.inputs, self.data)

        self._set_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def info_E_step(self):
        self._normalizer, self.smoothed_mus, \
        self.smoothed_sigmas, E_xtp1_xtT = \
            info_E_step(*self.info_params)

        self._set_expected_stats(
            self.smoothed_mus, self.smoothed_sigmas, E_xtp1_xtT)

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        # Get the emission stats
        p, n, d, T, inputs, data = \
            self.D_emission, self.D_latent, self.D_input, self.T, \
            self.inputs, self.data

        E_x_xT = smoothed_sigmas + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:, :, None] * self.inputs[:, None, :]
        E_u_uT = self.inputs[:, :, None] * self.inputs[:, None, :]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT, E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0, 2, 1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1].sum(0)

        E_xtp1_xtp1T = E_x_xT[1:].sum(0)
        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        E_xtp1_utT = (smoothed_mus[1:, :, None] * inputs[:-1, None, :]).sum(0)
        E_xtp1_xutT = np.hstack((E_xtp1_xtT, E_xtp1_utT))

        # def is_symmetric(A):
        #     return np.allclose(A, A.T)
        # assert is_symmetric(E_xt_xtT)
        # assert is_symmetric(E_xtp1_xtp1T)

        self.E_dynamics_stats = np.array(
            [E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, self.T - 1])

        # Emission statistics
        E_yyT = np.sum(data**2, axis=0) if self.diagonal_noise else data.T.dot(data)
        E_yxT = data.T.dot(smoothed_mus)
        E_yuT = data.T.dot(inputs)
        E_yxuT = np.hstack((E_yxT, E_yuT))

        self.E_emission_stats = objarray([E_yyT, E_yxuT, E_xu_xuT.sum(0), T])

######################
#  algorithm mixins  #
######################

class _LDSStatesGibbs(_LDSStates):
    def resample(self, niter=1):
        self.resample_gaussian_states()

    def _init_gibbs_from_mf(self):
        raise NotImplementedError  # TODO

    def resample_gaussian_states(self):
        self._normalizer, self.gaussian_states = \
            info_sample(*self.info_params)

class _LDSStatesMeanField(_LDSStates):
    @property
    def expected_info_dynamics_params(self):
        J_pair_22, J_pair_21, J_pair_11, logdet_pair = \
            self.dynamics_distn.meanfield_expectedstats()

        # Compute E[B^T Q^{-1}] and E[B^T Q^{-1} A]
        n = self.D_latent
        E_Qinv = J_pair_22.copy("C")
        E_AT_Qinv = (J_pair_21[:,:n].T).copy("C")
        E_BT_Qinv = (J_pair_21[:,n:].T).copy("C")
        E_AT_Qinv_A = J_pair_11[:n,:n].copy("C")
        E_BT_Qinv_A = J_pair_11[n:,:n].copy("C")
        E_BT_Qinv_B = J_pair_11[n:,n:].copy("C")

        h_pair_1 = (-self.inputs[:-1].dot(E_BT_Qinv_A)).copy("C")
        h_pair_2 = (self.inputs[:-1].dot(E_BT_Qinv)).copy("C")

        log_Z_pair = 1./2 * logdet_pair * np.ones(self.T-1)
        log_Z_pair -= self.D_latent / 2. * np.log(2 * np.pi)
        log_Z_pair -= 1. / 2 * np.einsum('ij,ti,tj->t', E_BT_Qinv_B, self.inputs[:-1], self.inputs[:-1])

        return E_AT_Qinv_A, -E_AT_Qinv, E_Qinv, h_pair_1, h_pair_2, log_Z_pair

    @property
    def expected_info_emission_params(self):
        J_yy, J_yx, J_node, logdet_node = \
            self.emission_distn.meanfield_expectedstats()

        n = self.D_latent
        E_Rinv = J_yy
        E_Rinv_C = J_yx[:,:n].copy("C")
        E_Rinv_D = J_yx[:,n:].copy("C")
        E_CT_Rinv_C = (J_node[:n,:n]).copy("C")
        E_DT_Rinv_C = (J_node[n:,:n]).copy("C")
        E_DT_Rinv_D = (J_node[n:,n:]).copy("C")

        h_node = self.data.dot(E_Rinv_C)
        h_node -= self.inputs.dot(E_DT_Rinv_C)

        log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
        log_Z_node += 1. / 2 * logdet_node

        # E[(y-Du)^T R^{-1} (y-Du)]
        log_Z_node -= 1. / 2 * np.einsum('ij,ti,tj->t', E_Rinv,
                                         self.data, self.data)
        log_Z_node -= 1. / 2 * np.einsum('ij,ti,tj->t', -2*E_Rinv_D,
                                         self.data, self.inputs)
        log_Z_node -= 1. / 2 * np.einsum('ij,ti,tj->t', E_DT_Rinv_D,
                                         self.inputs, self.inputs)

        return E_CT_Rinv_C, h_node, log_Z_node

    @property
    def expected_info_params(self):
        return self.info_init_params + \
               self.expected_info_dynamics_params + \
               self.expected_info_emission_params

    def meanfieldupdate(self):
        self._mf_lds_normalizer, self.smoothed_mus, self.smoothed_sigmas, \
            E_xtp1_xtT = info_E_step(*self.expected_info_params)

        self._set_expected_stats(
            self.smoothed_mus,self.smoothed_sigmas,E_xtp1_xtT)

    def get_vlb(self):
        return self._mf_lds_normalizer

    def meanfield_smooth(self):
        if self.diagonal_noise:
            E_C, _, _, _ = self.emission_distn.mf_expectations
        else:
            ed = self.emission_distn
            _,_,E_C,_ = ed._natural_to_standard(ed.mf_natural_hypparam)
        return np.hstack((self.smoothed_mus, self.inputs)).dot(E_C.T)

####################
#  states classes  #
####################

class LDSStates(
    _LDSStatesGibbs,
    _LDSStatesMeanField):
    pass


class LDSStatesMissingData(_LDSStatesGibbs, _LDSStatesMeanField):
    def __init__(self, model, T=None, data=None, mask=None, **kwargs):
        if mask is not None:
            assert mask.shape == data.shape
            self.mask = mask

        elif (data is not None) and isinstance(data, np.ndarray):
            if np.any(np.isnan(data)):
                warn("data includes NaN's. Treating these as missing data.")
                self.mask = ~np.isnan(data)
                data[np.isnan(data)] = 0
            else:
                self.mask = np.ones_like(data, dtype=bool)
        else:
            self.mask = np.ones((T, model.emission_distn.D_out), dtype=bool)

        super(LDSStatesMissingData, self).__init__(model, T=T, data=data, **kwargs)

    @property
    def info_emission_params(self):
        if self.mask is None:
            return super(LDSStatesMissingData, self).info_emission_params

        if self.diagonal_noise:
            return self._info_emission_params_diag
        else:
            return self._info_emission_params_dense

    @property
    def _info_emission_params_diag(self):
        C, D = self.C, self.D
        sigmasq = self.emission_distn.sigmasq_flat
        J_obs = self.mask / sigmasq
        centered_data = self.data - self.inputs.dot(D.T)

        CCT = np.array([np.outer(cp, cp) for cp in C]).\
            reshape((self.D_emission, self.D_latent ** 2))

        J_node = np.dot(J_obs, CCT)
        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))

        # h_node = y^T R^{-1} C - u^T D^T R^{-1} C
        h_node = (centered_data * J_obs).dot(C)

        log_Z_node = -self.mask.sum(1) / 2. * np.log(2 * np.pi)
        log_Z_node -= 1. / 2 * np.sum(self.mask * np.log(sigmasq), axis=1)
        log_Z_node -= 1. / 2 * np.sum(centered_data ** 2 * J_obs, axis=1)

        return J_node, h_node, log_Z_node

    @property
    def _info_emission_params_dense(self):
        # raise Exception("This must be updated with log normalizers")
        T, D_latent = self.T, self.D_latent
        data, inputs, mask = self.data, self.inputs, self.mask

        C, D, R = self.C, self.D, self.sigma_obs
        centered_data = data - inputs.dot(D.T)

        # Sloowwwwww
        J_node = np.zeros((T, D_latent, D_latent))
        h_node = np.zeros((T, D_latent))
        log_Z_node = np.zeros(T)
        for t in range(T):
            m_t = mask[t].sum()
            if m_t == 0:
                continue

            centered_data_t = centered_data[t][mask[t]]
            C_t = C[mask[t]]
            R_t = R[np.ix_(mask[t], mask[t])]
            Rinv_t = np.linalg.inv(R_t)

            J_node[t] = C_t.T.dot(Rinv_t).dot(C_t)
            h_node[t] = (centered_data_t).dot(Rinv_t).dot(C_t)

            log_Z_node[t] -= m_t / 2. * np.log(2 * np.pi)
            log_Z_node[t] -= 1. / 2 * np.linalg.slogdet(R_t)[1]
            log_Z_node[t] -= 1. / 2 * centered_data_t.dot(Rinv_t).dot(centered_data_t)

        J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))


        return J_node, h_node, log_Z_node

    @property
    def expected_info_emission_params(self):
        n = self.D_latent
        if self.mask is None:
            return super(LDSStatesMissingData, self).expected_info_emission_params

        raise Exception("Mean field for masked data is not implemented correctly. We need to handle "
                        "inputs properly, and we need to ravel E_CCT to make the dot product work.")
        if self.diagonal_noise:
            E_C, E_CDCDT, E_sigmasq_inv, E_logdet_node = self.emission_distn.mf_expectations
            E_C, E_D = E_C[:,:n], E_C[:,n:]
            E_CCT = E_CDCDT[:n, :n]
            J_obs = self.mask * E_sigmasq_inv

            J_node = np.dot(J_obs, E_CCT)
            h_node = (self.data * J_obs).dot(E_C)
            h_node -= (self.inputs.dot(E_D.T) * J_obs).dot(E_C)

            J_node = J_node.reshape((self.T, self.D_latent, self.D_latent))

            log_Z_node = -self.D_emission / 2. * np.log(2 * np.pi) * np.ones(self.T)
            log_Z_node += 1. / 2 * E_logdet_node

            # E[(y-Du)^T R^{-1} (y-Du)]
            # E[yT R^{-1}y  -2 y^T R^{-1} Du + u^T D^T R^{-1} D u ]
            log_Z_node -= 1./2 * (self.data * J_obs).T.dot(self.data)

            log_Z_node -= 1. / 2 * np.einsum('ij,ti,tj->t', -2 * E_Rinv_D,
                                             self.data, self.inputs)
            log_Z_node -= 1. / 2 * np.einsum('ij,ti,tj->t', E_DT_Rinv_D,
                                             self.inputs, self.inputs)

        else:
            raise NotImplementedError("Only supporting diagonal regression class with missing data now")

        return J_node, h_node

    def _set_expected_stats(self, smoothed_mus, smoothed_sigmas, E_xtp1_xtT):
        if self.mask is None:
            return super(LDSStatesMissingData, self).\
                _set_expected_stats(smoothed_mus, smoothed_sigmas, E_xtp1_xtT)

        # Get the emission stats
        p, n, d, T, mask, inputs, data = \
            self.D_emission, self.D_latent, self.D_input, self.T, \
            self.mask, self.inputs, self.data
        E_x_xT = smoothed_sigmas + self.smoothed_mus[:, :, None] * self.smoothed_mus[:, None, :]
        E_x_uT = smoothed_mus[:,:,None] * self.inputs[:,None,:]
        E_u_uT = self.inputs[:,:,None] * self.inputs[:,None,:]

        E_xu_xuT = np.concatenate((
            np.concatenate((E_x_xT,   E_x_uT), axis=2),
            np.concatenate((np.transpose(E_x_uT, (0,2,1)), E_u_uT), axis=2)),
            axis=1)
        E_xut_xutT = E_xu_xuT[:-1].sum(0)

        E_xtp1_xtp1T = E_x_xT[1:].sum(0)
        E_xt_xtT = E_x_xT[:-1].sum(0)
        E_xtp1_xtT = E_xtp1_xtT.sum(0)

        E_xtp1_utT = (smoothed_mus[1:,:,None] * inputs[:-1, None, :]).sum(0)
        E_xtp1_xutT = np.hstack((E_xtp1_xtT, E_xtp1_utT))


        def is_symmetric(A):
            return np.allclose(A, A.T)

        assert is_symmetric(E_xt_xtT)
        assert is_symmetric(E_xtp1_xtp1T)

        self.E_dynamics_stats = np.array(
            [E_xtp1_xtp1T, E_xtp1_xutT, E_xut_xutT, self.T - 1])

        # Emission statistics
        E_ysq = np.sum(data**2 * mask, axis=0)
        E_yxT = (data * mask).T.dot(smoothed_mus)
        E_yuT = (data * mask).T.dot(inputs)
        E_yxuT = np.hstack((E_yxT, E_yuT))
        E_xuxuT_vec = E_xu_xuT.reshape((T, -1))
        E_xuxuT = np.array([np.dot(self.mask[:, i], E_xuxuT_vec).reshape((n+d, n+d))
                          for i in range(p)])
        Tp = np.sum(self.mask, axis=0)

        self.E_emission_stats = objarray([E_ysq, E_yxuT, E_xuxuT, Tp])


class LDSStatesCountData(LDSStatesMissingData, _LDSStatesGibbs):
    def __init__(self, model, data=None, mask=None, **kwargs):
        super(LDSStatesCountData, self). \
            __init__(model, data=data, mask=mask, **kwargs)

        # Check if the emission matrix is a count regression
        from pypolyagamma.distributions import _PGLogisticRegressionBase
        if isinstance(self.emission_distn, _PGLogisticRegressionBase):
            self.has_count_data = True

            # Initialize the Polya-gamma samplers
            import pypolyagamma as ppg
            num_threads = ppg.get_omp_num_threads()
            seeds = np.random.randint(2 ** 16, size=num_threads)
            self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

            # Initialize auxiliary variables, omega
            self.omega = np.ones((self.T, self.D_emission), dtype=np.float)
        else:
            self.has_count_data = False

    @property
    def sigma_obs(self):
        if self.has_count_data:
            raise Exception("Count data does not have sigma_obs")
        return super(LDSStatesCountData, self).sigma_obs

    @property
    def info_emission_params(self):
        if not self.has_count_data:
            return super(LDSStatesCountData, self).info_emission_params

        # Otherwise, use the Polya-gamma augmentation
        # log p(y_{tn} | x, om)
        #   = -0.5 * om_{tn} * (c_n^T x_t + d_n^T u_t + b_n)**2
        #     + kappa * (c_n * x_t + d_n^Tu_t + b_n)
        #   = -0.5 * om_{tn} * (x_t^T c_n c_n^T x_t
        #                       + 2 x_t^T c_n d_n^T u_t
        #                       + 2 x_t^T c_n b_n)
        #     + x_t^T (kappa_{tn} * c_n)
        #   = -0.5 x_t^T (c_n c_n^T * om_{tn}) x_t
        #     +  x_t^T * (kappa_{tn} - d_n^T u_t * om_{tn} -b_n * om_{tn}) * c_n
        #
        # Thus
        # J = (om * mask).dot(CCT)
        # h = ((kappa - om * d) * mask).dot(C)
        T, D_latent, D_emission = self.T, self.D_latent, self.D_emission
        data, inputs, mask, omega = self.data, self.inputs, self.mask, self.omega
        # TODO: This is hacky...
        mask = self.mask if self.mask is not None else np.ones_like(self.data)
        emission_distn = self.emission_distn

        C = emission_distn.A[:, :D_latent]
        D = emission_distn.A[:,D_latent:]
        b = emission_distn.b
        CCT = np.array([np.outer(cp, cp) for cp in C]).\
            reshape((D_emission, D_latent ** 2))

        J_node = np.dot(omega * mask, CCT)
        J_node = J_node.reshape((T, D_latent, D_latent))

        kappa = emission_distn.kappa_func(data)
        h_node = ((kappa - omega * b.T - omega * inputs.dot(D.T)) * mask).dot(C)

        # TODO: Implement the log normalizer for the Polya-gamma augmentation
        # after augmentation, the normalizer includes the terms in the PG
        # augmented density that do not contain x. Specifically, we have,
        #
        #     logZ = -b(y) log 2 - log PG(omega | b(y), 0) - log c(y),
        #
        # where b(y) and c(y) come from the count distribution. The hard part is
        # that the PG density is expensive to evaluate. For now, we will just
        # ignore all these terms.
        log_Z_node = np.zeros(self.T)

        return J_node, h_node, log_Z_node

    @property
    def expected_info_emission_params(self):
        if self.has_count_data:
            raise NotImplementedError("Mean field with count observations is not yet supported")

        return super(LDSStatesCountData, self).expected_info_emission_params

    def log_likelihood(self):
        if self.has_count_data:
            ll = self.emission_distn.log_likelihood(
                (np.hstack((self.gaussian_states, self.inputs)), self.data),
                mask=self.mask).sum()
            return ll

        else:
            return super(LDSStatesCountData, self).log_likelihood()

    def resample(self, niter=1):
        self.resample_gaussian_states()

        if self.has_count_data:
            self.resample_auxiliary_variables()

    def resample_auxiliary_variables(self):
        C, D, ed = self.C, self.D, self.emission_distn
        psi = self.gaussian_states.dot(C.T) + self.inputs.dot(D.T) + ed.b.T

        b = ed.b_func(self.data)
        import pypolyagamma as ppg
        ppg.pgdrawvpar(self.ppgs, b.ravel(), psi.ravel(), self.omega.ravel())

    def smooth(self):
        if not self.has_count_data:
            return super(LDSStatesCountData, self).smooth()

        X = np.column_stack((self.gaussian_states, self.inputs))
        mean = self.emission_distn.mean(X)

        return mean


class LDSStatesZeroInflatedCountData(LDSStatesMissingData, _LDSStatesGibbs):
    """
    In many cases, the observation dimension is so large and so sparse
    that a Bernoulli, Poisson, etc. is not a good model. Moreover, it
    is computationally demanding to compute the likelihood for so many
    terms. Zero-inflated models address both concerns. Let,

        z_{t,n} ~ Bern(rho)
        y_{t,n} ~ p(y_{t,n} | c_n.dot(x_t) + d_n))  if z_{t,n} = 1
                = 0                                 o.w.

    If z_{t,n} = 1, we say that datapoint was "exposed." That is, the
    observation y_{t,n} reflects the underlying latent state. The
    observation may be zero, but that is still informative. However,
    if the datapoint was not exposed (which can only happen if y_{t,n}=0),
    then this term does not reflect the underlying state.

    Thus, Z is effectively a mask on the data, and the likelihood only
    depends on places where z_{t,n} = 1. Moreover, we only have to
    introduce auxiliary variables for the entries that are unmasked.
    """
    def __init__(self,model, data=None, **kwargs):

        # The data must be provided in sparse row format
        # This makes it easy to iterate over rows. Basically,
        # for each row, t, it is easy to get the output dimensions, n,
        # such that y_{t,n} > 0.
        super(LDSStatesZeroInflatedCountData, self).\
            __init__(model, data=data, **kwargs)

        # Initialize the Polya-gamma samplers
        num_threads = ppg.get_omp_num_threads()
        seeds = np.random.randint(2 ** 16, size=num_threads)
        self.ppgs = [ppg.PyPolyaGamma(seed) for seed in seeds]

        # Initialize the masked data
        if data is not None:
            assert isinstance(data, csr_matrix), "Data must be a sparse row matrix for zero-inflated models"

            # Initialize a sparse matrix of masked data. The mask
            # specifies which observations were "exposed" and which
            # were determinisitcally zero. In other words, the mask
            # gives the data values at the places where z_{t,n} = 1.
            T, N, C, D, b = self.T, self.D_emission, self.C, self.D, self.emission_distn.b
            indptr = [0]
            indices = []
            vals = []
            offset = 0
            for t in range(T):
                # Get the nonzero entries in the t-th row
                ns_t = data.indices[data.indptr[t]:data.indptr[t + 1]]
                y_t = np.zeros(N)
                y_t[ns_t] = data.data[data.indptr[t]:data.indptr[t + 1]]

                # Sample zero inflation mask
                z_t = np.random.rand(N) < self.rho
                z_t[ns_t] = True

                # Construct the sparse matrix
                t_inds = np.where(z_t)[0]
                indices.append(t_inds)
                vals.append(y_t[t_inds])
                offset += t_inds.size
                indptr.append(offset)

            # Construct a sparse matrix
            vals = np.concatenate(vals)
            indices = np.concatenate(indices)
            indptr = np.array(indptr)
            self.masked_data = csr_matrix((vals, indices, indptr), shape=(T, N))

            # DEBUG: Start with all the data
            # dense_data = data.toarray()
            # values = dense_data.ravel()
            # indices = np.tile(np.arange(self.D_emission), (self.T,))
            # indptrs = np.arange(self.T+1) * self.D_emission
            # self.masked_data = csr_matrix((values, indices, indptrs), (self.T, self.D_emission))
            # assert np.allclose(self.masked_data.toarray(), dense_data)

            self.resample_auxiliary_variables()
        else:
            self.masked_data = None
            self.omega = None

    @property
    def rho(self):
        return self.model.rho

    @property
    def sigma_obs(self):
        raise Exception("Count data does not have sigma_obs")

    def generate_obs(self):
        # Go through each time bin, get the discrete latent state,
        # use that to index into the emission_distns to get samples
        T, p = self.T, self.D_emission
        ed = self.emission_distn
        gss = self.gaussian_states
        data = np.empty((T,p),dtype='double')

        # TODO: Do this sparsely
        for t in range(self.T):
            data[t] = \
                ed.rvs(x=np.hstack((gss[t][None, :], self.inputs[t][None,:])),
                       return_xy=False)

            # Zero out data
            zeros = np.random.rand(p) > self.rho
            data[t][zeros] = 0

        data = csr_matrix(data)
        return data

    @property
    def info_emission_params(self):
        T, D_latent, D_emission = self.T, self.D_latent, self.D_emission

        masked_data, inputs, omega = self.masked_data, self.inputs, self.omega
        emission_distn = self.emission_distn

        C = emission_distn.A[:, :D_latent]
        CCT = np.array([np.outer(c, c) for c in C]).reshape((D_emission, D_latent**2))
        D = emission_distn.A[:,D_latent:]
        b = emission_distn.b

        J_node = omega.dot(CCT).reshape((T, D_latent, D_latent))

        kappa = emission_distn.kappa_func(masked_data.data)
        kappa = csr_matrix((kappa, masked_data.indices, masked_data.indptr), shape=masked_data.shape)
        h_node = kappa.dot(C)

        # Unfortunately, the following operations would require dense arrays of size (TxD_emisison)
        # h_node += -(omega * b.T).dot(C)
        # h_node += -(omega * inputs.dot(D.T)).dot(C)
        # This might not be much faster, but it should avoid making
        # dense arrays
        for t in range(T):
            ns_t = masked_data.indices[masked_data.indptr[t]:masked_data.indptr[t+1]]
            om_t = omega.data[omega.indptr[t]:omega.indptr[t+1]]
            h_node[t] -= (om_t * b[ns_t][:,0]).dot(C[ns_t])
            h_node[t] -= (om_t * inputs[t].dot(D[ns_t].T)).dot(C[ns_t])

        # TODO: See comment in _LDSStatesCountData for info on the log normalizers
        # The same applies to this zero-inflated data
        log_Z_node = np.zeros(self.T)

        return J_node, h_node, log_Z_node

    @property
    def expected_info_emission_params(self):
        raise NotImplementedError("Mean field with count observations is not yet supported")

    @property
    def expected_extra_info_params(self):
        raise NotImplementedError("Mean field with count observations is not yet supported")

    @property
    def psi(self):
        T, C, D, ed = self.T, self.C, self.D, self.emission_distn
        data, size, indices, indptr \
            = self.masked_data, self.masked_data.size, \
              self.masked_data.indices, self.masked_data.indptr

        psi = np.zeros(size)
        offset = 0
        for t in range(T):
            for n in indices[indptr[t]:indptr[t + 1]]:
                psi[offset] = self.gaussian_states[t].dot(C[n])
                psi[offset] += self.inputs[t].dot(D[n])
                psi[offset] += ed.b[n]
                offset += 1
        return csr_matrix((psi, indices, indptr), shape=data.shape)

    def resample(self, niter=1):
        self.resample_zeroinflation_variables()
        self.resample_auxiliary_variables()
        self.resample_gaussian_states()

    def resample_zeroinflation_variables(self):
        """
        There's no way around the fact that we have to look at every
        data point, even the zeros here.
        """
        # TODO: move this to cython?
        T, N, C, D, b = self.T, self.D_emission, self.C, self.D, self.emission_distn.b
        indptr = [0]
        indices = []
        vals = []
        offset = 0
        X = np.hstack((self.gaussian_states, self.inputs))
        for t in range(T):
            # Evaluate probability of data
            y_t = np.zeros(N)
            ns_t = self.data.indices[self.data.indptr[t]:self.data.indptr[t+1]]
            y_t[ns_t] = self.data.data[self.data.indptr[t]:self.data.indptr[t+1]]
            ll = self.emission_distn._elementwise_log_likelihood((X[t], y_t))
            ll = ll.ravel()

            # Evaluate the probability that each emission was "exposed",
            # i.e. p(z_tn = 1 | y_tn, x_tn)
            log_p_exposed = np.log(self.rho) + ll
            log_p_exposed -= np.log(np.exp(log_p_exposed) + (1-self.rho) * (y_t == 0))

            # Sample zero inflation mask
            z_t = np.random.rand(N) < np.exp(log_p_exposed)

            # Construct the sparse matrix
            t_inds = np.where(z_t)[0]
            indices.append(t_inds)
            vals.append(y_t[t_inds])
            offset += t_inds.size
            indptr.append(offset)

        # Construct a sparse matrix
        vals = np.concatenate(vals)
        indices = np.concatenate(indices)
        indptr = np.array(indptr)
        self.masked_data = csr_matrix((vals, indices, indptr), shape=(T, N))

    def resample_auxiliary_variables(self):
        # TODO: move this to cython
        T, C, D, ed = self.T, self.C, self.D, self.emission_distn
        data, size, indices, indptr \
            = self.masked_data, self.masked_data.size, \
              self.masked_data.indices, self.masked_data.indptr
        psi = np.zeros(size)
        offset = 0
        for t in range(T):
            for n in indices[indptr[t]:indptr[t+1]]:
                psi[offset] = self.gaussian_states[t].dot(C[n])
                psi[offset] += self.inputs[t].dot(D[n])
                psi[offset] += ed.b[n]
                offset += 1
        psi = csr_matrix((psi, indices, indptr), shape=data.shape)
        b = ed.b_func(data)

        # Allocate vector for omega
        self.omega = np.zeros(size)
        ppg.pgdrawvpar(self.ppgs, b.data, psi.data, self.omega)
        self.omega = csr_matrix((self.omega, indices, indptr), shape=data.shape)

    def smooth(self):
        # TODO: By assumption, the data is too large to construct
        # TODO: a dense smoothing matrix. Let's support a column-wise
        # TODO: smoothing operation instead.
        warn("Zero inflated smoothing is instantiating a dense matrix!")
        X = np.column_stack((self.gaussian_states, self.inputs))
        mean = self.rho * self.emission_distn.mean(X)
        return mean

