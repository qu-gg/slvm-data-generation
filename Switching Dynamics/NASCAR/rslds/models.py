import numpy as np

from NASCAR.pyhsmm.internals.initial_state import UniformInitialState
from NASCAR.pyhsmm.models import _HMMGibbsSampling
from NASCAR.pyslds.models import _SLDSGibbsMixin
from NASCAR.rslds.states import InputHMMStates, PGRecurrentSLDSStates, SoftmaxRecurrentSLDSStates
import NASCAR.rslds.transitions as transitions


### Input-driven HMMs
class _InputHMMMixin(object):
    # Subclasses must specify the type of transition model
    _trans_class = None

    # custom init method, just so we call custom input trans class stuff
    def __init__(self,
                 obs_distns,
                 D_in=0,
                 trans_distn=None, trans_params={},
                 init_state_distn=None, init_state_concentration=None, pi_0=None,
                 ):
        self.obs_distns = obs_distns
        self.states_list = []
        self.D_in = D_in

        # our trans class
        if trans_distn is None:
            self.trans_distn = self._trans_class(num_states=len(obs_distns),
                                                 covariate_dim=D_in,
                                                 **trans_params)
        else:
            self.trans_distn = trans_distn

        if init_state_distn is not None:
            if init_state_distn == 'uniform':
                self.init_state_distn = UniformInitialState(model=self)
            else:
                self.init_state_distn = init_state_distn
        else:
            self.init_state_distn = self._init_state_class(
                model=self,
                init_state_concentration=init_state_concentration,
                pi_0=pi_0)

        self._clear_caches()

    # custom add_data - includes a covariates arg
    def add_data(self, data, covariates=None, **kwargs):
        # NOTE! Our convention is that covariates[t] drives the
        # NOTE! transition matrix going into time t. However, for
        # NOTE! implementation purposes, it is easier if these inputs
        # NOTE! are lagged so that covariates[t] drives the input to
        # NOTE! z_{t+1}. Then, we only have T-1 inputs for the T-1
        # NOTE! transition matrices in the heterogeneous model.

        # Offset the covariates by one so that
        # the inputs at time {t-1} determine the transition matrix
        # from z_{t-1} to z_{t}.
        offset_covariates = covariates[1:]
        self.states_list.append(
            self._states_class(
                model=self, data=data,
                covariates=offset_covariates, **kwargs))

    def generate(self, T=100, covariates=None, keep=True):
        if covariates is None:
            covariates = np.zeros((T, self.D_in))
        else:
            assert covariates.ndim == 2 and \
                   covariates.shape[0] == T
        s = self._states_class(model=self, covariates=covariates[1:], T=T, initialize_from_prior=True)
        data = self._generate_obs(s)
        if keep:
            self.states_list.append(s)
        return (data, covariates), s.stateseq

    def resample_trans_distn(self):
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
        )
        self._clear_caches()


class PGInputHMM(_InputHMMMixin, _HMMGibbsSampling):
    _trans_class = transitions.InputHMMTransitions
    _states_class = InputHMMStates


### Stick-breaking transition models with Pólya-gamma augmentation
class _RecurrentSLDSBase(object):
    def __init__(self, dynamics_distns, emission_distns, init_dynamics_distns,
                 fixed_emission=False, **kwargs):
        self.fixed_emission = fixed_emission

        # This class must always be used in conjunction with an SLDS class
        super(_RecurrentSLDSBase, self).__init__(
            dynamics_distns, emission_distns, init_dynamics_distns,
            D_in=dynamics_distns[0].D_out, **kwargs)

    def add_data(self, data, **kwargs):
        self.states_list.append(
            self._states_class(model=self, data=data, **kwargs))


class PGRecurrentSLDS(_RecurrentSLDSBase, _SLDSGibbsMixin, PGInputHMM):
    _states_class = PGRecurrentSLDSStates
    _trans_class = transitions.InputHMMTransitions

    def resample_trans_distn(self):
        # Include the auxiliary variables used for state resampling
        self.trans_distn.resample(
            stateseqs=[s.stateseq for s in self.states_list],
            covseqs=[s.covariates for s in self.states_list],
            omegas=[s.trans_omegas for s in self.states_list]
        )
        self._clear_caches()

    def resample_emission_distns(self):
        if self.fixed_emission:
            return
        super(PGRecurrentSLDS, self).resample_emission_distns()
