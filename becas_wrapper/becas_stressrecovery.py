

import numpy as np
import time
import os

from openmdao.api import Component, Group, ParallelGroup

from becas_wrapper import BECASWrapper


class BECASCSStressRecovery(Component):
    """
    component for calling BECAS on individual sections to
    compute stresses and strains.
    """

    def __init__(self, name, config, s, ncases):
        super(BECASCSStressRecovery, self).__init__()

        self.name = name
        self.s = s
        self.basedir = os.getcwd()

        # not so nice hack to ensure unique directory names when
        # running parallel FD
        # the hash is generated in the upstream BECASBeamStructure class
        self.add_param(name + ':hash', 0.)

        self.add_param('load_cases_%s' % name, np.zeros((ncases, 6)))
        self.add_output('blade_failure_index_%s' % name, np.zeros(ncases))

        self.becas = BECASWrapper(s, **config['BECASWrapper'])
        self.becas.analysis_mode = 'stress_recovery'

    def solve_nonlinear(self, params, unknowns, resids):


        becas_hash = params[self.name + ':hash']
        workdir = 'becas_%s_%i' % (self.name, int(becas_hash))
        print 'workdir', workdir
        os.chdir(workdir)
        self.becas.load_cases = params['load_cases_%s' % self.name]
        self.becas.compute()
        unknowns['blade_failure_index_%s' % self.name] = self.becas.max_failure_ks

        os.chdir(self.basedir)


class SRAggregator(Component):


    def __init__(self, config, s, ncases):
        super(SRAggregator, self).__init__()

        self.nsec = s.shape[0]
        for i in range(s.shape[0]):
            name = 'sec%03d' % i
            self.add_param('blade_failure_index_%s' % name, np.zeros(ncases))

        self.add_output('blade_failure_index', np.zeros((ncases, s.shape[0])))

    def solve_nonlinear(self, params, unknowns, resids):

        for i in range(self.nsec):
            unknowns['blade_failure_index'][:, i] = params['blade_failure_index_sec%03d'%i]

class BECASStressRecovery(Group):
    """
    Group for computing stresses and strains using BECAS
    on the cross-sections already created using the
    `BECASBeamStructure` class.

    parameters
    ----------
    load_cases_sec%03d: array
        Load cases containing forces and moments for each cross-section:
        (Fx, Fy, Fz, Mx, My, Mz), size ((nsec, ncases)).

    returns
    -------
    blade_failure_index_%03d: array
        failure index computing using the method specified in blade structural
        definition. size ((nsec, ncase))
    """

    def __init__(self, config, s, ncases):
        super(BECASStressRecovery, self).__init__()
        self.nsec = s.shape[0]
        self.ncases = ncases
        self.s = s

        par = self.add('par', ParallelGroup(), promotes=['*'])

        for i in range(self.nsec):
            b = par.add('sec%03d' % i, BECASCSStressRecovery('sec%03d' % i, config, s[i], ncases), promotes=['*'])

        self.add('agg', SRAggregator(config, s, ncases), promotes=['*'])
