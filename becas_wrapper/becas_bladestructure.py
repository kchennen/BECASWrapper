
import numpy as np
import os

from openmdao.api import Component, Group, ParallelGroup, ExecComp

from cs2dtobecas import CS2DtoBECAS
from becas_wrapper import BECASWrapper

from fusedwind.lib.geom_tools import calculate_length

class BECASCSStructure(Component):
    """
    Component for computing beam structural properties
    using the cross-sectional structure code BECAS.

    The code firstly calls CS2DtoBECAS which is a wrapper around
    shellexpander that comes with BECAS, and second
    calls BECAS using a file interface.

    parameters
    ----------
    config: dict
        dictionary of model specific inputs
    coords: array
        cross-sectional shape. size ((ni_chord, 3))
    matprops: array
        material stiffness properties. Size ((10, nmat)).
    failmat: array
        material strength properties. Size ((18, nmat)).
    DPs: array
        vector of DPs. Size: (nDP)
    coords: array
        blade section coordinates. Size: ((ni_chord, 3))
    r<xx><lname>T: float
        layer thicknesses, e.g. r01triaxT.
    r<xx><lname>A: float
        layer angles, e.g. r01triaxA.
    w<xx><lname>T: float
        web thicknesses, e.g. r01triaxT.
    w<xx><lname>A: float
        web angles, e.g. r01triaxA.

    returns
    -------
    cs_props: array
        vector of cross section properties. Size (19) or (30)
        for standard HAWC2 output or the fully populated stiffness
        matrix, respectively.

    csprops_ref: array
        vector of cross section properties as of BECAS output w.r.t BECAS reference
        coordinate system. Size (18).
    """

    def __init__(self, name, becas_hash, config, st3d, s, ni_chord, cs_size, cs_size_ref):
        """
        parameters
        ----------
        config: dict
            dictionary with inputs to CS2DtoBECAS and BECASWrapper
        st3d: dict
            dictionary with blade structural definition
        s: array
            spanwise location of the cross-section
        ni_chord: int
            number of points definiting the cross-section shape
        cs_size: int
            size of blade_beam_structure array (19 or 30)
        cs_size_ref: int
            size of blade_beam_csprops_ref array (18)
        """
        super(BECASCSStructure, self).__init__()

        self.basedir = os.getcwd()
        self.becas_hash = becas_hash
        self.nr = len(st3d['regions'])
        self.ni_chord = ni_chord

        # fix mesh distribution function after first run
        # defaults to True
        try:
            self.fix_mesh_distribution = config['fix_mesh_distribution']
        except:
            self.fix_mesh_distribution = True

        # add materials properties array ((10, nmat))
        self.add_param('matprops', st3d['matprops'])

        # add materials strength properties array ((18, nmat))
        self.add_param('failmat', st3d['failmat'])

        # add DPs array
        self.add_param('%s:DPs' % name, np.zeros(self.nr + 1))

        # add coords coords
        self._varnames = []
        self.add_param('%s:coords' % name, np.zeros((ni_chord, 3)))

        self.cs2di = {}
        self.cs2di['materials'] = st3d['materials']
        self.cs2di['matprops'] = st3d['matprops']
        self.cs2di['failcrit'] = st3d['failcrit']
        self.cs2di['failmat'] = st3d['failmat']
        self.cs2di['web_def'] = st3d['web_def']
        self.cs2di['s'] = s
        self.cs2di['DPs'] = np.zeros(self.nr + 1)
        self.cs2di['regions'] = []
        self.cs2di['webs'] = []
        for ireg, reg in enumerate(st3d['regions']):
            r = {}
            r['layers'] = reg['layers']
            nl = len(reg['layers'])
            r['thicknesses'] = np.zeros(nl)
            r['angles'] = np.zeros(nl)
            self.cs2di['regions'].append(r)
            for i, lname in enumerate(reg['layers']):
                varname = '%s:r%02d%s' % (name, ireg, lname)
                self._varnames.append(varname)
        for ireg, reg in enumerate(st3d['webs']):
            r = {}
            r['layers'] = reg['layers']
            nl = len(reg['layers'])
            r['thicknesses'] = np.zeros(nl)
            r['angles'] = np.zeros(nl)
            self.cs2di['webs'].append(r)
            for i, lname in enumerate(reg['layers']):
                varname = '%s:w%02d%s' % (name, ireg, lname)
                self._varnames.append(varname)
        self.add_param(name + ':tvec', np.zeros(len(self._varnames)*2))

        # add outputs
        self.add_output('%s:cs_props' % name, np.zeros(cs_size))
        self.add_output('%s:csprops_ref' % name, np.zeros(cs_size_ref))
        self.add_output('%s:k_matrix' % name, shape=(6,6))
        self.add_output('%s:m_matrix' % name, shape=(6,6))
        self.cs_props_m1 = np.zeros(cs_size)
        self.csprops_ref_m1 = np.zeros(cs_size_ref)
        self.k_matrix_m1 = np.zeros((6,6))
        self.m_matrix_m1 = np.zeros((6,6))

        self.add_output('%s:DPcoords' % name, np.zeros((self.nr + 1, 3)))

        self.workdir = 'becas_%s_%i' % (name, self.becas_hash)
        # not so nice hack to ensure unique directory names when
        # running parallel FD
        # the hash is passed to downstream BECASStressRecovery class
        self.add_output(name + ':hash', float(self.becas_hash))

        self.mesher = CS2DtoBECAS(self.cs2di, **config['CS2DtoBECAS'])
        self.becas = BECASWrapper(self.cs2di['s'], **config['BECASWrapper'])
        self.redistribute_flag = True

    def _params2dict(self, params):
        """
        convert the OpenMDAO params dictionary into
        the dictionary format used in CS2DtoBECAS.
        """
        tvec = params[self.name+':tvec']

        self.cs2d = {}
        # constants
        self.cs2d['s'] = self.cs2di['s']
        self.cs2d['web_def'] = self.cs2di['web_def']
        self.cs2d['failcrit'] = self.cs2di['failcrit']
        self.cs2d['materials'] = self.cs2di['materials']

        # params
        self.cs2d['coords'] = params['%s:coords' % self.name][:, :2]
        self.cs2d['matprops'] = params['matprops']
        self.cs2d['failmat'] = params['failmat']
        self.cs2d['DPs'] = params['%s:DPs' % self.name]
        self.cs2d['regions'] = []
        self.cs2d['webs'] = []
        counter = 0
        nvar = len(self._varnames)
        for ireg, reg in enumerate(self.cs2di['regions']):
            self.cs2d['regions'].append({})
            Ts = []
            As = []
            layers = []
            for i, lname in enumerate(reg['layers']):
                if tvec[counter] > 0.:
                    Ts.append(tvec[counter])
                    As.append(tvec[nvar+counter])
                    layers.append(lname)
                counter += 1
            self.cs2d['regions'][ireg]['thicknesses'] = np.asarray(Ts)
            self.cs2d['regions'][ireg]['angles'] = np.asarray(As)
            self.cs2d['regions'][ireg]['layers'] = layers
        for ireg, reg in enumerate(self.cs2di['webs']):
            self.cs2d['webs'].append({})
            Ts = []
            As = []
            layers = []
            for i, lname in enumerate(reg['layers']):
                if tvec[counter] > 0.:
                    Ts.append(tvec[counter])
                    As.append(tvec[nvar+counter])
                    layers.append(lname)
                counter += 1
            self.cs2d['webs'][ireg]['thicknesses'] = np.asarray(Ts)
            self.cs2d['webs'][ireg]['angles'] = np.asarray(As)
            self.cs2d['webs'][ireg]['layers'] = layers

    def solve_nonlinear(self, params, unknowns, resids):
        """
        calls CS2DtoBECAS/shellexpander to generate mesh
        and BECAS to compute the cs_props and csprops
        """

        try:
            os.mkdir(self.workdir)
        except:
            pass
        os.chdir(self.workdir)

        self._params2dict(params)

        self.mesher.cs2d = self.cs2d

        try:
            self.mesher.compute(self.redistribute_flag)
            self.becas.compute()
            if self.becas.success:
                self.unknowns['%s:DPcoords' % self.name][:,0:2] = np.array(self.mesher.DPcoords)
                self.unknowns['%s:cs_props' % self.name] = self.becas.cs_props
                self.unknowns['%s:csprops_ref' % self.name] = self.becas.csprops
                self.unknowns['%s:k_matrix' % self.name] = self.becas.k_matrix
                self.unknowns['%s:m_matrix' % self.name] = self.becas.m_matrix
                self.cs_props_m1 = self.becas.cs_props.copy()
                self.csprops_ref_m1 = self.becas.csprops.copy()
                self.k_matrix_m1 = self.becas.k_matrix.copy()
                self.m_matrix_m1 = self.becas.m_matrix.copy()
            else:
                self.unknowns['%s:cs_props' % self.name] = self.cs_props_m1
                self.unknowns['%s:csprops_ref' % self.name] = self.csprops_ref_m1
                self.unknowns['%s:k_matrix' % self.name] = self.k_matrix_m1
                self.unknowns['%s:m_matrix' % self.name] = self.m_matrix_m1
                print('BECAS crashed for section %f' % self.cs2d['s'])
        except:
            self.unknowns['%s:cs_props' % self.name] = self.cs_props_m1
            self.unknowns['%s:csprops_ref' % self.name] = self.csprops_ref_m1
            self.unknowns['%s:k_matrix' % self.name] = self.k_matrix_m1
            self.unknowns['%s:m_matrix' % self.name] = self.m_matrix_m1
            print('BECAS crashed for section %f' % self.cs2d['s'])

        os.chdir(self.basedir)
        if self.fix_mesh_distribution:
            self.redistribute_flag = False


class Slice(Component):
    """
    simple component for slicing arrays into vectors
    for passing to sub-comps computing the csprops

    parameters
    ----------
    DP<xx>: array
        arrays of DPs along span. Size: (nsec)
    blade_surface_norm_st: array
        blade surface with structural discretization, no twist and prebend.
        Size: ((ni_chord, nsec, 3))

    returns
    -------
    sec<xxx>DPs: array
        Vector of DPs along chord for each section. Size (nDP)
    sec<xxx>coords: array
        Array of cross section coords shapes. Size ((ni_chord, 3))
    """

    def __init__(self, st3d, sdim):
        """
        parameters
        ----------
        DPs: array
            DPs array, size: ((nsec, nDP))
        sdim: array
            blade surface. Size: ((ni_chord, nsec, 3))
        """
        super(Slice, self).__init__()

        self.nsec = sdim[1]
        DPs = st3d['DPs']
        self.nDP = DPs.shape[1]

        for i in range(self.nDP):
            self.add_param('DP%02d' % i, DPs[:, i])

        self.add_param('blade_surface_norm_st', np.zeros(sdim))
        self.add_param('blade_length', 0.)

        vsize = 0
        self._varnames = []
        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                self.add_param(varname + 'T', np.zeros(self.nsec))
                self.add_param(varname + 'A', np.zeros(self.nsec))
                self._varnames.append(varname)
        for ireg, reg in enumerate(st3d['webs']):
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                self.add_param(varname + 'T', np.zeros(self.nsec))
                self.add_param(varname + 'A', np.zeros(self.nsec))
                self._varnames.append(varname)

        for i in range(self.nsec):
            self.add_output('sec%03d:DPs' % i, DPs[i, :])
            self.add_output('sec%03d:coords' % i, np.zeros((sdim[0], sdim[2])))
            self.add_output('sec%03d:tvec' % i, np.zeros(len(self._varnames)*2))

    def solve_nonlinear(self, params, unknowns, resids):

        nvar = len(self._varnames)
        for i in range(self.nsec):
            DPs = np.zeros(self.nDP)
            for j in range(self.nDP):
                DPs[j] = params['DP%02d' % j][i]
            unknowns['sec%03d:DPs' % i] = DPs
            unknowns['sec%03d:coords' % i] = params['blade_surface_norm_st'][:, i, :] * \
                                             params['blade_length']
            for ii, name in enumerate(self._varnames):
                unknowns['sec%03d:tvec' % i][ii] = params[name + 'T'][i]
                unknowns['sec%03d:tvec' % i][nvar+ii] = params[name + 'A'][i]


class PostprocessCS(Component):
    """
    component for gathering cross section props
    into array as function of span

    parameters
    ----------
    cs_props<xxx>: array
        array of cross section props in HAWC2 format. Size (19).
    csprops_ref<xxx>: array
        array of cross section props wrt ref axis. Size (18).
    blade_x: array
        dimensionalised x-coordinate of blade axis
    blade_y: array
        dimensionalised y-coordinate of blade axis
    blade_z: array
        dimensionalised z-coordinate of blade axis
    hub_radius: float
        dimensionalised hub length

    returns
    -------
    blade_beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    blade_beam_csprops_ref: array
        array of beam cs properties. Size ((nsec, 18)).
    blade_mass: float
        blade mass integrated from dm in beam properties
    blade_mass_moment: float
        blade mass moment integrated from dm in beam properties
    """

    def __init__(self, nsec, cs_size, cs_size_ref):
        """
        parameters
        ----------
        nsec: int
            number of blade sections.
        cs_size: int
            size of blade_beam_structure array (19 or 30).
        cs_size_ref: int
            size of blade_beam_csprops_ref array (18).
        """
        super(PostprocessCS, self).__init__()

        self.nsec = nsec

        for i in range(nsec):
            self.add_param('cs_props%03d' % i, np.zeros(cs_size),
                desc='cross-sectional props for sec%03d' % i)
            self.add_param('csprops_ref%03d' % i, np.zeros(cs_size_ref),
                desc='cross-sectional props for sec%03d' % i)
            self.add_param('k_matrix%03d' % i, shape=(6,6),
                desc='stiffness matrix for sec%03d' % i)
            self.add_param('m_matrix%03d' % i, shape=(6,6),
                desc='mass matrix for sec%03d' % i)
        self.add_param('hub_radius', 0., units='m', desc='Hub length')
        self.add_param('blade_length', 0., units='m', desc='Blade length')

        self.add_param('x_st', np.zeros(nsec), units='m',
            desc='non-dimensionalised x-coordinate of blade axis in structural grid')
        self.add_param('y_st', np.zeros(nsec), units='m',
            desc='non-dimensionalised y-coordinate of blade axis in structural grid')
        self.add_param('z_st', np.zeros(nsec), units='m',
            desc='non-dimensionalised y-coordinate of blade axis in structural grid')
        self.add_param('chord_st', np.zeros(nsec), units='m',
            desc='blade chord distribution in structural grid')
        self.add_param('p_le_st', np.zeros(nsec), units='m',
            desc='blade pitch axis aft leading edge in structural grid')


        self.add_output('blade_beam_structure', np.zeros((nsec, cs_size)),
            desc='Beam properties of the blade')
        self.add_output('blade_beam_csprops_ref', np.zeros((nsec, cs_size_ref)),
            desc='Cross section properties of the blade')
        self.add_output('blade_mass', 0., units='kg', desc='Blade mass')
        self.add_output('blade_mass_moment', 0., units='N*m',
            desc='Blade mass moment')
        self.add_output('MStruct', shape=(6,6,nsec))
        self.add_output('KStruct', shape=(6,6,nsec))

    def solve_nonlinear(self, params, unknowns, resids):
        """
        aggregate results and integrate mass and mass moment using np.trapz.
        """
        for i in range(self.nsec):
            cname = 'cs_props%03d' % i
            cs = params[cname]
            unknowns['blade_beam_structure'][i, :] = cs

        # offset chordwise position of x_cg, x_sh, and to half chord
        unknowns['blade_beam_structure'][:, 2]  += (0.5 - params['p_le_st']) * params['chord_st'] * params['blade_length']
        unknowns['blade_beam_structure'][:, 6]  += (0.5 - params['p_le_st']) * params['chord_st'] * params['blade_length']
        unknowns['blade_beam_structure'][:, 17] += (0.5 - params['p_le_st']) * params['chord_st'] * params['blade_length']

        # compute mass and mass moment
        x = params['x_st'] * params['blade_length']
        y = params['y_st'] * params['blade_length']
        z = params['z_st'] * params['blade_length']
        hub_radius = params['hub_radius']
        s = calculate_length(np.array([x, y, z]).T)

        unknowns['blade_beam_structure'][:, 0] = s

        dm = unknowns['blade_beam_structure'][:, 1]
        g = 9.81

        # mass
        m = np.trapz(dm, s)
        unknowns['blade_mass'] = m

        # mass moment
        mm = np.trapz(g * dm * (z + hub_radius), s)
        unknowns['blade_mass_moment'] = mm

        print('blade mass %10.3f' % m)

        for i in range(self.nsec):
            cname = 'csprops_ref%03d' % i
            cs = params[cname]
            unknowns['blade_beam_csprops_ref'][i, :] = cs

        for i in range(self.nsec):
            unknowns['KStruct'][:,:,i] = params['k_matrix%03d' % i]
            unknowns['MStruct'][:,:,i] = params['m_matrix%03d' % i]


class BECASBeamStructure(Group):
    """
    Group for computing beam structure properties
    using the cross-sectional structure code BECAS.

    The geometric and structural inputs used are defined
    in detail in FUSED-Wind.

    parameters
    ----------
    blade_x: array
        dimensionalised x-coordinates of blade axis with structural discretization.
    blade_y: array
        dimensionalised y-coordinates of blade axis with structural discretization.
    blade_z: array
        dimensionalised z-coordinates of blade axis with structural discretization.
    blade_surface_norm_st: array
        blade surface with structural discretization, no twist and prebend.
        Size: ((ni_chord, nsec, 3))
    matprops: array
        material stiffness properties. Size (10, nmat).
    failmat: array
        material strength properties. Size (18, nmat).
    sec<xx>DPs: array
        2D array of DPs. Size: ((nsec, nDP))
    sec<xx>coords: array
        blade surface. Size: ((ni_chord, nsec, 3))
    sec<xx>r<yy><lname>T: array
        region layer thicknesses, e.g. r01triaxT. Size (nsec)
    sec<xx>r<yy><lname>A: array
        region layer angles, e.g. r01triaxA. Size (nsec)
    sec<xx>w<yy><lname>T: array
        web layer thicknesses, e.g. r01triaxT. Size (nsec)
    sec<xx>w<yy><lname>A: array
        web layer angles, e.g. r01triaxA. Size (nsec)

    returns
    -------
    blade_beam_structure: array
        array of beam structure properties. Size ((nsec, 19)).
    blade_mass: float
        blade mass integrated from blade_beam_structure dm
    blade_mass_moment: float
        blade mass moment integrated from blade_beam_structure dm
    blade_beam_csprops_ref: array
        array of beam cs properties. Size ((nsec, 18)).
    KStruct: array size (6,6,nsec)
        array of stiffness matrix
        variables: K_11 K_12 K_13 K_14 K_15 K_16 K_22 K_23 K_24 K_25 K_26
                   K_33 K_34 K_35 K_36 K_44 K_45 K_46 K_55 K_56 K_66
    MStruct: array size (6,6,nsec)
        array of mass matrix
        variables: M_11 M_12 M_13 M_14 M_15 M_16 M_22 M_23 M_24 M_25 M_26
                   M_33 M_34 M_35 M_36 M_44 M_45 M_46 M_55 M_56 M_66
    """

    def __init__(self, group, config, st3d, sdim):
        """
        initializes parameters and adds a csprops component
        for each section

        parameters
        ----------
        config: dict
            dictionary of inputs for the cs_code class
        st3d: dict
            dictionary of blade structure properties
        surface: array
            blade surface with structural discretization.
            Size: ((ni_chord, nsec, 3))
        """
        super(BECASBeamStructure, self).__init__()

        # check that the config is ok
        if not 'CS2DtoBECAS' in config.keys():
            raise RuntimeError('You need to supply a config dict',
                               'for CS2DtoBECAS')
        if not 'BECASWrapper' in config.keys():
            raise RuntimeError('You need to supply a config dict',
                               'for BECASWrapper')
        try:
            analysis_mode = config['BECASWrapper']['analysis_mode']
            if not analysis_mode == 'stiffness':
                config['BECASWrapper']['analysis_mode'] = 'stiffness'
                print 'BECAS analysis mode wasnt set to `stiffness`,',\
                      'trying to set it for you'
        except:
            print 'BECAS analysis mode wasnt set to `stiffness`,',\
                  'trying to set it for you'
            config['BECASWrapper']['analysis_mode'] = 'stiffness'

        try:
            if config['BECASWrapper']['hawc2_FPM']:
                cs_size = 30
            else:
                cs_size = 19
        except:
            cs_size = 19

        cs_size_ref = 18

        self.st3d = st3d
        nr = len(st3d['regions'])
        nsec = st3d['s'].shape[0]

        # create a unique ID for this group so that FD's are not overwritten
        self.add('hash_c', ExecComp('becas_hash=%f' % float(self.__hash__())), promotes=['*'])

        # add comp to slice the 2D arrays DPs and surface
        self.add('slice', Slice(st3d, sdim), promotes=['*'])

        self._varnames = []
        for ireg, reg in enumerate(st3d['regions']):
            for i, lname in enumerate(reg['layers']):
                varname = 'r%02d%s' % (ireg, lname)
                self._varnames.append(varname)
        for ireg, reg in enumerate(st3d['webs']):
            for i, lname in enumerate(reg['layers']):
                varname = 'w%02d%s' % (ireg, lname)
                self._varnames.append(varname)

        # # now add a component for each section
        par = self.add('par', ParallelGroup(), promotes=['*'])

        for i in range(nsec):
            secname = 'sec%03d' % i
            par.add(secname, BECASCSStructure(secname, self.__hash__(), config, st3d,
                                              st3d['s'][i], sdim[0], cs_size, cs_size_ref), promotes=['*'])

        promotions = ['hub_radius',
                      'blade_length',
                      'x_st',
                      'y_st',
                      'z_st',
                      'chord_st',
                      'p_le_st',
                      'blade_beam_structure',
                      'blade_beam_csprops_ref',
                      'blade_mass',
                      'blade_mass_moment',
                      'KStruct',
                      'MStruct']
        self.add('postpro', PostprocessCS(nsec, cs_size, cs_size_ref), promotes=promotions)
        for i in range(nsec):
            secname = 'sec%03d' % i
            self.connect('%s:cs_props' % secname, 'postpro.cs_props%03d' % i)
            self.connect('%s:csprops_ref' % secname, 'postpro.csprops_ref%03d' % i)
            self.connect('%s:k_matrix' % secname, 'postpro.k_matrix%03d' % i)
            self.connect('%s:m_matrix' % secname, 'postpro.m_matrix%03d' % i)


class BECASCSStructureKM(Component):

    def __init__(self, name, becas_hash, config, input_folder, s):
        """
        parameters
        ----------
        config: dict
            dictionary with inputs to BECASWrapper
        input_folder: list
            list with becas input folders
        s: array
            spanwise location of the cross-section
        """

        super(BECASCSStructureKM, self).__init__()

        self.basedir = os.getcwd()
        self.becas_hash = becas_hash

        # add outputs
        self.add_output('%s:k_matrix' % name, shape=(6,6))
        self.add_output('%s:m_matrix' % name, shape=(6,6))

        self.workdir = 'becas_%s_%i' % (name, self.becas_hash)
        # not so nice hack to ensure unique directory names when
        # running parallel FD
        # the hash is passed to downstream BECASStressRecovery class
        self.add_output(name + ':hash', float(self.becas_hash))

        config['BECASWrapper']['path_input'] = os.path.join(self.basedir, input_folder)

        self.becas = BECASWrapper(s, **config['BECASWrapper'])
        self.s = s

    def solve_nonlinear(self, params, unknowns, resids):
        """
        calls BECAS to compute the stiffness and mass terms
        """

        try:
            os.mkdir(self.workdir)
        except:
            pass
        os.chdir(self.workdir)

        self.becas.compute()
        self.unknowns['%s:k_matrix' % self.name] = self.becas.k_matrix
        self.unknowns['%s:m_matrix' % self.name] = self.becas.m_matrix

        #remove becas output files and folders related to hash
#         os.remove('becas_section.m')
#         os.remove('BECAS_SetupPath.m')
#         os.remove('becas_utils%.3f.mat' % self.s)
#         os.rmdir(os.getcwd())

        os.chdir(self.basedir)

class PostprocessCSKM(Component):

    def __init__(self, nsec, becas_span):
        super(PostprocessCSKM, self).__init__()
        """
        parameters
        ----------
        nsec: float
            number of sections to read in
        becas_span: array
            spanwise location vector of the cross-section
        """

        for i in range(nsec):
            self.add_param('k_matrix%03d' % i, shape=(6,6), desc='stiffness matrix for sec%03d' % i)
            self.add_param('m_matrix%03d' % i, shape=(6,6), desc='mass matrix for sec%03d' % i)

        self.add_output('MStruct', shape=(6,6,nsec))
        self.add_output('KStruct', shape=(6,6,nsec))
        self.add_output('sStruct', shape=(nsec))

        self.nsec = nsec
        self.becas_span = becas_span

    def solve_nonlinear(self, params, unknowns, resids):

        unknowns['sStruct'] = self.becas_span
        for i in range(self.nsec):
            unknowns['KStruct'][:,:,i] = params['k_matrix%03d' % i]
            unknowns['MStruct'][:,:,i] = params['m_matrix%03d' % i]


class BECASBeamStructureKM(Group):
    """
    Group for computing mass and stiffness matrix
    using the cross-sectional structure code BECAS.

    returns
    -------
    KStruct: array size (6,6,nsec)
        array of stiffness matrix
        variables: K_11 K_12 K_13 K_14 K_15 K_16 K_22 K_23 K_24 K_25 K_26
                   K_33 K_34 K_35 K_36 K_44 K_45 K_46 K_55 K_56 K_66
    MStruct: array size (6,6,nsec)
        array of mass matrix
        variables: M_11 M_12 M_13 M_14 M_15 M_16 M_22 M_23 M_24 M_25 M_26
                   M_33 M_34 M_35 M_36 M_44 M_45 M_46 M_55 M_56 M_66
    sStruct: array size nsec
        array of spanwise positions
    """

    def __init__(self, group, config, becasInp):
        """
        initializes parameters and adds a BECASStructure component
        for each section

        parameters
        ----------
        config: dict
            dictionary of configurations for BECASWrapper
        becasInp: dict
            dictionary of blade structure properties
            variables: s, nsec, path_input_folders
        """
        super(BECASBeamStructureKM, self).__init__()

        # check that the config is ok
        if not 'BECASWrapper' in config.keys():
            raise RuntimeError('You need to supply a config dict',
                               'for BECASWrapper')
        try:
            analysis_mode = config['BECASWrapper']['analysis_mode']
            if not analysis_mode == 'stiffness':
                config['BECASWrapper']['analysis_mode'] = 'stiffness'
                print 'BECAS analysis mode wasnt set to `stiffness`,',\
                      'trying to set it for you'
        except:
            print 'BECAS analysis mode wasnt set to `stiffness`,',\
                  'trying to set it for you'
            config['BECASWrapper']['analysis_mode'] = 'stiffness'


        # create a unique ID for this group so that FD's are not overwritten
        self.add('hash_c', ExecComp('becas_hash=%f' % float(self.__hash__())), promotes=['*'])

        # # now add a component for each section
        par = self.add('par', ParallelGroup(), promotes=['*'])

        nsec = becasInp['nsec']
        for i in range(nsec):
            secname = 'sec%03d' % i
            par.add(secname, BECASCSStructureKM(secname, self.__hash__(), config, becasInp['path_input_folders'][i], becasInp['s'][i]), promotes=['*'])

        self.add('postpro', PostprocessCSKM(nsec, becasInp['s']), promotes=['KStruct', 'MStruct', 'sStruct'])
        for i in range(nsec):
            secname = 'sec%03d' % i
            self.connect('%s:k_matrix' % secname, 'postpro.k_matrix%03d' % i)
            self.connect('%s:m_matrix' % secname, 'postpro.m_matrix%03d' % i)
