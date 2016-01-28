
import os
import time
import numpy as np
import unittest
import pkg_resources

from openmdao.core.mpi_wrap import MPI
from openmdao.api import Problem, Group, Component, IndepVarComp, ExecComp

from PGL.main.distfunc import distfunc
from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       SplinedBladePlanform, \
                                       PGLRedistributedPlanform
from fusedwind.turbine.structure import read_bladestructure, \
                                        write_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure

from becas_wrapper.becas_bladestructure import BECASBeamStructure

if MPI:
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    from openmdao.core.basic_impl import BasicImpl as impl

# the data for this example is located in the becas_wrapper/test/data
# directory, so we need to load that path
PATH = pkg_resources.resource_filename('becas_wrapper', 'test')

p = Problem(impl=impl, root=Group())
root = p.root
# --- 3 -----

# Geometry

# Number of structural calculation points
nsec_st = 8

# number of aerodynamic calculation points (not relevant for this example)
nsec_ae = 30

root.add('blade_length_c', ExecComp('blade_length = 86.366'), promotes=['*'])

pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))

# distribute structural and aerodynamic grid evenly
s_ae = np.linspace(0, 1, nsec_ae)
s_st = np.linspace(0, 1, nsec_st)

# spline the planform and interpolate onto s_ae distribution
pf = redistribute_planform(pf, s=s_ae)

# add planform spline component defined in FUSED-Wind
spl_ae = p.root.add('pf_splines', SplinedBladePlanform(pf), promotes=['*'])
spl_ae.configure()

# component for interpolating planform onto structural mesh
redist = root.add('pf_st', PGLRedistributedPlanform('_st', nsec_ae, s_st),
                  promotes=['*'])

# configure blade surface for structural solver
cfg = {}
cfg['redistribute_flag'] = False

# read the airfoil family used on the blade
afs = []
for f in [os.path.join(PATH, 'data/ffaw3241.dat'),
          os.path.join(PATH, 'data/ffaw3301.dat'),
          os.path.join(PATH, 'data/ffaw3360.dat'),
          os.path.join(PATH, 'data/cylinder.dat')]:
    afs.append(np.loadtxt(f))

cfg['base_airfoils'] = afs

# this array contains the interpolator for the lofted surface
# in this case the relative thicknesses of the airfoils loaded above
cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])

surf = root.add('blade_surf', PGLLoftedBladeSurface(cfg, size_in=nsec_st,
                size_out=(200, nsec_st, 3), suffix='_st'),
                promotes=['*'])

# read the blade structure
st3d = read_bladestructure(os.path.join(PATH, 'data/DTU10MW'))

# and interpolate onto new distribution
st3dn = interpolate_bladestructure(st3d, s_st)

# add component for generating the splined blade structure
# defined in FUSED-Wind
spl_st = root.add('st_splines', SplinedBladeStructure(st3dn),
                  promotes=['*'])
spl_st.configure()

# inputs to CS2DtoBECAS and BECASWrapper
config = {}
cfg = {}
cfg['dry_run'] = False
cfg['dominant_elsets'] = ['REGION04', 'REGION08']
cfg['max_layers'] = 10
config['CS2DtoBECAS'] = cfg
cfg = {}
cfg['hawc2_FPM'] = False
cfg['plot_paraview'] = True
cfg['dry_run'] = False
cfg['analysis_mode'] = 'stiffness'
config['BECASWrapper'] = cfg

# add the BECASBeamStructure group to the workflow
root.add('stiffness', BECASBeamStructure(root, config, st3dn,
         (200, nsec_st, 3)), promotes=['*'])

# add the recorder
from openmdao.api import SqliteRecorder, DumpRecorder
recorder = SqliteRecorder('optimization.sqlite')
p.driver.add_recorder(recorder)

# call OpenMDAO's setup method before running the problem
p.setup()

# run it
t0 = time.time()
p.run()
print 'Total time', time.time() - t0

# save the computed HAWC2 beam properties to a file
np.savetxt('hawc2_blade_st.dat', p['blade_beam_structure'])
