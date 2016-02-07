
# --- 1 -----

import os
import time
import numpy as np
import unittest
import pkg_resources

from openmdao.core.mpi_wrap import MPI
from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.api import SqliteRecorder

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

# --- 2 -----

# geometry

# number of structural calculation points
nsec_st = 8

# number of aerodynamic calculation points (not relevant for this example)
nsec_ae = 30

# distribute structural and aerodynamic grid evenly
s_ae = np.linspace(0, 1, nsec_ae)
s_st = np.linspace(0, 1, nsec_st)

# add an ExecComp defining blade length TODO: this shouldn't be necessary
root.add('blade_length_c', ExecComp('blade_length = 86.366'), promotes=['*'])

# read the blade planform into a simple dictionary format
pf = read_blade_planform(os.path.join(PATH, 'data/DTU_10MW_RWT_blade_axis_prebend.dat'))

# spline the planform and interpolate onto s_ae distribution
pf = redistribute_planform(pf, s=s_ae)

# add planform spline component defined in FUSED-Wind
spl_ae = p.root.add('pf_splines', SplinedBladePlanform(pf), promotes=['*'])

# this method adds IndepVarComp's for all pf quantities
# in no splines are defined, see FUSED-Wind docs for more details
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

# --- 3 -----

# read the blade structure
st3d = read_bladestructure(os.path.join(PATH, 'data/DTU10MW'))

# and interpolate onto new distribution
st3dn = interpolate_bladestructure(st3d, s_st)

# add component for generating the splined blade structure
# defined in FUSED-Wind
spl_st = root.add('st_splines', SplinedBladeStructure(st3dn),
                  promotes=['*'])
spl_st.configure()

# --- 4 -----

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

# --- 5 -----

# add the recorder
recorder = SqliteRecorder('optimization.sqlite')
recorder.options['record_params'] = True
recorder.options['record_metadata'] = True
p.driver.add_recorder(recorder)

# call OpenMDAO's setup method before running the problem
p.setup()

# run it
t0 = time.time()
p.run()
p.driver.recorders[0].close()
print 'Total time', time.time() - t0

# save the computed HAWC2 beam properties to a file
np.savetxt('hawc2_blade_st.dat', p['blade_beam_structure'])

# --- 6 -----

# access recorded data
import sqlitedict
# load data base
db = sqlitedict.SqliteDict('optimization.sqlite', 'openmdao')
u = db['Driver/1']['Unknowns']
p = db['Driver/1']['Parameters']

# --- 7 -----

# plot recorded data
import matplotlib.pylab as plt
plt.figure()
plt.title('csprops_ref') # w.r.t reference coordinate system
plt.plot(u['s_st'], u['blade_beam_csprops_ref'][:,0], 'o-', label = 'ShearX')
plt.plot(u['s_st'], u['blade_beam_csprops_ref'][:,2], 'o-', label = 'ElasticX')
plt.plot(u['s_st'], u['blade_beam_csprops_ref'][:,5], 'o-', label = 'MassX')
plt.legend(loc='best')

plt.figure()
plt.title('KStruct') # w.r.t reference coordinate system
plt.plot(u['s_st'], u['KStruct'][0,0,:], 'o-', label = 'K11') #kGAx
plt.plot(u['s_st'], u['KStruct'][1,1,:], 'o-', label = 'K22') #kGAy
plt.plot(u['s_st'], u['KStruct'][2,2,:], 'o-', label = 'K33') #AE
plt.plot(u['s_st'], u['KStruct'][3,3,:], 'o-', label = 'K44') #EIx
plt.plot(u['s_st'], u['KStruct'][4,4,:], 'o-', label = 'K55') #EIy
plt.plot(u['s_st'], u['KStruct'][5,5,:], 'o-', label = 'K66') #GJ
plt.legend(loc='best')

plt.figure()
plt.title('cs_props') # HAWC2 props
plt.plot(u['s_st'], u['blade_beam_structure'][:,2], 'o-', label = 'x_cg')
plt.plot(u['s_st'], u['blade_beam_structure'][:,6], 'o-', label = 'x_sh')
plt.plot(u['s_st'], u['blade_beam_structure'][:,17], 'o-', label = 'x_e')
plt.legend(loc='best')
plt.show()