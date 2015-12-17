
import os
import numpy as np
import unittest

from openmdao.core.mpi_wrap import MPI
from openmdao.api import Problem, Group

from becas_wrapper.becas_bladestructure import BECASBeamStructureKM
from distutils.spawn import find_executable


_matlab_installed = find_executable('matlab')

# stuff for running in parallel under MPI
def mpi_print(prob, *args):
    """ helper function to only print on rank 0"""
    if prob.root.comm.rank == 0:
        print(args)

if MPI:
    # if you called this script with 'mpirun', then use the petsc data passing
    from openmdao.core.petsc_impl import PetscImpl as impl
else:
    # if you didn't use `mpirun`, then use the numpy data passing
    from openmdao.core.basic_impl import BasicImpl as impl


k_11 = np.r_[1.96534355e+09, 2.41676013e+07]
k_33 = np.r_[1.78531537e+10, 1.13053773e+08]
m_11 = np.r_[1195.21193886, 9.70035181]
m_66 = np.r_[7.71938494e+03, 3.61075256e-01]

def configure(nsec, exec_mode):
    user_home = os.getenv('HOME')
    
    p = Problem(impl=impl, root=Group())
    
    config = {}
    
    cfg = {}
    cfg['exec_mode'] = exec_mode
    cfg['analysis_mode'] = 'stiffness'
    cfg['debug_mode'] = False
    cfg['plot_paraview'] = False
    cfg['hawc2_FPM'] = False
    cfg['plot_paraview'] = False
    config['BECASWrapper'] = cfg

    path_data = 'data/BECAS_inputs'
    path_sections = os.path.join(path_data, 'shellexpander_sections.log')
    
    becasInp = {}
    becasInp['s'] = np.loadtxt(path_sections, usecols = (1,))
    becasInp['nsec'] = nsec
    becasInp['path_input_folders'] = []
    
    becas_input_folders = np.loadtxt(path_sections, usecols = (0,), dtype=np.str) 
    for j in range(nsec):
        becasInp['path_input_folders'].append(os.path.join(path_data, becas_input_folders[j]))
        
    p.root.add('stiffness', BECASBeamStructureKM(p.root, config, becasInp), promotes=['*'])
    p.setup()
    
    return p

class BECASBladeStructureTestCase(unittest.TestCase):

    def tearDown(self):
        pass
        
    def test_becas_bladestructure_KM_octave(self):
        self.p = configure(2, 'octave')
        self.p.run()
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['KStruct'][0,0,:], k_11), None)
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['KStruct'][2,2,:], k_33), None)
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['MStruct'][0,0,:], m_11), None)
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['MStruct'][5,5,:], m_66), None)
        
    @unittest.skipIf(not _matlab_installed,
                     "Matlab not available on this system")
    def test_becas_bladestructure_KM_matlab(self):
        self.p = configure(2, 'matlab')
        self.p.run()
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['KStruct'][0,0,:], k_11), None)
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['KStruct'][2,2,:], k_33), None)
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['MStruct'][0,0,:], m_11), None)
        self.assertEqual(np.testing.assert_allclose(self.p.root.unknowns['MStruct'][5,5,:], m_66), None)
        
if __name__ == "__main__":
    unittest.main()