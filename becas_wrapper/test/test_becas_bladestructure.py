
import os
import numpy as np
import unittest

from openmdao.core.mpi_wrap import MPI
from openmdao.api import Problem, Group, IndepVarComp

from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       SplinedBladePlanform, \
                                       PGLRedistributedPlanform

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure

from becas_wrapper.becas_bladestructure import BECASBeamStructureKM,\
                                               BECASBeamStructure
from becas_wrapper.becas_stressrecovery import BECASStressRecovery

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

beam_st_FPM = np.array([[  0.00000000000000000e+00,   1.19521193885999992e+03,
         -1.58887898064000009e-04,  -3.00944017198000006e-04,
          1.76131070914999999e+00,   1.83204131143000004e+00,
         -7.51412054007000023e+01,  -9.15749989104999989e-05,
         -3.30665544272999994e-04,   2.56207659461000013e+09,
         -3.27500076546999998e+06,   0.00000000000000000e+00,
          0.00000000000000000e+00,   0.00000000000000000e+00,
         -1.16983506118999999e+07,   1.92507781420000005e+09,
          0.00000000000000000e+00,   0.00000000000000000e+00,
          0.00000000000000000e+00,   1.17021813443000009e+06,
          1.78531536877000008e+10,  -1.87896160527999991e-03,
         -1.98219605526000001e-03,   0.00000000000000000e+00,
          6.12577139919000015e+10,  -1.51824951172000009e-03,
          0.00000000000000000e+00,   6.23586204180999985e+10,
          0.00000000000000000e+00,   2.77173903102999992e+10],
       [  2.88218237909054018e+01,   5.92342555208000022e+02,
          5.86068111529447489e-01,   2.87804141909000014e-02,
          7.52477901045999986e-01,   1.55094657454999996e+00,
         -3.19152120124655259e+00,  -1.15032731063999999e-01,
          2.32411405136999996e-02,   5.90514012875000000e+08,
         -4.17756444070999995e+07,   0.00000000000000000e+00,
          0.00000000000000000e+00,   0.00000000000000000e+00,
         -7.76529514949000031e+07,   4.81881108911000013e+08,
          0.00000000000000000e+00,   8.53647602413447504e-01,
          0.00000000000000000e+00,   2.93354831842000008e+08,
          8.83676096323999977e+09,  -2.21357843469999985e-03,
         -6.62295848082000054e-04,   0.00000000000000000e+00,
          5.92741400235999966e+09,  -8.64267349243000077e-06,
          0.00000000000000000e+00,   1.92298973880000000e+10,
          0.00000000000000000e+00,   1.36293121035999990e+09],
       [  5.76426110358382431e+01,   2.79786025282000026e+02,
          4.44805718479113654e-01,   2.78619768033999994e-02,
          3.54735047108000023e-01,   8.92295246341999970e-01,
         -5.50894844469886236e-01,  -2.93689026679000015e-02,
          2.43241481304000001e-02,   2.94350752138999999e+08,
         -2.64440276513000019e+06,   0.00000000000000000e+00,
          0.00000000000000000e+00,   0.00000000000000000e+00,
         -1.15934116077999994e+07,   1.88919311620000005e+08,
          0.00000000000000000e+00,   5.75735318020113684e-01,
          0.00000000000000000e+00,   5.78236619543000013e+07,
          4.57734253118000031e+09,   2.02932082145000005e-05,
          3.06244830553000009e-05,   0.00000000000000000e+00,
          6.58516722047000051e+08,   9.12919640540999953e-05,
          0.00000000000000000e+00,   3.10491973975000000e+09,
          0.00000000000000000e+00,   1.49595113433999985e+08],
       [  8.64615413145398009e+01,   9.70035180726000057e+00,
          1.10846039556338066e-02,   6.11581711583999987e-03,
          4.19918378121000024e-02,   1.71117805302999998e-01,
         -7.91506679662466128e-01,  -6.96165596063000047e-02,
          5.67709822883000030e-03,   2.41613130614000000e+07,
          3.40329701912000019e+05,   0.00000000000000000e+00,
          0.00000000000000000e+00,   0.00000000000000000e+00,
         -4.46561547654000024e+04,   6.49156560156999994e+06,
          0.00000000000000000e+00,   8.99993052875338040e-02,
          0.00000000000000000e+00,   6.59574502506999997e+05,
          1.13053772737000003e+08,  -8.84124942719000012e-08,
          5.85442320220000005e-07,   0.00000000000000000e+00,
          2.30114001400000008e+05,   1.32699824461999993e-08,
          0.00000000000000000e+00,   3.36675987376999995e+06,
          0.00000000000000000e+00,   2.31669066137999995e+05]])

beam_st = np.array([[  0.00000000000000000e+00,   1.19521193885999992e+03,
         -1.58887898064000009e-04,  -3.00944017198000006e-04,
          1.76131070914999999e+00,   1.83204131143000004e+00,
          4.47485757119000030e-03,   2.59976211833000013e-04,
          1.26159102968999996e+10,   2.53496575090000010e+09,
          4.85559206989000014e+00,   4.94285540643999965e+00,
          1.09340081587999993e+01,   5.47860552454000005e-01,
          7.02981769565000025e-01,   1.41513004354000005e+00,
         -7.51412054007000023e+01,  -9.15749989104999989e-05,
         -3.30665544272999994e-04],
       [  2.88218237909054018e+01,   5.92342555208000022e+02,
          5.86068111529447489e-01,   2.87804141909000014e-02,
          7.52477901045999986e-01,   1.55094657454999996e+00,
          1.34445062168944762e+00,   6.95983866971000009e-02,
          8.08438947622000027e+09,   1.47805598468000007e+09,
          7.33192533561000004e-01,   2.37864559156000022e+00,
          7.98142264205000007e-01,   3.61531463509999984e-01,
          3.02239218490999995e-01,   1.09306472545000011e+00,
         -4.04516880366000020e+00,   7.38614871349447477e-01,
          2.32411405136999996e-02],
       [  5.76426110358382431e+01,   2.79786025282000026e+02,
          4.44805718479113654e-01,   2.78619768033999994e-02,
          3.54735047108000023e-01,   8.92295246341999970e-01,
          8.52590874989113678e-01,   5.49502917682999983e-02,
          1.22415043356000004e+10,   2.15250433790999985e+09,
          5.37937743593000020e-02,   2.53638740357000014e-01,
          6.10923460018000034e-02,   3.65534945624000018e-01,
          2.34901763744999992e-01,   3.73919937101000022e-01,
         -1.12663016248999992e+00,   5.46366415352213641e-01,
          2.43241481304000001e-02],
       [  8.64615413145398009e+01,   9.70035180726000057e+00,
          1.10846039556338066e-02,   6.11581711583999987e-03,
          4.19918378121000024e-02,   1.71117805302999998e-01,
          1.22198088969933810e-01,   7.39276762007000030e-03,
          7.07933313146000004e+09,   2.18528333788999987e+09,
          3.25050392637999967e-05,   4.75575850331000003e-04,
          7.52273451700999980e-05,   6.92521216036000031e-01,
          1.85835246965000006e-01,   1.59695511764000007e-02,
         -8.81505984949999988e-01,   2.03827456812337993e-02,
          5.67709822883000030e-03]])

k_11 = np.r_[1.965344e+09,   5.840942e+08,   2.942060e+08,   2.416760e+07]
k_33 = np.r_[1.785315e+10,   8.836761e+09,   4.577343e+09,   1.130538e+08]
m_11 = np.r_[1195.211939,   592.34256 ,   279.786025,     9.700352]
m_66 = np.r_[7.719385e+03,   1.789339e+03,   2.600945e+02,   3.610753e-01]


blade_beam_csprops_ref = np.array([[4.474857571200e-03,   2.599762118283e-04,   -9.157499886237e-05,
   -3.306655442824e-04,   1.195211938859e+03,   -1.588878980638e-04,
   -3.009440171980e-04,   3.992316433565e+03,   3.727068507990e+03,
   -7.404571580370e+01,   1.340715450484e-04,   -1.092717574164e-04,
    3.938231661555e+00,    2.802814511784e+00,   -3.152343526124e-01,
    1.415130043541e+00,   -1.311462270413e+00,   -1.311461438155e+00],
 [  4.908030518195e-01,   6.959836503382e-02,  -1.150327041031e-01,
    2.324113459673e-02,    5.923425595388e+02,   -2.675794703838e-01,
    2.878040953273e-02,    3.385769964145e+02,    1.450762033874e+03,
   -6.162361026397e+01,   -7.087869476120e-01,    4.144758980677e-02,
    4.334589485158e-01,    3.515283448483e+00,   -1.193268354796e-01,
    1.093064725326e+00,   -7.175630872012e-02,   -7.060151774729e-02],
 [  2.768555647326e-01,   5.495028591497e-02,   -2.936890196238e-02,
    2.432414778754e-02,    2.797860246237e+02,   -1.309295989256e-01,
    2.786197648689e-02,    3.550486639273e+01,    2.245896822866e+02,
   -4.892896266637e+00,   -4.310699863189e-01,    3.902041195802e-02,
    3.542503985710e-02,    4.195892599719e-01,   -1.412923459245e-02,
    3.739199369997e-01,   -2.098822377911e-02,   -1.966340490644e-02],
 [  3.219878368248e-02,   7.392767620068e-03,  -6.961655960636e-02,
    5.677098228827e-03,   9.700351807258e+00,  -7.891470133185e-02,
    6.115817115837e-03,   1.753947279759e-02,  3.435357828118e-01,
   -9.091536441662e-03,  -6.366067838082e-02,   6.318474747329e-03,
    3.189399797468e-05,   4.525135763937e-04,  -1.433510628713e-05,
    1.596955117636e-02,  -2.523556151881e-02,  -1.538518181341e-02]])

def configure_BECASBeamStructure(nsec, exec_mode, path_data, dry_run=False, FPM=False, with_sr=False):

    p = Problem(impl=impl, root=Group())

    p.root.add('blade_length_c', IndepVarComp('blade_length', 86.366), promotes=['*'])

    pf = read_blade_planform(os.path.join(path_data,'DTU_10MW_RWT_blade_axis_prebend.dat'))
    nsec_ae = 50
    nsec_st = nsec
    s_ae = np.linspace(0, 1, nsec_ae)
    s_st = np.linspace(0, 1, nsec_st)
    pf = redistribute_planform(pf, s=s_ae)

    spl = p.root.add('pf_splines', SplinedBladePlanform(pf), promotes=['*'])
    spl.configure()
    redist = p.root.add('pf_st', PGLRedistributedPlanform('_st', nsec_ae, s_st), promotes=['*'])

    cfg = {}
    cfg['redistribute_flag'] = False
    cfg['blend_var'] = np.array([0.241, 0.301, 0.36, 1.0])
    afs = []
    for f in ['data/ffaw3241.dat',
              'data/ffaw3301.dat',
              'data/ffaw3360.dat',
              'data/cylinder.dat']:

        afs.append(np.loadtxt(f))
    cfg['base_airfoils'] = afs
    surf = p.root.add('blade_surf', PGLLoftedBladeSurface(cfg, size_in=nsec_st,
                                    size_out=(200, nsec_st, 3), suffix='_st'), promotes=['*'])

    # read the blade structure
    st3d = read_bladestructure(os.path.join(path_data, 'DTU10MW'))

    # and interpolate onto new distribution
    st3dn = interpolate_bladestructure(st3d, s_st)

    spl = p.root.add('st_splines', SplinedBladeStructure(st3dn), promotes=['*'])
    spl.add_spline('DP04', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('r04uniax00T', np.linspace(0, 1, 4), spline_type='bezier')
    spl.add_spline('w02biax00T', np.linspace(0, 1, 4), spline_type='bezier')
    spl.configure()
    # inputs to CS2DtoBECAS and BECASWrapper
    config = {}
    cfg = {}
    cfg['dry_run'] = dry_run
    cfg['dominant_elsets'] = ['REGION04', 'REGION08']
    cfg['max_layers'] = 0
    config['CS2DtoBECAS'] = cfg
    cfg = {}
    cfg['exec_mode'] = exec_mode
    cfg['hawc2_FPM'] = FPM
    cfg['dry_run'] = dry_run
    cfg['exec_mode'] = exec_mode
    cfg['analysis_mode'] = 'stiffness'
    cfg['debug_mode'] = False
    config['BECASWrapper'] = cfg

    p.root.add('stiffness', BECASBeamStructure(p.root, config, st3dn, (200, nsec_st, 3)), promotes=['*'])
    p.root.add('stress_recovery', BECASStressRecovery(config, s_st, 2), promotes=['*'])

    p.setup()
    p['hub_radius'] = 2.8
    for k, v in pf.iteritems():
        if k in p.root.pf_splines.params.keys():
            p.root.pf_splines.params[k] = v

    # set some arbitrary values in the load vectors used to compute strains
    for i, x in enumerate(s_st):
        try:
            p['load_cases_sec%03d' % i] = np.ones((2, 6))
        except:
            pass

    return p

def configure_BECASBeamStructureKM(nsec, exec_mode):
    
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

class BECASBeamStructureTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_dry_run(self):
    #
    #     p = configure(4, True)
    #     p.run()

    def test_standard_octave(self):
        p = configure_BECASBeamStructure(4, 'octave', 'data', False, False)
        p.run()
  
        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_structure'][:,1:]/beam_st[:,1:], np.ones((4,18)), decimal=6), None)
  
        self.assertAlmostEqual(p['blade_mass']/42499.350315582917, 1.e0, places=6)
        self.assertAlmostEqual(p['blade_mass_moment']/10670946.166707618, 1.e0, places=6)
        
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][0,0,:], k_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][2,2,:], k_33, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][0,0,:], m_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][5,5,:], m_66, 1E-6), None)
        
        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_csprops_ref'][:,:]/blade_beam_csprops_ref[:,:], np.ones((4,18)), decimal=6), None)
        
  
        # when hooked up to a constraint these outputs ought to be
        # available on all procs
        if not MPI:
  
            self.assertAlmostEqual(p['blade_failure_index_sec000'][0], 0.17026370426021892, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec001'][0], 0.16552789587300576, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec002'][0], 0.16292259732314465, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec003'][0], 0.15931231052281988, places=6)
    
    def test_standard_octave_data_version_1(self):
        p = configure_BECASBeamStructure(4, 'octave', 'data_version_1', False, False)
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_structure'][:,1:]/beam_st[:,1:], np.ones((4,18)), decimal=6), None)

        self.assertAlmostEqual(p['blade_mass']/42499.350315582917, 1.e0, places=6)
        self.assertAlmostEqual(p['blade_mass_moment']/10670946.166707618, 1.e0, places=6)
        
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][0,0,:], k_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][2,2,:], k_33, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][0,0,:], m_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][5,5,:], m_66, 1E-6), None)

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_csprops_ref'][:,:]/blade_beam_csprops_ref[:,:], np.ones((4,18)), decimal=6), None)

        # when hooked up to a constraint these outputs ought to be
        # available on all procs
        if not MPI:

            self.assertAlmostEqual(p['blade_failure_index_sec000'][0], 0.17026370426021892, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec001'][0], 0.16552789587300576, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec002'][0], 0.16292259732314465, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec003'][0], 0.15931231052281988, places=6)
    
    
    @unittest.skipIf(not _matlab_installed,
                 "Matlab not available on this system")
    def test_standard_matlab(self):
        p = configure_BECASBeamStructure(4, 'matlab', 'data', False, False)
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_structure'][:,1:]/beam_st[:,1:], np.ones((4,18)), decimal=6), None)

        self.assertAlmostEqual(p['blade_mass']/42499.350315582917, 1.e0, places=6)
        self.assertAlmostEqual(p['blade_mass_moment']/10670946.166707618, 1.e0, places=6)
        
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][0,0,:], k_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][2,2,:], k_33, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][0,0,:], m_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][5,5,:], m_66, 1E-6), None)

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_csprops_ref'][:,:]/blade_beam_csprops_ref[:,:], np.ones((4,18)), decimal=6), None)


        # when hooked up to a constraint these outputs ought to be
        # available on all procs
        if not MPI:

            self.assertAlmostEqual(p['blade_failure_index_sec000'][0], 0.17026370426021892, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec001'][0], 0.16552789587300576, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec002'][0], 0.16292259732314465, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec003'][0], 0.15931231052281988, places=6)

    # def test_FPM(self):
    #
    #     p = configure(4, False, True)
    #     p.setup()
    #     p.run()
    #     self.assertEqual(np.testing.assert_array_almost_equal(p['beam_structure'], beam_st_FPM, decimal=6), None)

    # def test_stiffness_and_stress_recovery_run(self):
    #
    #     p = configure(4, False, False, True)
    #     p.setup()
    #     p.run()
    #     self.assertEqual(np.testing.assert_array_almost_equal(p['beam_structure'], beam_st, decimal=4), None)


class BECASBeamStructureKMTestCase(unittest.TestCase):

    def tearDown(self):
        pass
        
    def test_becas_bladestructure_KM_octave(self):
        p = configure_BECASBeamStructureKM(4, 'octave')
        p.run()
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][0,0,:], k_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][2,2,:], k_33, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][0,0,:], m_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][5,5,:], m_66, 1E-6), None)
        
    @unittest.skipIf(not _matlab_installed,
                     "Matlab not available on this system")
    def test_becas_bladestructure_KM_matlab(self):
        p = configure_BECASBeamStructureKM(4, 'matlab')
        p.run()
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][0,0,:], k_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['KStruct'][2,2,:], k_33, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][0,0,:], m_11, 1E-6), None)
        self.assertEqual(np.testing.assert_allclose(p['MStruct'][5,5,:], m_66, 1E-6), None)
        
if __name__ == "__main__":
    unittest.main()
    #p = configure_BECASBeamStructure(4, 'matlab', 'data', False, False)
    #p.run()
    #np.set_printoptions(precision=12)
    #print(p['blade_beam_csprops_ref'][:,:])

