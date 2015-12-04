
import time
import os
import numpy as np
import unittest

from openmdao.core.mpi_wrap import MPI
from openmdao.api import Problem, Group, IndepVarComp

from fusedwind.turbine.structure import read_bladestructure, \
                                        interpolate_bladestructure, \
                                        SplinedBladeStructure
from becas_wrapper.becas_bladestructure import BECASBeamStructure
from becas_wrapper.becas_stressrecovery import BECASStressRecovery

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

from fusedwind.turbine.geometry import read_blade_planform,\
                                       redistribute_planform,\
                                       PGLLoftedBladeSurface,\
                                       SplinedBladePlanform, \
                                       PGLRedistributedPlanform



def configure(nsec, exec_mode, dry_run=False, FPM=False, with_sr=False):

    p = Problem(impl=impl, root=Group())

    p.root.add('blade_length_c', IndepVarComp('blade_length', 86.366), promotes=['*'])

    pf = read_blade_planform('data/DTU_10MW_RWT_blade_axis_prebend.dat')
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
    st3d = read_bladestructure('data/DTU10MW')

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
    cfg['hawc2_FPM'] = FPM
    cfg['dry_run'] = dry_run
    cfg['analysis_mode'] = 'stiffness'
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

class BECASWrapperTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # def test_dry_run(self):
    #
    #     p = configure(4, True)
    #     p.run()

    def test_standard_octave(self):
        p = configure(4, 'octave', False, False)
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_structure'], beam_st, decimal=4), None)

        self.assertAlmostEqual(p['blade_mass'], 42499.350315582917, places=6)
        self.assertAlmostEqual(p['blade_mass_moment'], 10670946.166707618, places=6)


        # when hooked up to a constraint these outputs ought to be
        # available on all procs
        if not MPI:

            self.assertAlmostEqual(p['blade_failure_index_sec000'][0], 0.17026370426021892, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec001'][0], 0.16552789587300576, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec002'][0], 0.16292259732314465, places=6)
            self.assertAlmostEqual(p['blade_failure_index_sec003'][0], 0.15931231052281988, places=6)
    
    def test_standard_matlab(self):
        p = configure(4, 'matlab', False, False)
        p.run()

        self.assertEqual(np.testing.assert_array_almost_equal(p['blade_beam_structure'][:,1:]/beam_st[:,1:], np.ones((4,18)), decimal=6), None)

        self.assertAlmostEqual(p['blade_mass']/42499.350315582917, 1.e0, places=6)
        self.assertAlmostEqual(p['blade_mass_moment']/10670946.166707618, 1.e0, places=6)


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


if __name__ == "__main__":

    unittest.main()
    # import time
    # t0 = time.time()
    # p = configure(4, False, True)
    # t1 = time.time()
    # p.run()
    # t2 = time.time()
    # if p.root.comm.rank == 0:
    #     print 'Total time:', t2-t0
    #     print 'configure time:', t1-t0
    #     print 'run time:', t2-t1
    #     print('mass %f'%p['blade_mass'])
