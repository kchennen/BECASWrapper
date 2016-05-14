
import unittest
import os

from becas_wrapper.cs2dtobecas import CS2DtoBECAS
from becas_wrapper.becas_wrapper import BECASWrapper
import numpy as np

class BECASClosedCrossSection(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        os.remove('./becas_section.m')
        os.remove('./BECAS_SetupPath.m')
        os.remove('./becas_utils0.500.mat')
        os.remove('./airfoil_abaqus.inp')

    def test_closed_section(self):

        cs2d={}
        #'matprops' is a numpy array, with shape: (N_Matl, 10)
        cs2d['matprops']=np.reshape(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),(1,10));
        #'DPs' is a numpy array, with shape: (N_Reg+1,)
        cs2d['DPs']=np.array([0.0,1.0]);
        #'failmat' is a numpy array, with shape: (N_Matl, 23)
        cs2d['failmat']=np.reshape(np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]),(1,23));
        #'regions' is a list with size N_Reg with the following contents:
        #	List entry 0 is a dictionary with size 3 with the following contents:
        regions=[]
        reg1={}
        mat=[]
        mat.append('materialA00')
        #		'layers' is a list with size 1 with the following contents:
        reg1['layers']=mat;
        #		'thicknesses' is a numpy array, with shape: (1,)
        reg1['thicknesses']=np.array([0.01]);
        #		'angles' is a numpy array, with shape: (1,)
        reg1['angles']=np.array([0.0]);
        regions.append(reg1);
        cs2d['regions']=regions;
        #'webs' is a list with size 3 with the following contents:
        #	List entry 0 is a dictionary with size 3 with the following contents:
        regions=[]
        cs2d['webs']=regions;
        #'s' is a float with value: 0.75
        cs2d['s']=0.5;
        #'materials' is a dictionary with size N_Matl with the following contents:
        matl_name_list={}
        matl_name_list['materialA']=0
        cs2d['materials']=matl_name_list
        #'coords' is a numpy array, with shape: (200, 2)
        cs2d['coords']=np.reshape(np.array([1.0,1.0, 0.0,1.0, 0.0, 0.0,1.0, 0.0,1.0,1.0]),(5,2));
        #'failcrit' is a list with size N_Matl with the following contents:
        my_fail_criteria=[]
        my_fail_criteria.append('maximum_strain')
        cs2d['failcrit']=my_fail_criteria
        #'web_def' is a list with size 3 with the following contents:
        web_list=[]
        cs2d['web_def']=web_list

        names=['s','m','x_cog','y_cog','rad_of_gyration_x','rad_of_gyration_y','shear_x','shear_y','E','G','Ix','Iy','K','shear_k_x','shear_k_y','A','theta','x_elastic','y_elastic']

        config={}

        # This sets up the configuration for BECAS
        cfg = {}
        cfg['dry_run'] = False
        cfg['dominant_elsets'] = []
        cfg['max_layers'] = 10
        cfg['spline_type'] = 'linear'
        config['CS2DtoBECAS'] = cfg
        # This sets up the configuration for BECAS
        cfg = {}
        cfg['hawc2_FPM'] = False
        cfg['plot_paraview'] = True
        cfg['dry_run'] = False
        cfg['analysis_mode'] = 'stiffness'

        config['BECASWrapper'] = cfg

        mesher = CS2DtoBECAS(cs2d, **config['CS2DtoBECAS'])
        becas = BECASWrapper(0.5, **config['BECASWrapper'])
     
        mesher.cs2d=cs2d
        mesher.cs2d['DPs'][0]=-1.0
        mesher.compute()
        becas.compute()
 
        self.assertAlmostEqual(becas.cs_props[ 0]/0.500000000000000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 1]/0.039248528148600, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 2]/0.500000000072000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 3]/0.500000000072000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 4]/0.403378494624000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 5]/0.403378494624000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 6]/0.500000000251000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 7]/0.500000001100000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 8]/0.999999999894000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[ 9]/1.000000000000000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[10]/0.006388837421020, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[11]/0.006388837429090, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[12]/0.009719898363820, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[13]/0.419788921835000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[14]/0.419788902738000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[15]/0.039248528148600, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[16]/84.80778818780000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[17]/0.500000000083000, 1.e0, places=6)
        self.assertAlmostEqual(becas.cs_props[18]/0.500000000068000, 1.e0, places=6)

if __name__ == "__main__":
    unittest.main()

