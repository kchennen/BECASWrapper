
Computing Beam Properties For a Blade
=====================================

In this example we'll demonstrate how to use ``BECASWrapper`` to compute
the beam properties of a wind turbine blade defined both in terms of the
outer mold line geometry as well as the detailed internal structural geometry
and material properties.

The example is located in ``becas_wrapper/examples/compute_beamprops.py``.



First step is to make all the necessary imports of the classes we'll use
in this example.
As you can see, in addition to the necessary OpenMDAO classes,
a number of classes and methods are imported from FUSED-Wind.
Before getting started with this example, you're therefore encouraged
to familiarize
yourself with the turbine geometry and structure classes defined in
``FUSED-Wind``, since ``BECASWrapper`` relies on this code
for generating defining and splining the blade structure.

Since OpenMDAO can take advantage of all the processors on your machine using
MPI, this example makes a check of whether we're inside an MPI environment
or not.

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 1
    :end-before: # --- 2

Next step is to define the lofted blade geometry, which is done entirely using
methods and classes defined in FUSED-Wind.
The `nsec_st` parameter defines how many cross-sections we want to compute
properties for along the blade.
You can change this according to the system you're on and how patient you're feeling.

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 2
    :end-before: # --- 3

Next, the internal structure defining material properties and material stacking
sequences is read from the data for the DTU 10MW RWT located in
`becas_wrapper/test/data`.
Since the data is defined with high resolution, we need to interpolate the
properties onto the resolution defined by `s_st` using ``interpolate_bladestructure``.
The ``SplinedBladeStructure`` class allows you to attach splines to all the
structural properties either for manual modification of for an optimizer to control.

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 3
    :end-before: # --- 4

Finally, we arrive at the step of adding the ``BECASBeamStructure`` class to
our problem.
This class wraps two underlying tools: *shellexpander* and *BECAS*.
In ``BECASWrapper`` we've defined a class which converts the inputs defined
using FUSED-Wind into the format used by *shellexpander*, called ``CS2DtoBECAS``.
This class serves the same purpose as the tool *airfoil2becas*, which is distributed
as part of the *BECAS* package.
If you've used *shellexpander* there are some inputs that you'll recognize,
such as ``dominant_elsets`` which controls how two neighbouring regions in the
cross-section are joined.
In this case, we define the spar cap regions Region 4 and 8 as dominant.
``max_layers`` defines how many cells are used to discretize the structure
in the thickness direction.
If you don't set this, the number of cells will be equal to the maximum number
of materials in all of the regions in the cross-section.

BECAS can output beam properties with a fully populated 6x6 stiffness matrix,
which is now supported aeroelastic codes such as HAWC2 and FAST 8.
BECAS also has a convenience function for outputting properties using the
standard HAWC2 format.
Set the `hawc2_FPM` parameter to `True` to output the full 6x6 beam properties.

BECAS can also generate plotting files for inspecting the mesh and solution
using Paraview.
Set the `plot_paraview` to True to generate these plotting files.

The underlying BECASWrapper can be run either for computing cross-sectional
properties or recovering stresses.
In this example we're using it to compute properties, so set the
``analysis_mode`` input to ``stiffness``.

The ``BECASBeamStructure`` class needs to be instantiated with a number
of parameters: the `config` dictionary described above, the full blade structure
dictionary for adding the needed input and output parameters, and finally
the size of the input blade surface, in this case `(200, nsec_st, 3)`.

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 4
    :end-before: # --- 5

In the final step we add an OpenMDAO recorder to save all the data, and call
OpenMDAO's `Problem.setup` method.

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 5
    :end-before: # --- 6
    
The recorded data can be accessed with sqlitedict methods:

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 6
    :end-before: # --- 7
    
and be plotted in the following manner:

.. literalinclude:: ../becas_wrapper/examples/compute_beamprops.py
    :start-after: # --- 7
    :end-before: # --- 8

If you have MPI installed, you can execute the code using `mpirun`:

.. code-block:: bash

    $ mpirun -np 4 python compute_beamprops.py

or to run in serial, simply:

.. code-block:: bash

    python compute_beamprops.py

If you would like to inspect the problem interactively you can also use
*iPython* to run the problem.
