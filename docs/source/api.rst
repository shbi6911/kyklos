API Reference
=============

This page lists the public classes and functions of Kyklos, grouped by purpose.
Every name below is available from the top-level package (``import kyklos as
ky``). Short aliases ``OE``, ``Sat``, and ``Traj`` are provided for
``OrbitalElements``, ``Satellite``, and ``Trajectory``. For a narrative
description of each source module, see :doc:`modules`.

Core classes
------------

.. autosummary::
   :toctree: generated

   kyklos.OrbitalElements
   kyklos.System
   kyklos.TwoBodySystem
   kyklos.CR3BPSystem
   kyklos.Trajectory
   kyklos.Satellite

Supporting types
----------------

.. autosummary::
   :toctree: generated

   kyklos.BodyParams
   kyklos.AtmoParams
   kyklos.OEType
   kyklos.SysType

Trajectory nodes
----------------

.. autosummary::
   :toctree: generated

   kyklos.Node
   kyklos.BoundaryNode
   kyklos.StartBoundaryNode
   kyklos.EndBoundaryNode
   kyklos.ImpulsiveBoundaryNode
   kyklos.JunctionNode
   kyklos.NullJunctionNode
   kyklos.ImpulsiveJunctionNode
   kyklos.FreeJunctionNode

Differential correction
-----------------------

.. autosummary::
   :toctree: generated

   kyklos.DifferentialCorrector
   kyklos.ShooterResult
   kyklos.NodeSpec

Constraints
^^^^^^^^^^^

.. autosummary::
   :toctree: generated

   kyklos.Constraint
   kyklos.TerminalConstraint
   kyklos.FreeVarConstraint
   kyklos.TargetState
   kyklos.Periodicity
   kyklos.JacobiConstraint
   kyklos.CallableConstraint
   kyklos.PseudoArclength
   kyklos.FreeVarPin
   kyklos.PhaseConstraint

CR3BP periodic orbits and continuation
--------------------------------------

.. autosummary::
   :toctree: generated

   kyklos.PeriodicOrbit
   kyklos.OrbitFamily
   kyklos.CorrectorGuess
   kyklos.SeederResult
   kyklos.correct_as
   kyklos.march_family
   kyklos.available_recipes
   kyklos.available_layouts
   kyklos.available_schemes

Default systems
---------------

.. autosummary::
   :toctree: generated

   kyklos.earth_2body
   kyklos.earth_j2
   kyklos.earth_drag
   kyklos.earth_moon_cr3bp
   kyklos.earth_sun_cr3bp
   kyklos.moon_2body
   kyklos.moon_j2
   kyklos.mars_2body
   kyklos.mars_j2

Bodies and atmosphere
---------------------

.. autosummary::
   :toctree: generated

   kyklos.mercury
   kyklos.venus
   kyklos.earth
   kyklos.moon
   kyklos.mars
   kyklos.jupiter
   kyklos.saturn
   kyklos.uranus
   kyklos.neptune
   kyklos.sun
   kyklos.EARTH_STD_ATMO

Default and constructed orbits
------------------------------

.. autosummary::
   :toctree: generated

   kyklos.iss_orbit
   kyklos.geo_orbit
   kyklos.leo_orbit
   kyklos.sso_orbit
   kyklos.default_molniya_orbit
   kyklos.lyapunov_orbit
   kyklos.gateway_orbit
   kyklos.circular_orbit
   kyklos.synchronous_orbit
   kyklos.molniya_orbit
   kyklos.sun_synchronous_orbit

Configuration and utilities
---------------------------

.. autosummary::
   :toctree: generated

   kyklos.config.KyklosConfig
   kyklos.temp_config
   kyklos.Timer

Exceptions
----------

.. autosummary::
   :toctree: generated

   kyklos.exceptions.KyklosError
   kyklos.exceptions.CorrectionError
   kyklos.ConvergenceError
   kyklos.exceptions.ClosureError
