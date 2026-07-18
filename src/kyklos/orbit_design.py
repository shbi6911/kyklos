'''
Convenience functions to return specific forms of orbits within the 2BP and 
perturbed 2BP.  Exposed to users with arguments for any Keplerian orbital
elements not fixed by the orbit type (e=0.0 for circular/synchronous, etc.)

These functions work for any input BodyParams, they are used by defaults.py
to define default Earth orbits.
'''

import numpy as np
from .system import BodyParams
from .orbital_elements import OrbitalElements
from .utils import validation_error

def circular_orbit(body: BodyParams, altitude: float,
                i: float = 0.0, omega: float = 0.0, nu: float = 0.0) -> OrbitalElements:
    """
    Construct a circular orbit at a given altitude above a body.

    The semi-major axis is set to the body's equatorial radius plus the
    requested altitude, with zero eccentricity. Because a circular orbit has
    no periapsis, the argument of periapsis is undefined and is pinned to
    zero; the true anomaly ``nu`` then acts as the argument of latitude
    (angular position measured from the ascending node).

    Parameters
    ----------
    body : BodyParams
        Central body. Only ``radius`` and ``mu`` are used.
    altitude : float
        Altitude above the body's equatorial radius [km]. Must be positive.
    i : float, optional
        Inclination [rad], measured from the body's north pole. Default 0.0
        (equatorial).
    omega : float, optional
        Right ascension of the ascending node (RAAN) [rad]. Default 0.0.
    nu : float, optional
        True anomaly [rad]; for a circular orbit this is the argument of
        latitude. Default 0.0.

    Returns
    -------
    OrbitalElements
        Keplerian elements for the circular orbit, tagged with ``mu = body.mu``.

    Notes
    -----
    This is a pure two-body (Keplerian) construction. The returned elements
    describe an ideal osculating orbit and include no perturbations; a real
    orbit at this altitude will depart from these elements over time under
    oblateness, drag, and third-body effects.
    """
    if altitude <= 0.0:
        validation_error(
            f"Altitude is not positive (orbit above the surface), got {altitude}."
        )
    a = body.radius + altitude

    return OrbitalElements(a=a, e=0.0, i=i, omega=omega,
                           w=0.0, nu=nu, mu=body.mu)


def synchronous_orbit(body: BodyParams, i: float | None = None,
                      omega: float = 0.0, nu: float = 0.0) -> OrbitalElements:
    """
    Construct a synchronous (period-matched) circular orbit.

    A synchronous orbit has an orbital period equal to the body's sidereal
    rotation period. This condition fixes only the semi-major axis,

        a = (mu / rotation_rate^2)^(1/3),

    which depends on the magnitude of the rotation rate but not its sign, so a
    retrograde rotator has the same synchronous radius as a prograde one at the
    same rate. Inclination is left free: any inclination yields a valid
    synchronous orbit (an inclined synchronous orbit traces a closed
    figure-eight ground track rather than holding a fixed point).

    When ``i`` is not supplied, the default is the *stationary* orbit -- the
    single inclination at which the satellite remains fixed over one point of
    the rotating body. That requires the orbit to share the body's spin
    direction: ``i = 0`` for a prograde rotator, ``i = pi`` for a retrograde
    rotator (negative ``rotation_rate``). Supplying ``i`` explicitly overrides
    this and produces a non-stationary synchronous orbit at that inclination.

    Parameters
    ----------
    body : BodyParams
        Central body. Requires a defined, nonzero ``rotation_rate``.
    i : float or None, optional
        Inclination [rad]. If None (default), the stationary inclination is
        chosen from the sign of ``rotation_rate`` (0.0 prograde, pi retrograde).
        Any value in [0, pi] is a valid synchronous orbit.
    omega : float, optional
        Right ascension of the ascending node (RAAN) [rad]. Default 0.0.
    nu : float, optional
        True anomaly [rad]; the argument of latitude for this circular orbit.
        Default 0.0.

    Returns
    -------
    OrbitalElements
        Keplerian elements for the synchronous orbit, tagged with
        ``mu = body.mu``.

    Raises
    ------
    ValueError
        If ``body.rotation_rate`` is None (rotation undefined) or exactly zero
        (the synchronous radius diverges).

    Notes
    -----
    This is a pure two-body construction. The period match and, for the default
    case, the stationary condition are exact only in the Keplerian
    approximation; under J2 and higher perturbations a real synchronous orbit
    precesses and requires station-keeping to hold its ground track. The
    geostationary orbit is the ``i = 0`` special case of this function for a
    prograde body.
    """
    if body.rotation_rate is None:
        raise ValueError(
            f"Body {body.name!r} has no rotation_rate; a synchronous orbit is "
            "undefined."
        )
    if body.rotation_rate == 0.0:
        raise ValueError(
            f"Body {body.name!r} has zero rotation_rate; no synchronous orbit "
            "exists (infinite semi-major axis)."
        )

    omega_body = body.rotation_rate
    a = (body.mu / omega_body**2) ** (1.0 / 3.0)

    if i is None:
        i = np.pi if omega_body < 0.0 else 0.0

    return OrbitalElements(a=a, e=0.0, i=i, omega=omega, w=0.0, nu=nu, mu=body.mu)


def molniya_orbit(body: BodyParams, perigee_alt: float = 600,
                  omega: float = 0.0, w: float = 3.0 * np.pi / 2.0,
                  nu: float = 0.0) -> OrbitalElements:
    """
    Construct a Molniya orbit: half-sidereal-day, critically inclined, eccentric.

    A Molniya orbit combines three features:

    - a period of half the body's sidereal rotation period, fixing

          a = (mu / (2 * rotation_rate)^2)^(1/3);

    - the critical inclination, at which the secular apsidal drift from J2
      vanishes,

          4 - 5 sin^2(i) = 0  ->  i = arcsin(2 / sqrt(5)) ~ 63.435 deg;

    - a high eccentricity with the argument of periapsis placed so that apogee
      dwells over one hemisphere.

    The semi-major axis and inclination are fixed by the body; perigee altitude and
    the orientation angles are inputs. The necessary eccentricity is derived from
    the required semimajor axis and input perigee altitude.  The argument of periapsis 
    defaults to 3*pi/2 (270 deg), which parks apogee over the northern hemisphere, the
    configuration the orbit exists to provide. Because the inclination is
    critical, this argument of periapsis does not drift under J2 perturbation.

    Parameters
    ----------
    body : BodyParams
        Central body. Requires a defined, nonzero ``rotation_rate`` and a
        defined, nonzero ``J2``.
    perigee_alt : float, optional
        Altitude at perigee [km], usually chosen to minimize atmospheric drag at
        perigee.  The classic value of ~600 km results in the usual eccentricity of
        ~ 0.737.
    omega : float, optional
        Right ascension of the ascending node (RAAN) [rad]. Default 0.0.
    w : float, optional
        Argument of periapsis [rad]. Default 3*pi/2, placing apogee over the
        northern hemisphere.
    nu : float, optional
        True anomaly [rad]. Default 0.0.

    Returns
    -------
    OrbitalElements
        Keplerian elements for the Molniya orbit, tagged with ``mu = body.mu``.

    Raises
    ------
    ValueError
        If ``body.rotation_rate`` is None or zero (no half-sidereal-day orbit),
        if ``body.J2`` is None or zero (the critical inclination is undefined
        without oblateness), or if ``e`` is outside [0, 1).

    Notes
    -----
    J2 enters only to *derive* the critical inclination; the returned elements
    are a pure two-body set and carry no perturbation model. The apsidal-drift
    cancellation that motivates the critical inclination is realized only when
    these elements are propagated in a force model that includes J2 -- under
    pure two-body dynamics there is no drift to cancel. The critical inclination
    value depends on J2 being nonzero but not on its magnitude.

    Note that the required eccentricity is derived by adding the input perigee
    altitude to the body.radius of the input body, which is likely the equatorial
    radius.  Higher-fidelity applications should consider calculating the Molniya
    elements independently of this function.
    """
    if body.rotation_rate is None:
        raise ValueError(
            f"Body {body.name!r} has no rotation_rate; a Molniya orbit is "
            "undefined."
        )
    if body.rotation_rate == 0.0:
        raise ValueError(
            f"Body {body.name!r} has zero rotation_rate; no half-sidereal-day "
            "orbit exists."
        )
    if not body.J2:
        validation_error(
            f"Molniya orbit does not technically require nonzero J2, (got {body.J2}), "
            f"but the critical inclination is undefined without an oblateness term."
            )

    if perigee_alt < 0:
        validation_error(f"Input perigee altitude is negative, i.e. below the surface "
                         f"of the central body. Consider if this orbit is physically "
                         f"relevant.")

    crit_i = np.arcsin(np.sqrt(4 / 5))

    omega_body = body.rotation_rate
    a = (body.mu / (2.0 * omega_body) ** 2) ** (1.0 / 3.0)

    e = 1 - ((perigee_alt + body.radius) / a)

    if not 0.0 <= e < 1.0:
        raise ValueError(f"Derived eccentricity is outside [0, 1), got {e}. "
                         f"Reevaluate the input perigee altitude.")

    return OrbitalElements(a=a, e=e, i=crit_i, omega=omega, w=w, nu=nu, mu=body.mu)


def sun_synchronous_orbit(body: BodyParams, a: float, e: float = 0.0,
                          omega: float = 0.0, w: float = 0.0,
                          nu: float = 0.0, *, node_rate : float) -> OrbitalElements:
    """
    Construct a sun-synchronous orbit by solving for its inclination.

    A sun-synchronous orbit is one whose orbital plane precesses at a
    prescribed rate -- for a true sun-synchronous orbit, the body's mean
    heliocentric rate, so that the plane maintains a fixed orientation to the
    Sun. The J2 secular nodal regression is

        node_rate = -(3/2) * n * J2 * (radius / p)^2 * cos(i),

    with mean motion n = sqrt(mu / a^3) and semi-latus rectum p = a*(1 - e^2).
    Given the semi-major axis, eccentricity, and target ``node_rate``, this is
    solved for the inclination:

        cos(i) = -(2/3) * (node_rate / (J2 * n)) * (p / radius)^2.

    The semi-major axis and eccentricity are design inputs (they set the mean
    motion and the semi-latus rectum); the inclination is the dependent
    quantity. For a positive ``node_rate`` -- the usual case, the primary's
    heliocentric mean motion -- the leading minus sign forces cos(i) < 0, so
    the resulting orbit is retrograde (i > pi/2), which is characteristic of
    real sun-synchronous orbits (~98 deg for low Earth orbit).

    ``node_rate`` is keyword-only and has no default, because the target rate
    is a property of the body's relationship to its primary, which BodyParams
    does not encode. For an Earth sun-synchronous orbit it is the mean rate of
    2*pi radians per tropical year (~1.991e-7 rad/s).

    Parameters
    ----------
    body : BodyParams
        Central body. Requires a defined, nonzero ``J2``.
    a : float
        Semi-major axis [km]. A design input.
    e : float, optional
        Eccentricity, in [0, 1). Default 0.0 (circular).
    omega : float, optional
        Right ascension of the ascending node (RAAN) [rad]. Default 0.0.
    w : float, optional
        Argument of periapsis [rad]. Default 0.0.
    nu : float, optional
        True anomaly [rad]. Default 0.0.
    node_rate : float, keyword-only
        Target nodal precession rate [rad/s]. For a true sun-synchronous orbit,
        the primary's mean heliocentric rate. No default.

    Returns
    -------
    OrbitalElements
        Keplerian elements for the sun-synchronous orbit, tagged with
        ``mu = body.mu``.

    Raises
    ------
    ValueError
        If ``body.J2`` is None or zero (nodal regression, and thus a
        sun-synchronous orbit, is undefined), if ``e`` is outside [0, 1), or if
        no inclination satisfies the condition at the requested ``a`` and ``e``
        (the required cos(i) falls outside [-1, 1], which happens when the
        semi-major axis is too large for the target node rate).

    Notes
    -----
    As with the Molniya construction, J2 is used only to *derive* the
    inclination; the returned elements are a pure two-body set with no
    perturbation model attached. The sun-synchronous precession is realized
    only when these elements are propagated under a force model that includes
    J2. The condition here uses the leading-order secular J2 term only; a
    high-fidelity design would refine the inclination against the full force
    model.
    """
    if not body.J2:
        raise ValueError(
            f"Sun-synchronous orbit requires nonzero J2, got {body.J2}."
        )
    if not 0.0 <= e < 1.0:
        raise ValueError(f"Eccentricity must be in [0, 1), got {e}.")

    n = np.sqrt(body.mu / a**3)          # mean motion
    p = a * (1.0 - e**2)                 # semi-latus rectum

    cos_i = -(2 / 3) * (node_rate / (body.J2 * n)) * (p / body.radius) ** 2

    if abs(cos_i) > 1.0:
        raise ValueError(
            "No sun-synchronous orbit exists for this body at a="
            f"{a}, e={e}: the required inclination has cos(i)={cos_i:.4f}, "
            "outside [-1, 1]. Try a smaller semi-major axis."
        )

    i = np.arccos(cos_i)

    return OrbitalElements(a=a, e=e, i=i,
                           omega=omega, w=w, nu=nu, mu=body.mu)
