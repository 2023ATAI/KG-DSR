from math import pi
import numpy as np
from scipy.special import gamma as gamma_func
from .metlib import calc_rho,calc_c_p
from .MO_similarity import psi_m_brutsaert,psi_h_brutsaert
import xarray as xr
'''
DESCRIPTION
===========
This module includes functions for calculating the resistances for
heat and momentum trasnport for both One- and Two-Source Energy Balance models.
Additional functions needed in are imported from the following packages

* :doc:`meteo_utils` for the estimation of meteorological variables.
* :doc:`MO_similarity` for the estimation of the Monin-Obukhov length and stability functions.

PACKAGE CONTENTS
================
Resistances
-----------
* :func:`calc_R_A` Aerodynamic resistance.
* :func:`calc_R_S_Choudhury` [Choudhury1988]_ soil resistance.
* :func:`calc_R_S_McNaughton` [McNaughton1995]_ soil resistance.
* :func:`calc_R_S_Kustas` [Kustas1999]_ soil resistance.
* :func:`calc_R_x_Choudhury` [Choudhury1988]_ canopy boundary layer resistance.
* :func:`calc_R_x_McNaughton` [McNaughton1995]_ canopy boundary layer resistance.
* :func:`calc_R_x_Norman` [Norman1995]_ canopy boundary layer resistance.



Estimation of roughness
-----------------------
* :func:`calc_d_0` Zero-plane displacement height.
* :func:`calc_roughness` Roughness for different land cover types.
* :func:`calc_z_0M` Aerodynamic roughness lenght.
* :func:`raupach` Roughness and displacement height factors for discontinuous canopies.

'''
# Landcover classes and values come from IGBP Land Cover Type Classification
WATER = 0
CONIFER_E = 1
BROADLEAVED_E = 2
CONIFER_D = 3
BROADLEAVED_D = 4
FOREST_MIXED = 5
SHRUB_C = 6
SHRUB_O = 7
SAVANNA_WOODY = 8
SAVANNA = 9
GRASS = 10
WETLAND = 11
CROP = 12
URBAN = 13
CROP_MOSAIC = 14
SNOW = 15
BARREN = 16

# Leaf stomata distribution
AMPHISTOMATOUS = 2
HYPOSTOMATOUS = 1
# von Karman's constant
KARMAN = 0.41
# acceleration of gravity (m s-2)
gravity = 9.8
# Universal gas constant (kPa m3 mol-1 K-1)
R_u = 0.0083144

CM_a = 0.01  # Choudhury and Monteith 1988 leaf drag coefficient
KN_b = 0.012  # Value propoesd in Kustas et al 1999
KN_c = 0.0038  # Coefficient from Kustas et al. 2016
KN_C_dash = 90.0  # value proposed in Norman et al. 1995


def calc_Km(h_C):
    '''
    Eddy diffusivity decay constant (Shuttleworth and Gurney,1990)
    or 
    extinction coefficient of the eddy diffusion (Brutsaert, 1982)
    
    calculate eddy diffusivity decay constant based on canopy height
    
    Parameters
    ----------
    h_C : float
        canopy height (m).
    
    Returns
    -------
    Km : float
        Eddy diffusivity decay constant (-).

    References
    ----------
    .. [Zhou2006] Zhou, M. C., Ishidaira, H., Hapuarachchi, H. P., Magome, J., Kiem, A. S., & Takeuchi, K. (2006). 
        Estimating potential evapotranspiration using Shuttleworth–Wallace model and NOAA-AVHRR NDVI data to feed a
        distributed hydrological model over the Mekong River basin. Journal of Hydrology, 327(1-2), 151-173.
    .. [Monteith1973] Monteith, J.L., 1973. Principles of Environmental Physics. Edward Arnold, London, 214 pp.
    .. [Brutsaert1982] Brutsaert, W., 1982. Evaporation into the Atmosphere. D. Reidel,Dordrecht, Holland, 299 pp. (pp106)

    '''
    Km = h_C*0.194+2.306
    Km[h_C <= 1.0] = 2.5
    Km[h_C >= 10.0] = 4.25
    return Km

def calc_Kh(Ustar,Zc,d0):
    '''
    the eddy diffusion coefficient at the top of the canopy (Shuttleworth and Gurney,1990)

    eddy diffusion coefficient at the top of the canopy     
    Parameters
    ----------
    h_C : float
        canopy height (m).
    
    Returns
    -------
    Km : float
        Eddy diffusivity decay constant (-).

    References
    ----------
    .. [Zhou2006] Zhou, M. C., Ishidaira, H., Hapuarachchi, H. P., Magome, J., Kiem, A. S., & Takeuchi, K. (2006). 
        Estimating potential evapotranspiration using Shuttleworth–Wallace model and NOAA-AVHRR NDVI data to feed a
        distributed hydrological model over the Mekong River basin. Journal of Hydrology, 327(1-2), 151-173.
    .. [Monteith1973] Monteith, J.L., 1973. Principles of Environmental Physics. Edward Arnold, London, 214 pp.
    .. [Brutsaert1982] Brutsaert, W., 1982. Evaporation into the Atmosphere. D. Reidel,Dordrecht, Holland, 299 pp. (pp106)

    '''
    Kh=0.4 * Ustar * (Zc-d0)
    return np.asarray(Kh)

def calc_d_0(h_C):
    ''' Zero-plane displacement height

    Calculates the zero-plane displacement height based on a
    fixed ratio of canopy height.

    Parameters
    ----------
    h_C : float
        canopy height (m).

    Returns
    -------
    d_0 : float
        zero-plane displacement height (m).'''

    d_0 = h_C * 0.666

    return np.asarray(d_0)

def calc_z_0m_wange2014(h_C):
    ''' the roughness length govering momentum, heat and vapor  transfer above vegetation canopy (m)

    Calculates roughness length based on a fixed ratio of canopy height.

    Parameters
    ----------
    h_C : float
        canopy height (m).

    Returns
    -------
    Z0mv : float
        the roughness length govering momentum transfer above vegetation canopy (m)
        
        '''

    Z0mv = h_C * 0.123  # Z0mv is the roughness length govering momentum transfer above vegetation canopy (m)
    Z0hv = 0.1*Z0mv     # Z0hv is the roughness length govering heat and vapor transfer above vegetation canopy (m)

    return np.asarray(Z0mv),np.asarray(Z0hv)

def calc_roughness_Schaudt2000(LAI, h_C, w_C=1, landcover=CROP, f_c=None):
    ''' Surface roughness and zero displacement height for different vegetated surfaces.

    Calculates the roughness using different approaches depending we are dealing with
    crops or grasses (fixed ratio of canopy height) or shrubs and forests,depending of LAI
    and canopy shape, after [Schaudt2000]_

    Parameters
    ----------
    LAI : float
        Leaf (Plant) Area Index.
    h_C : float
        Canopy height (m)
    w_C : float, optional
        Canopy height to width ratio.
    landcover : int, optional
        landcover type, use 11 for crops, 2 for grass, 5 for shrubs,
        4 for conifer forests and 3 for broadleaved forests.
    f_c :    horizontal area index

    Returns
    -------
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    d : float
        Zero-plane displacement height (m).

    References
    ----------
    .. [Schaudt2000] K.J Schaudt, R.E Dickinson, An approach to deriving roughness length
        and zero-plane displacement height from satellite data, prototyped with BOREAS data,
        Agricultural and Forest Meteorology, Volume 104, Issue 2, 8 August 2000, Pages 143-155,
        http://dx.doi.org/10.1016/S0168-1923(00)00153-2.
    '''

    # Convert input scalars to numpy arrays
    LAI, h_C, w_C, landcover = map(np.asarray, (LAI, h_C, w_C, landcover))

    # Initialize fractional cover and horizontal area index
    lambda_ = np.zeros(LAI.shape)
    if f_c is None:
        f_c = np.zeros(LAI.shape)

        # Needleleaf canopies
        mask = np.logical_or(landcover == 1, landcover == 3)
        f_c[mask] = 1. - np.exp(-0.5 * LAI[mask])

        # Broadleaved canopies
        mask = np.logical_or.reduce((landcover == 2, landcover == 4,
                                     landcover == 5, landcover == 8))
        f_c[mask] = 1. - np.exp(-LAI[mask])

        # Shrublands
        mask = np.logical_or(landcover == 6, landcover == 7)
        f_c[mask] = 1. - np.exp(-0.5 * LAI[mask])

    # Needleleaf canopies
    mask = np.logical_or(landcover == 1, landcover == 3)
    lambda_[mask] = (2. / pi) * f_c[mask] * w_C[mask]

    # Broadleaved canopies
    mask = np.logical_or.reduce((landcover == 2, landcover == 4,
                                 landcover == 5, landcover == 8))
    lambda_[mask] = f_c[mask] * w_C[mask]

    # Shrublands
    mask = np.logical_or(landcover == 6, landcover == 7)
    lambda_[mask] = f_c[mask] * w_C[mask]
    
    del w_C, f_c
    # Calculation of the Raupach (1994) formulae
    z0M_factor, d_factor = raupach(lambda_)

    del lambda_

    # Calculation of correction factors from  Lindroth
    fz = np.asarray(0.3299 * LAI**1.5 + 2.1713)
    fd = np.asarray(1. - 0.3991 * np.exp(-0.1779 * LAI))

    # LAI <= 0
    fz[LAI <= 0] = 1.0
    fd[LAI <= 0] = 1.0

    # LAI >= 0.8775:
    fz[LAI >= 0.8775] = 1.6771 * np.exp(-0.1717 * LAI[LAI >= 0.8775]) + 1.
    fd[LAI >= 0.8775] = 1. - 0.3991 * np.exp(-0.1779 * LAI[LAI >= 0.8775])

    # Application of the correction factors to roughness and displacement
    # height
    z0M_factor = np.asarray(z0M_factor * fz)
    d_factor = np.asarray(d_factor * fd)

    del fz, fd
    
    # For crops and grass we use a fixed ratio of canopy height
    mask = np.logical_or.reduce((landcover == 12, landcover == 10,
                                 landcover == 9, landcover == 14))
    z0M_factor[mask] = 1. / 8.
    d_factor[mask] = 0.65

    # Calculation of rouhgness length
    z_0M = np.asarray(z0M_factor * h_C)

    # Calculation of zero plane displacement height
    d = np.asarray(d_factor * h_C)

    # For barren surfaces (bare soil, water, etc.)
    mask = np.logical_or.reduce((landcover == 0, landcover == 13,
                                 landcover == 15, landcover == 16))
    z_0M[mask] = 0.01
    d[mask] = 0

    return z_0M, d

def calc_z_0H(z_0M, kB=0):
    """Estimate the aerodynamic routhness length for heat trasport.

    Parameters
    ----------
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    kB : float
        kB parameter, default = 0.

    Returns
    -------
    z_0H : float
        aerodynamic roughness length for momentum transport (m).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    """

    z_0H = z_0M / np.exp(kB)
    return z_0H

def calc_z_0m(h_C):
    """ Aerodynamic roughness lenght.

    Estimates the aerodynamic roughness length for momentum trasport
    as a ratio of canopy height.

    Parameters
    ----------
    h_C : float
        Canopy height (m).

    Returns
    -------
    z_0M : float
        aerodynamic roughness length for momentum transport (m).
    """

    z_0M = h_C * 0.125
    return z_0M

def calc_z_0m_Shuttleworth1990(LAI,Zc,z0_soil,cd,d0):
    '''
        References
    ----------
    .. [Zhou2006] Zhou, M. C., Ishidaira, H., Hapuarachchi, H. P., Magome, J., Kiem, A. S., & Takeuchi, K. (2006). 
        Estimating potential evapotranspiration using Shuttleworth–Wallace model and NOAA-AVHRR NDVI data to feed a
        distributed hydrological model over the Mekong River basin. Journal of Hydrology, 327(1-2), 151-173.

    .. [Choudhury1988] Choudhury, B.J., Monteith, J.L., 1988. A four-layer model for the
        heat budget of homogeneous land surfaces. Quart. J. Royal
        Meteorol. Soc. 114, 373–398.
    '''
    X=cd*LAI
    z0m=z0_soil+0.3*Zc*(X**0.5)
    z0m=np.where(X>0.2,z0m,0.3*(Zc-d0))
    return z0m

def calc_cd_Shuttleworth1990(h_C):
    '''
        References
    ----------
    .. [Zhou2006] Zhou, M. C., Ishidaira, H., Hapuarachchi, H. P., Magome, J., Kiem, A. S., & Takeuchi, K. (2006). 
        Estimating potential evapotranspiration using Shuttleworth–Wallace model and NOAA-AVHRR NDVI data to feed a
        distributed hydrological model over the Mekong River basin. Journal of Hydrology, 327(1-2), 151-173.
    '''
    z0c=0.139*h_C - 0.009*h_C*h_C
    z0c=np.where(h_C<=1.0,0.13*h_C,z0c)
    z0c=np.where(h_C>=10.0,0.05*h_C,z0c)
    cd =(-1.0+np.exp(0.909-3.03*z0c/h_C))**4.0/4.0
    cd=np.where(h_C<0.01,1.4/1000.,cd)
    return cd,z0c

def calc_ustar_Shuttleworth1990(wind,Za,d0,z0m):
    ustar=KARMAN*wind/np.log((Za-d0)/z0m)
    return ustar

def calc_d_0_Shuttleworth1990(h_C,z0c,cd,LAI):
    ''' 
    Zero-plane displacement height
    '''
    #cd,z0c=calc_cd_Shuttleworth1990(h_C)
    #print(cd)
    d_0=h_C-z0c/0.3
    d_0=np.where(LAI<4.0,1.1*h_C*np.log(1.0+(cd*LAI)**0.25),d_0)
    return d_0

def raupach(lambda_):
    """Roughness and displacement height factors for discontinuous canopies

    Estimated based on the frontal canopy leaf area, based on Raupack 1994 model,
    after [Schaudt2000]_

    Parameters
    ----------
    lambda_ : float
        roughness desnsity or frontal area index.

    Returns
    -------
    z0M_factor : float
        height ratio of roughness length for momentum transport
    d_factor : float
        height ratio of zero-plane displacement height

    References
    ----------
    .. [Schaudt2000] K.J Schaudt, R.E Dickinson, An approach to deriving roughness length
        and zero-plane displacement height from satellite data, prototyped with BOREAS data,
        Agricultural and Forest Meteorology, Volume 104, Issue 2, 8 August 2000, Pages 143-155,
        http://dx.doi.org/10.1016/S0168-1923(00)00153-2.
    """

    # Convert input scalar to numpy array
    lambda_ = np.asarray(lambda_)
    z0M_factor = np.zeros(lambda_.shape)
    d_factor = np.asarray(np.zeros(lambda_.shape) + 0.65)

    # Calculation of the Raupach (1994) formulae
    # if lambda_ > 0.152:
    i = lambda_ > 0.152
    z0M_factor[i] = ((0.0537 / (lambda_[i]**0.510))
                     * (1. - np.exp(-10.9 * lambda_[i]**0.874)) + 0.00368)
    # else:
    z0M_factor[~i] = 5.86 * np.exp(-10.9 * lambda_[~i]**1.12) * lambda_[~i]**1.33 + 0.000860
    # if lambda_ > 0:
    i = lambda_ > 0
    d_factor[i] = 1. - \
        (1. - np.exp(-np.sqrt(15.0 * lambda_[i]))) / np.sqrt(15.0 * lambda_[i])

    return np.asarray(z0M_factor), np.asarray(d_factor)

#---------------------------------------------------------------------------

def calc_R_A_Norman1995(z_T, ustar, L, d_0, z_0H):
    ''' Estimates the aerodynamic resistance to heat transport based on the
    MO similarity theory.

    Parameters
    ----------
    z_T : float
        air temperature measurement height (m).
    ustar : float
        friction velocity (m s-1).
    L : float
        Monin Obukhov Length for stability
    d_0 : float
        zero-plane displacement height (m).
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    z_0H : float
        aerodynamic roughness length for heat trasport (m).

    Returns
    -------
    R_A : float
        aerodyamic resistance to heat transport in the surface layer (s m-1).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    '''

    # Convert input scalars to numpy arrays
    #z_T, ustar, L, d_0, z_0H = map(np.asarray, (z_T, ustar, L, d_0, z_0H))
    #R_A_log = np.asarray(np.log((z_T - d_0) / z_0H))
    
    R_A_log = (np.log((z_T - d_0) / z_0H))

    # if L -> infinity, z./L-> 0 and there is neutral atmospheric stability
    # other atmospheric conditions
    L[L == 0] = 1e-36
    Psi_H = psi_h_brutsaert((z_T - d_0) / L)
    Psi_H0 = psi_h_brutsaert(z_0H / L)
    del L, z_0H, z_T

    # i = np.logical_and(z_star>0, z_T<=z_star)
    # Psi_H_star[i] = MO.calc_Psi_H_star(z_T[i], L[i], d_0[i], z_0H[i], z_star[i])

    #i = ustar != 0
    #R_A = np.asarray(np.ones(ustar.shape) * float('inf'))
    R_A = (R_A_log - Psi_H + Psi_H0) / (ustar * KARMAN)

    return R_A #np.asarray(R_A)

def calc_R_A_wang2014(Zu_m,Zt_m,U,L,d0,Z0mv):
    """ 
    The aerodynamic resistance Parameters
    ----------
    Zh : float
        measurement height of flux (m)
    U    : float
        wind speed (m/s)
    ObukhovLength   : float
        Obukhov Length (m)
    Zt_m : float
        the height of temperature and humidity measurement (m)
    Zu_m : float
        measurement height of wind speed measurement (m)
    """
    Z0hv=0.1*Z0mv 
    ZL   =(Zu_m - d0) / L
    ZL   = np.where(ZL > -10.0, ZL, -10.0)
    ZL   = np.where(ZL < 10.0, ZL, 10.0)
    stab_m=psi_m_brutsaert(ZL)
    stab_h=psi_h_brutsaert(ZL)
    rav  = (np.log( (Zu_m - d0) / Z0mv) - stab_m) * (np.log(((Zt_m)-d0) / Z0hv)-stab_h) / ((KARMAN**2)*U)
    return rav

#---------------------------------------------------------------------------
def calc_R_S_Choudhury1988(u_star, h_C, z_0M, d_0, zm, z0_soil=0.01, alpha_k=2.0):
    ''' Aerodynamic resistance at the  soil boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    K-Theory model of [Choudhury1988]_.

    Parameters
    ----------
    u_star : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    d_0 : float
        zero-plane displacement height (m).
    zm : float
        height on measurement of wind speed (m).
    z0_soil : float, optional
        roughness length of the soil layer, use z0_soil=0.01.
    alpha_k : float, optional
        Heat diffusion coefficient, default=2.

    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).

    References
    ----------
    .. [Choudhury1988] Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model
        for the heat budget of homogeneous land surfaces.
        Royal Meteorological Society, Quarterly Journal, 114(480), 373-398.
        http://dx/doi.org/10.1002/qj.49711448006.
    '''

    # Soil resistance eqs. 24 & 25 [Choudhury1988]_
    K_h = KARMAN * u_star * (h_C - d_0)
    del u_star
    R_S = ((h_C * np.exp(alpha_k) / (alpha_k * K_h))
           * (np.exp(-alpha_k * z0_soil / h_C) - np.exp(-alpha_k * (d_0 + z_0M) / h_C)))

    return np.asarray(R_S)

def calc_R_S_McNaughton1995(u_friction):
    ''' Aerodynamic resistance at the  soil boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    Lagrangian model of [McNaughton1995]_.

    Parameters
    ----------
    u_friction : float
        friction velocity (m s-1).

    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1)

    References
    ----------
    .. [McNaughton1995] McNaughton, K. G., & Van den Hurk, B. J. J. M. (1995).
        A 'Lagrangian' revision of the resistors in the two-layer model for calculating
        the energy budget of a plant canopy. Boundary-Layer Meteorology, 74(3), 261-288.
        http://dx/doi.org/10.1007/BF00712121.

    '''

    R_S = 10.0 / u_friction
    return np.asarray(R_S)

def calc_R_S_Kustas1999(u_S, deltaT, params=None):
    """ Aerodynamic resistance at the  soil boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    original equations in TSEB [Kustas1999]_.

    Parameters
    ----------
    u_S : float
        wind speed at the soil boundary layer (m s-1).
    deltaT : float
        Surface to air temperature gradient (K).

    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).

    References
    ----------
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    """

    if params is None:
        params = {}

    # Set model parameters
    if "KN_b" in params:
        b = params["KN_b"]
    else:
        b = KN_b
    if "KN_c" in params:
        c = params['KN_c']
    else:
        c = KN_c

    # Convert input scalars to numpy arrays
    u_S, deltaT = map(np.asarray, (u_S, deltaT))

    deltaT = np.asarray(np.maximum(deltaT, 0.0))
    R_S = 1.0 / (c * deltaT**(1.0 / 3.0) + b * u_S)
    return np.asarray(R_S)

def calc_R_S_Li2019(u, h_c, zm, rho, c_p, z0_soil=0.01, f_cover=0, w_C=1,
                      c_d=0.2, a_r=3, a_s=5, k=0.1):
    """ Aerodynamic resistance at the  soil boundary layer.
    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    soil resistance formulation adapted by [Li2019]_.
    Parameters
    ----------
    % U           | wind velocity                     [m s-1]
    % h_c         | (cylindrical) vegettaion height   [m]      =0 for bare soil
    zm : float
        height on measurement of wind speed (m).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    rho : float
        Density of air (km m-3).
    h_C : float
        canopy height (m).
    z_0M: float
        aerodynamic roughness length for momentum trasport (m).
    d_0 : float
        zero-plane displacement height (m).

    z0_soil : float, optional
        roughness length of the soil layer, use z0_soil=0.01.
    alpha_k : float, optional
        Heat diffusion coefficient, default=2.

    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).

    References
    ----------
    ..[Li2019] Li, Yan, et al.
        "Evaluating Soil Resistance Formulations in Thermal?Based
        Two?Source Energy Balance (TSEB) Model:
        Implications for Heterogeneous Semiarid and Arid Regions."
        Water Resources Research 55.2 (2019): 1059-1078.
        https://doi.org/10.1029/2018WR022981.
    ..[Haghighi2015] Haghighi, Erfan, and Dani Or.
        "Interactions of bluff-body obstacles with turbulent airflows affecting
        evaporative fluxes from porous surfaces."
        Journal of Hydrology 530 (2015): 103-116.
        https://doi.org/10.1016/j.jhydrol.2015.09.048
    ..[Haghighi2013] Haghighi, E., and Dani Or.
        "Evaporation from porous surfaces into turbulent airflows:
        Coupling eddy characteristics with pore scale vapor diffusion."
        Water Resources Research 49.12 (2013): 8432-8442.
        https://doi.org/10.1002/2012WR013324.

    % -------------------------------------------------------------------------
    %  Inputs   |              Description
    % -------------------------------------------------------------------------
    % ps        | mean particle size of soil        [m]
    % n         | soil pore size distribution index [-]
    % phi       | porosity                          [-]
    % theta     | soil water content                [m3 m-3]
    % theta_res | residual water content            [m3 m-3]
    % z_w       | measurement height                [m]
    % eta       | vegetation cover fraction         [-]      =0 for bare soil
    % d         | (cylindrical) vegetation diameter [m]      =0 for bare soil
    % -------------------------------------------------------------------------
    """

    def calc_prod_alpha(alpha, n):
        out = 1.0
        for i in range(n):
            out = out * (2 * (alpha - i) + 1)
        return out

    def calc_prod_alpha_array(alpha_array):
        out_array = np.ones(alpha_array.shape)
        n_array = np.ceil(alpha_array).astype(np.int)
        ns = np.unique(n_array)
        for n in ns:
            index = n_array == n
            out_array[index] = calc_prod_alpha(alpha_array[index], n)
        return out_array



    # Define constanst
    nu = 15.11e-6   # [m2 s-1]    kinmeatic visocosity of air
    k_a = 0.024    # [W m-1 K-1] thermal conductivity of air
    c_2 = 2.2
    c_3 = 112.
    g_alpha = 21.7
    u, h_c, zm, z0_soil, f_cover, w_C = map(np.asarray, (u, h_c, zm, z0_soil, f_cover, w_C))

    width = h_c * w_C
    a_veg = (np.pi / 4) * width**2
    lambda_ = width * h_c * f_cover / a_veg

    h_c[f_cover == 0] = 0
    lambda_[f_cover == 0] = 0

    z_0sc = z0_soil * (1 + f_cover * ((zm - h_c) / zm - 1))
    f_r = np.exp(-a_r * lambda_ / (1. - f_cover)**k)
    f_s = np.exp(-a_s * lambda_ / (1. - f_cover)**k)
    c_sgc = KARMAN**2 * (np.log((zm - h_c) / z_0sc))**-2
    c_sg = KARMAN**2 * (np.log(zm / z0_soil))**-2
    f_v = 1. + f_cover * (c_sgc / c_sg - 1.)
    beta = (c_d / KARMAN ** 2) * ((np.log(h_c / z0_soil) - 1)**2 + 1)
    c_rg = beta * c_sg

    alpha_mean = (0.3 / np.sqrt(f_r * lambda_ * (1. - f_cover) * c_rg
                               + (f_s * (1. - f_cover) + f_v * f_cover) * c_sg)) - 1
    u_star = (u * np.sqrt(f_r * lambda_ * (1. - f_cover) * c_rg
              + (f_s * (1 - f_cover) + f_v * f_cover) * c_sg))

    prod_alpha = calc_prod_alpha_array(alpha_mean)
    g_alpha = c_2 * np.sqrt(c_3 * np.pi) / (gamma_func(alpha_mean + 1)
                                            * 2**(alpha_mean + 1)
                                            * np.sqrt(alpha_mean + 1))
    g_alpha[alpha_mean > 0] = g_alpha[alpha_mean > 0] * prod_alpha[alpha_mean > 0]
    delta = g_alpha * nu / u_star
    r_s = rho * c_p * delta / k_a

    return np.asarray(r_s)

def calc_r_ss_Haghighi2015(k_sat,u, h_c, zm, rho, c_p, z0_soil=0.01, f_cover=1, w_c=1,
                      theta=0.4, theta_res=0.1, phi=2.0, ps=0.001, n=0.5):
    """ Aerodynamic resistance at the  soil boundary layer.

    Estimates the aerodynamic resistance at the  soil boundary layer based on the
    soil resistance formulation adapted by [Li2019]_.

    % -------------------------------------------------------------------------
    %  Inputs   |              Description
    % -------------------------------------------------------------------------
    % ps        | mean particle size of soil        [m]
    % n         | soil pore size distribution index [-]
    % phi       | porosity                          [-]
    % theta     | soil water content                [m3 m-3]
    % theta_res | residual water content            [m3 m-3]
    % zm       | measurement height                [m]
    % U         | wind velocity                     [m s-1]
    % eta       | vegetation cover fraction         [-]      =0 for bare soil
    % h         | (cylindrical) vegettaion height   [m]      =0 for bare soil
    % d         | (cylindrical) vegetation diameter [m]      =0 for bare soil
    % -------------------------------------------------------------------------
    w_C : float, optional
        Canopy width to height ratio.
    Parameters
    ----------
    u_star : float
        friction velocity (m s-1).
    h_C : float
        canopy height (m).
    z_0M : float
        aerodynamic roughness length for momentum trasport (m).
    d_0 : float
        zero-plane displacement height (m).
    zm : float
        height on measurement of wind speed (m).
    z0_soil : float, optional
        roughness length of the soil layer, use z0_soil=0.01.
    alpha_k : float, optional
        Heat diffusion coefficient, default=2.

    Returns
    -------
    R_S : float
        Aerodynamic resistance at the  soil boundary layer (s m-1).

    References
    ----------
    ..[Li2019] Li, Yan, et al.
        "Evaluating Soil Resistance Formulations in Thermal?Based
        Two?Source Energy Balance (TSEB) Model:
        Implications for Heterogeneous Semiarid and Arid Regions."
        Water Resources Research 55.2 (2019): 1059-1078.
        https://doi.org/10.1029/2018WR022981.
    ..[Haghighi2015] Haghighi, Erfan, and Dani Or.
        "Interactions of bluff-body obstacles with turbulent airflows affecting
        evaporative fluxes from porous surfaces."
        Journal of Hydrology 530 (2015): 103-116.
        https://doi.org/10.1016/j.jhydrol.2015.09.048
    ..[Haghighi2013] Haghighi, E., and Dani Or.
        "Evaporation from porous surfaces into turbulent airflows:
        Coupling eddy characteristics with pore scale vapor diffusion."
        Water Resources Research 49.12 (2013): 8432-8442.
        https://doi.org/10.1002/2012WR013324.

    ..[Haghighi2013] Haghighi, E., Shahraeeni, E., Lehmann, P., and Or, D. (2013),
        Evaporation rates across a convective air boundary layer are 
        dominated by diffusion, Water Resour. Res., 49, 1602– 1610, doi:10.1002/wrcr.20166.


    """

    # Define constanst
    diff_w = 0.282e-4   # [m2 s-1]    water vapor diffusion coefficient in air
    nu = 15.11e-6   # [m2 s-1]    kinmeatic visocosity of air
    lambda_e = 2450e3   # [J/kg]      Latent heat of vaporization

    # ..[Haghighi2015]
    a_r = 3.
    a_s = 5.
    k = 0.1
    gamma = 150.
    f_alpha = 22.  # [Haghighi2013]_

    u, h_c, zm, z0_soil, f_cover, w_c, theta, theta_res, phi, ps, n = map(
        np.asarray, (u, h_c, zm, z0_soil, f_cover, w_c, theta, theta_res, phi, ps, n))

    f_theta = ((1. / np.sqrt(np.pi * (theta - theta_res)))
               * (np.sqrt(np.pi / (4 * (theta - theta_res))) - 1))

    theta = (theta - theta_res) / (phi - theta_res)
    del theta_res, phi

    #k_sat = (0.0077 * n**7.35) / (24. * 3600.)  # [m s-1]

    m = 1. - 1. / n
    k_eff = (4 * k_sat * np.sqrt(theta) * (1 - (1 - theta**(1 / m))**m)**2)  # [Haghighi2013]_
    del k_sat, m, n

    width = h_c * w_c
    a_veg = (np.pi / 4) * width**2
    lambda_ = width * h_c * f_cover / a_veg

    h_c[f_cover == 0] = 0
    lambda_[f_cover == 0] = 0

    z_0sc = z0_soil * (1 + f_cover * ((zm - h_c) / zm - 1))
    f_r = np.exp(-a_r * lambda_ / (1. - f_cover)**k)
    f_s = np.exp(-a_s * lambda_ / (1. - f_cover)**k)
    c_sgc = KARMAN**2 * (np.log((zm - h_c) / z_0sc))**(-2)
    c_sg = KARMAN**2 * (np.log(zm / z0_soil))**(-2)
    f_c = 1. + f_cover * (c_sgc / c_sg - 1.)
    c_rg = gamma * c_sg
    u_star = (u * np.sqrt(f_r * lambda_ * (1 - f_cover) * c_rg
              + (f_s * (1 - f_cover) + f_c * f_cover) * c_sg))
    delta = f_alpha * nu / u_star
    r_ss = (rho * c_p * ((delta + (ps / 3) * f_theta) / diff_w + 1.73e-5 / k_eff)
              / lambda_e)     # [Haghighi2013]_

    return np.asarray(r_ss*1000.0) #mm/s

def calc_r_ss_sellers1992(theta_surf,rss_v):
    rss     = np.exp(8.206-4.225*theta_surf)*(rss_v) #% Soil resistance
    return rss

#---------------------------------------------------------------------------
def calc_R_x_Choudhury1988(u_C, F, leaf_width, alpha_prime=3.0):
    ''' Estimates aerodynamic resistance at the canopy boundary layer.

    Estimates the aerodynamic resistance at the canopy boundary layer based on the
    K-Theory model of [Choudhury1988]_.

    Parameters
    ----------
    u_C : float
        wind speed at the canopy interface (m s-1).
    F : float
        local Leaf Area Index.
    leaf_width : float
        efective leaf width size (m).
    alpha_prime : float, optional
        Wind exctinction coefficient, default=3.

    Returns
    -------
    R_x : float
        Aerodynamic resistance at the canopy boundary layer (s m-1).

    References
    ----------
    .. [Choudhury1988] Choudhury, B. J., & Monteith, J. L. (1988). A four-layer model
        for the heat budget of homogeneous land surfaces.
        Royal Meteorological Society, Quarterly Journal, 114(480), 373-398.
        http://dx/doi.org/10.1002/qj.49711448006.
    '''

    # Eqs. 29 & 30 [Choudhury1988]_
    R_x = (1.0 / (F * (2.0 * CM_a / alpha_prime)
           * np.sqrt(u_C / leaf_width) * (1.0 - np.exp(-alpha_prime / 2.0))))
    # R_x=(alpha_u*(sqrt(leaf_width/U_C)))/(2.0*alpha_0*LAI*(1.-exp(-alpha_u/2.0)))
    return np.asarray(R_x)

def calc_R_x_McNaughton1995(F, leaf_width, u_star):
    ''' Estimates aerodynamic resistance at the canopy boundary layer.

    Estimates the aerodynamic resistance at the canopy boundary layer based on the
    Lagrangian model of [McNaughton1995]_.

    Parameters
    ----------
    F : float
        local Leaf Area Index.
    leaf_width : float
        efective leaf width size (m).
    u_d_zm : float
        wind speed at the height of momomentum source-sink.

    Returns
    -------
    R_x : float
        Aerodynamic resistance at the canopy boundary layer (s m-1).

    References
    ----------
    .. [McNaughton1995] McNaughton, K. G., & Van den Hurk, B. J. J. M. (1995).
        A 'Lagrangian' revision of the resistors in the two-layer model for calculating
        the energy budget of a plant canopy. Boundary-Layer Meteorology, 74(3), 261-288.
        http://dx/doi.org/10.1007/BF00712121.
    '''

    C_dash = 130.0
    C_dash_F = C_dash / F
    # Eq. 30 in [McNaugthon1995]
    R_x = C_dash_F * (leaf_width * u_star)**0.5 + 0.36 / u_star
    return np.asarray(R_x)

def calc_R_x_Norman1995(LAI, leaf_width, u_d_zm, params=None):
    """ Estimates aerodynamic resistance at the canopy boundary layer.

    Estimates the aerodynamic resistance at the canopy boundary layer based on the
    original equations in TSEB [Norman1995]_.

    Parameters
    ----------
    F : float
        local Leaf Area Index.
    leaf_width : float
        efective leaf width size (m).
    u_d_zm : float
        wind speed at the height of momomentum source-sink. .

    Returns
    -------
    R_x : float
        Aerodynamic resistance at the canopy boundary layer (s m-1).

    References
    ----------
    .. [Norman1995] J.M. Norman, W.P. Kustas, K.S. Humes, Source approach for estimating
        soil and vegetation energy fluxes in observations of directional radiometric
        surface temperature, Agricultural and Forest Meteorology, Volume 77, Issues 3-4,
        Pages 263-293, http://dx.doi.org/10.1016/0168-1923(95)02265-Y.
    """

    if params is None:
        params = {}

    # Set model parameters
    if "KN_C_dash" in params:
        C_dash = params["KN_C_dash"]
    else:
        C_dash = KN_C_dash

    R_x = (C_dash / LAI) * (leaf_width / u_d_zm)**0.5
    return np.asarray(R_x)

#---------------------------------------------------------------------------

def calc_rac_Shuttleworth1990(LAI,dl,Zc,z_U,Wind,Km,TimVars=None):
   # Km=calc_Km(Zc)
   # LAI,Km,dl,Zc,z_U,Wind = map(np.asarray, (LAI,Km,dl,Zc,z_U,Wind))
    Uh                          =    Wind / (1.0 + np.log(z_U - Zc +1.0)) 
    Uh                          =  np.where(Uh >0.1, Uh, 1.0)
    rb                          =    (100./ Km) * (dl/Uh) *0.5 / (1.0 - np.exp(-Km / 2.0))
    rac                         =    rb/(2.0*LAI)
    rac                         =  np.where(rac >1.0, rac, 1.0)
    return rac

def calc_raa_Shuttleworth1990(Ustar,Zc,z_U,z0,d0,Km,TimVars=None):
    
    Kh                                  =    0.4 * Ustar * (Zc-d0)
    raa                                 =   (1. / (0.4*Ustar))*np.log((z_U-d0)/(Zc-d0)) + (Zc / (Km *  Kh)) * (np.exp( Km * (1.0 - (d0 + z0 ) / Zc))-1.0)
    raa                                 =  np.where(raa >1.0, raa, 1.0)

    if TimVars is not None:
        TimVars['Kh'].values            =   Kh
        return np.asarray(raa),TimVars
    else:
        return raa
    
def calc_ras_Shuttleworth1990(Ustar,Zc,z0_soil,d0,Km,z0,TimVars=None):
    #Zc,Km,z0,d0,z0_soil,Ustar   =    map(np.asarray, (Zc,Km,z0,d0,z0_soil,Ustar))
    Kh                          =    0.4 * Ustar * (Zc-d0)
    ras                         =    (Zc *(np.exp(Km))/(Km * Kh)) * (np.exp((-Km *z0_soil) / Zc)-np.exp(-Km * ((d0+z0) / Zc)))
    ras                         =    np.where(ras >1.0, ras, 1.0)

    if TimVars is not None:
        TimVars['Kh'].values    =    Kh
        TimVars['ras'].values   =    np.asarray(ras)
        return np.asarray(ras), TimVars
    else:
        return ras

#---------------------------------------------------------------------------

# def calc_r_r(p, ea, t_k):
#     """ Calculates the resistance to radiative transfer
#     Parameters
#     ----------
#     p : float or array
#         Surface atmospheric pressure (mb)
#     ea : float or array
#         Vapour pressure (mb).
#     t_k : float or array
#         surface temperature (K)
#
#     Returns
#     -------
#     r_r : float or array
#         Resistance to radiative transfer (s m-1)
#
#     References
#     ----------
#     .. [Monteith2008] Monteith, JL, Unsworth MH, Principles of Environmental
#     Physics, 2008. ISBN 978-0-12-505103-5
#
#     """
#
#     rho = calc_rho(p, ea, t_k)
#     cp =  calc_c_p(p, ea)
#     r_r = rho * cp / (4. * rad.SB * t_k**3)  # Eq. 4.7 of [Monteith2008]_
#     return r_r




