import numpy as np

# ==============================================================================
# List of constants used in Meteorological computations
# ==============================================================================
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8
# heat capacity of dry air at constant pressure (J kg-1 K-1)
c_pd = 1003.5
# heat capacity of water vapour at constant pressure (J kg-1 K-1)
c_pv = 1865
# ratio of the molecular weight of water vapor to dry air
epsilon = 0.622
# Psicrometric Constant kPa K-1
psicr = 0.0658
# gas constant for dry air, J/(kg*degK)
R_d = 287.04
# acceleration of gravity (m s-2)
gravity = 9.8
# von Karman's constant
karman = 0.41

def calc_vpd(Rh_Avg,Ta):
    #Rh_Avg, Ta     =  map(np.asarray, (Rh_Avg,Ta))
    e_Avg          =  0.6108*np.exp(17.27*Ta/(Ta + 237.3))*Rh_Avg  #; %Kpa
    esat_air       =  0.6108*np.exp(17.27*Ta/(Ta + 237.3)) #Kpa
    VPD            =  esat_air-e_Avg
    return VPD

def calc_c_p(p, ea):
    ''' Calculates the heat capacity of air at constant pressure.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).

    Returns
    -------
    c_p : heat capacity of (moist) air at constant pressure (J kg-1 K-1).

    References
    ----------
    based on equation (6.1) from Maarten Ambaum (2010):
    Thermal Physics of the Atmosphere (pp 109).'''

    # first calculate specific humidity, rearanged eq (5.22) from Maarten
    # Ambaum (2010), (pp 100)
    q = epsilon * ea / (p + (epsilon - 1.0) * ea)
    # then the heat capacity of (moist) air
    c_p = (1.0 - q) * c_pd + q * c_pv
    return c_p  #np.asarray(c_p)

def calc_lambda(T_A_K):
    '''Calculates the latent heat of vaporization.

    Parameters
    ----------
    T_A_K : float
        Air temperature (Kelvin).

    Returns
    -------
    Lambda : float
        Latent heat of vaporisation (J kg-1).

    References
    ----------
    based on Eq. 3-1 Allen FAO98 '''

    Lambda = 1e6 * (2.501 - (2.361e-3 * (T_A_K - 273.15)))
    return np.asarray(Lambda)

def calc_pressure(z):
    ''' Calculates the barometric pressure above sea level.

    Parameters
    ----------
    z: float
        height above sea level (m).

    Returns
    -------
    p: float
        air pressure (mb).'''

    p = 1013.25 * (1.0 - 2.225577e-5 * z)**5.25588
    return np.asarray(p)

def calc_psicr(c_p, p, Lambda):
    ''' Calculates the psicrometric constant.

    Parameters
    ----------
    c_p : float
        heat capacity of (moist) air at constant pressure (J kg-1 K-1).
    p : float
        atmopheric pressure (mb).
    Lambda : float
        latent heat of vaporzation (J kg-1).

    Returns
    -------
    psicr : float
        Psicrometric constant (mb C-1).'''

    psicr = c_p * p / (epsilon * Lambda)
    return np.asarray(psicr)

def calc_rho(p, ea, T_A_K):
    '''Calculates the density of air.

    Parameters
    ----------
    p : float
        total air pressure (dry air + water vapour) (mb).
    ea : float
        water vapor pressure at reference height above canopy (mb).
    T_A_K : float
        air temperature at reference height (Kelvin).

    Returns
    -------
    rho : float
        density of air (kg m-3).

    References
    ----------
    based on equation (2.6) from Brutsaert (2005): Hydrology - An Introduction (pp 25).'''

    # p is multiplied by 100 to convert from mb to Pascals
    rho = ((p * 100.0) / (R_d * T_A_K)) * (1.0 - (1.0 - epsilon) * ea / p)
    return np.asarray(rho)

def calc_rho_w(T_K):
    """
    density of air-free water ata pressure of 101.325kPa
    :param T_K:
    :return:
    density of water (kg m-3)
    """
    t = T_K - 273.15  # Temperature in Celsius
    rho_w = (999.83952 + 16.945176 * t - 7.9870401e-3 * t**2
             - 46.170461e-6 * t**3 + 105.56302e-9 * t**4
             - 280.54253e-12 * t**5) / (1 + 16.897850e-3 * t)

    return rho_w

def calc_stephan_boltzmann(T_K):
    '''Calculates the total energy radiated by a blackbody.

    Parameters
    ----------
    T_K : float
        body temperature (Kelvin)

    Returns
    -------
    M : float
        Emitted radiance (W m-2)'''

    M = sb * T_K**4
    return np.asarray(M)

def calc_vapor_pressure(T_K,Rh_Avg):
    """Calculate the (saturation) water vapour pressure.

    Parameters
    ----------
    T_K : float
        temperature (K).

    Returns
    -------
    e_Avg : float
          water vapour pressure (Kpa).
    esat_air: float
          saturation water vapour pressure (Kpa).
    """
    T_C = T_K - 273.15
    e_Avg          =  0.6108*np.exp(17.27*T_C/(T_C + 237.3))*Rh_Avg  #; %Kpa
    esat_air       =  0.6108*np.exp(17.27*T_C/(T_C + 237.3)) #Kpa

    return e_Avg, esat_air

def calc_delta_vapor_pressure(T_K):
    """Calculate the slope of saturation water vapour pressure.

    Parameters
    ----------
    T_K : float
        temperature (K).

    Returns
    -------
    s : float
        slope of the saturation water vapour pressure (kPa K-1)
    """

    T_C = T_K - 273.15
    s = 4098.0 * (0.6108 * np.exp(17.27 * T_C / (T_C + 237.3))) / ((T_C + 237.3)**2)
    return np.asarray(s)

def calc_mixing_ratio(ea, p):
    '''Calculate ratio of mass of water vapour to the mass of dry air (-)

    Parameters
    ----------
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).

    Returns
    -------
    r : float or numpy array
        mixing ratio (-)

    References
    ----------
    http://glossary.ametsoc.org/wiki/Mixing_ratio'''

    r = epsilon * ea / (p - ea)
    return r

def calc_lapse_rate_moist(T_A_K, ea, p):
    '''Calculate moist-adiabatic lapse rate (K/m)

    Parameters
    ----------
    T_A_K : float or numpy array
        air temperature at reference height (K).
    ea : float or numpy array
        water vapor pressure at reference height (mb).
    p : float or numpy array
        total air pressure (dry air + water vapour) at reference height (mb).

    Returns
    -------
    Gamma_w : float or numpy array
        moist-adiabatic lapse rate (K/m)

    References
    ----------
    http://glossary.ametsoc.org/wiki/Saturation-adiabatic_lapse_rate'''

    r = calc_mixing_ratio(ea, p)
    c_p = calc_c_p(p, ea)
    lambda_v = calc_lambda(T_A_K)
    Gamma_w = ((gravity * (R_d * T_A_K**2 + lambda_v * r * T_A_K)
               / (c_p * R_d * T_A_K**2 + lambda_v**2 * r * epsilon)))
    return Gamma_w


UNSTABLE_THRES = None
TABLE_THRES = None

def calc_MO_Length(ustar, T_A_K, rho, c_p, H, LE):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
            Cambridge: Cambridge University Press.'''

    l_mo = calc_mo_length_hv(ustar, T_A_K, rho, c_p, H, LE)
    return np.asarray(l_mo)

def calc_mo_length(ustar, T_A_K, rho, c_p, H):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.'''

    # Convert input scalars to numpy arrays
    ustar, T_A_K, rho, c_p, H = map(
        np.asarray, (ustar, T_A_K, rho, c_p, H))

    L = np.asarray(np.ones(ustar.shape) * float('inf'))
    i = H != 0
    L[i] = - c_p[i] * T_A_K[i] * rho[i] * ustar[i]**3 / (karman * gravity * H[i])
    return np.asarray(L)

def calc_mo_length_hv(ustar, T_A_K, rho, c_p, H, LE):
    '''Calculates the Monin-Obukhov length.

    Parameters
    ----------
    ustar : float
        friction velocity (m s-1).
    T_A_K : float
        air temperature (Kelvin).
    rho : float
        air density (kg m-3).
    c_p : float
        Heat capacity of air at constant pressure (J kg-1 K-1).
    H : float
        sensible heat flux (W m-2).
    LE : float
        latent heat flux (W m-2).

    Returns
    -------
    L : float
        Obukhov stability length (m).

    References
    ----------
    .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
        Cambridge: Cambridge University Press.'''

    # Convert input scalars to numpy arrays
    ustar, T_A_K, rho, c_p, H, LE = map(
        np.asarray, (ustar, T_A_K, rho, c_p, H, LE))
    # first convert latent heat into rate of surface evaporation (kg m-2 s-1)
    Lambda = calc_lambda(T_A_K)  # in J kg-1
    E = LE / Lambda
    del LE, Lambda
    # Virtual sensible heat flux
    Hv = H + (0.61 * T_A_K * c_p * E)
    del H, E

    L = np.asarray(np.ones(ustar.shape) * float('inf'))
    i = Hv != 0
    L_const = np.asarray(karman * gravity / T_A_K)
    L[i] = -ustar[i]**3 / (L_const[i] * (Hv[i] / (rho[i] * c_p[i])))
    return np.asarray(L)

# def calc_Psi_H(zoL):
#     ''' Calculates the adiabatic correction factor for heat transport.
#
#     Parameters
#     ----------
#     zoL : float
#         stability coefficient (unitless).
#
#     Returns
#     -------
#     Psi_H : float
#         adiabatic corrector factor fof heat transport (unitless).
#
#     References
#     ----------
#     .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
#         Cambridge: Cambridge University Press.
#     '''
#     # Avoid free convection situations
#     if UNSTABLE_THRES is not None or STABLE_THRES is not None:
#         zoL = np.clip(zoL, UNSTABLE_THRES, STABLE_THRES)
#     Psi_H = psi_h_brutsaert(zoL)
#     return np.asarray(Psi_H)

def psi_h_dyer(zol):

    gamma = 16
    beta = 5
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    psi_h = np.zeros(zol.shape)
    finite = np.isfinite(zol)
    # for stable and netural (zoL = 0 -> Psi_H = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    psi_h[i] = -beta * zol[i]
    # for unstable conditions
    i = np.logical_and(finite, zol < 0.0)
    x = (1 - gamma * zol[i])**0.25
    psi_h[i] = 2 * np.log((1 + x**2) / 2)
    return psi_h

def psi_h_brutsaert(zol):
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    psi_h = np.zeros(zol.shape)
    finite = np.isfinite(zol)

    # for stable and netural (zoL = 0 -> Psi_H = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    a = 6.1
    b = 2.5
    psi_h[i] = -a * np.log(zol[i] + (1.0 + zol[i]**b)**(1. / b))
    
    # for unstable conditions
    i = np.logical_and(finite, zol < 0.0)
    y = -zol[i]
    del zol
    c = 0.33
    d = 0.057
    n = 0.78
    psi_h[i] = ((1.0 - d) / n) * np.log((c + y**n) / c)
    return psi_h

# def calc_Psi_M(zoL):
#     ''' Adiabatic correction factor for momentum transport.
#
#     Parameters
#     ----------
#     zoL : float
#         stability coefficient (unitless).
#
#     Returns
#     -------
#     Psi_M : float
#         adiabatic corrector factor fof momentum transport (unitless).
#
#     References
#     ----------
#     .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
#         Cambridge: Cambridge University Press.
#     '''
#     # Avoid free convection situations
#     if UNSTABLE_THRES is not None or STABLE_THRES is not None:
#         zoL = np.clip(zoL, UNSTABLE_THRES, STABLE_THRES)
#     Psi_M = psi_m_brutsaert(zoL)
#     return np.asarray(Psi_M)

def psi_m_dyer(zol):
    gamma = 16
    beta = 5
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    finite = np.isfinite(zol)
    psi_m = np.zeros(zol.shape)
    # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    psi_m[i] = -beta * zol[i]
    # for unstable conditions
    i = np.logical_and(finite, zol < 0.0)
    x = (1 - gamma * zol[i]) ** 0.25
    psi_m[i] = np.log((1 + x ** 2) / 2) + 2 * np.log((1 + x) / 2) \
               - 2 * np.arctan(x) + np.pi / 2.

    return psi_m

def psi_m_brutsaert(zol):
    # Convert input scalars to numpy array
    zol = np.asarray(zol)
    finite = np.isfinite(zol)
    psi_m = np.zeros(zol.shape)
    # for stable and netural (zoL = 0 -> Psi_M = 0) conditions
    i = np.logical_and(finite, zol >= 0.0)
    a = 6.1
    b = 2.5
    psi_m[i] = -a * np.log(zol[i] + (1.0 + zol[i]**b)**(1.0 / b))
    # for unstable conditions
    i = np.logical_and(finite, zol < 0)
    y = -zol[i]
    del zol
    a = 0.33
    b = 0.41
    x = np.asarray((y / a)**0.333333)

    psi_0 = -np.log(a) + 3**0.5 * b * a**0.333333 * np.pi / 6.0
    y = np.minimum(y, b**-3)
    psi_m[i] = (np.log(a + y) - 3.0 * b * y**0.333333 +
                (b * a**0.333333) / 2.0 * np.log((1.0 + x)**2 / (1.0 - x + x**2)) +
                3.0**0.5 * b * a**0.333333 * np.arctan((2.0 * x - 1.0) / 3**0.5) +
                psi_0)

    return psi_m

def calc_richardson(u, z_u, d_0, T_R0, T_R1, T_A0, T_A1):
    '''Richardson number.

    Estimates the Bulk Richardson number for turbulence using
    time difference temperatures.

    Parameters
    ----------
    u : float
        Wind speed (m s-1).
    z_u : float
        Wind speed measurement height (m).
    d_0 : float
        Zero-plane displacement height (m).
    T_R0 : float
        radiometric surface temperature at time 0 (K).
    T_R1 : float
        radiometric surface temperature at time 1 (K).
    T_A0 : float
        air temperature at time 0 (K).
    T_A1 : float
        air temperature at time 1 (K).

    Returns
    -------
    Ri : float
        Richardson number.

    References
    ----------
    .. [Norman2000] Norman, J. M., W. P. Kustas, J. H. Prueger, and G. R. Diak (2000),
        Surface flux estimation using radiometric temperature: A dual-temperature-difference
        method to minimize measurement errors, Water Resour. Res., 36(8), 2263-2274,
        http://dx.doi.org/10.1029/2000WR900033.
    '''

    # See eq (2) from Louis 1979
    Ri = -(gravity * (z_u - d_0) / T_A1) * \
          (((T_R1 - T_R0) - (T_A1 - T_A0)) / u**2) # equation (12) [Norman2000]
    return np.asarray(Ri)

# def calc_u_star(u, z_u, L, d_0, z_0M):
#     '''Friction velocity.
#
#     Parameters
#     ----------
#     u : float
#         wind speed above the surface (m s-1).
#     z_u : float
#         wind speed measurement height (m).
#     L : float
#         Monin Obukhov stability length (m).
#     d_0 : float
#         zero-plane displacement height (m).
#     z_0M : float
#         aerodynamic roughness length for momentum transport (m).
#
#     References
#     ----------
#     .. [Brutsaert2005] Brutsaert, W. (2005). Hydrology: an introduction (Vol. 61, No. 8).
#         Cambridge: Cambridge University Press.
#     '''
#
#     # Covert input scalars to numpy arrays
#     u, z_u, L, d_0, z_0M = map(np.asarray, (u, z_u, L, d_0, z_0M))
#
#     # calculate correction factors in other conditions
#     L[L == 0.0] = 1e-36
#     Psi_M = calc_Psi_M((z_u - d_0) / L)
#     Psi_M0 = calc_Psi_M(z_0M / L)
#     del L
#     u_star = u * karman / (np.log((z_u - d_0) / z_0M) - Psi_M + Psi_M0)
#     return np.asarray(u_star)

def calc_surface_temperature_WLR(Rld,Rlu,emis_surf):
    '''Surface temperature.

    Parameters
    ----------
    Rld : float
        Downward longwave radiation (W m-2).
    Rlu : float
        Upward longwave radiation (W m-2).
    emis_surf : float
        Surface emissivity (0-1). emis_surf=0.96 is set to default

    Returns
    -------
    T_surf : float
        Surface temperature (K).

    References
    ----------
    .. [Wang2021] Wang, J., Luo, S., Lv, Z., Li, W., Tan, X., Dong, Q., & Chen, Z. (2021). 
    Improving ground heat flux estimation: Considering the effect of freeze/thaw process on the
      seasonally frozen ground. Journal of Geophysical Research: Atmospheres, 126(24), e2021JD035445.
    '''

    # Convert input scalars to numpy arrays
    Rld, Rlu, emis_surf = map(np.asarray, (Rld, Rlu, emis_surf))
   
    # Calculate surface temperature
    T_surf = ((Rlu-(1-emis_surf)*Rld )/(emis_surf*sb)) **0.25

    return np.asarray(T_surf)


    


def moninobukini_bk(Tair_K,Psurf,Z_t,Z_U,d_0,Qair,LWdown,LWup,Qground,ur,z0m):
    
    #thm,        &! intermediate variable (tm+0.0098*ht)
    #th,         &! potential temperature (kelvin)
    #thv,        &! virtual potential temperature (kelvin)
    #dth,        &! diff of virtual temp. between ref. height and surface
    #dqh,        &! diff of humidity between ref. height and surface
    #dthv,       &! diff of vir. poten. temp. between ref. height and surface
    #zldis,      &! reference height "minus" zero displacement heght [m]
    #z0m,        &! aerodynamic roughness length for momentum transport [m]
    #qm,         &! specific humidity at reference height [kg/kg]

    # Constants
    grav   = 9.81  # gravitational acceleration [m/s^2]
    vonkar = 0.4  # von Karman constant
 
    # Local variables
    wc     = 0.5       # convective velocity [m/s]
    rib    = 0.0       # bulk Richardson number
    zeta   = 0.0       # dimensionless height used in Monin-Obukhov theory
    rgas   = 287.04    # gas constant for dry air [J/kg/K]
    cpair  = 1004.64   # specific heat of dry air [J/kg/K]
    # potential temperatur at the reference height
    thm = Tair_K  + 0.0098*Z_t              #  !intermediate variable equivalent to
                                                  #  !forc_t*(pgcm/forc_psrf)**(rgas/cpair)
    th  = Tair_K *(100./Psurf)**(rgas/cpair) #  potential T 
    thv = th*(1.+0.61*Qair)                     #  virtual potential T
    emis_surf=0.96
    tg  = calc_surface_temperature_WLR(LWdown, LWup, emis_surf)
    taf = 0.5 * (tg + thm)
    qaf = 0.5 * (Qair + Qground)

    dth = thm - taf
    dqh = Qair - qaf
    dthv = dth*(1.+0.61*Qair) + 0.61*th*dqh
    zldis = Z_U - d_0


    # Initial values of u* and convective velocity
    um = (ur**2 + wc**2)**0.5
    um =np.where(dthv >= 0.0,np.maximum(um, 0.1),um)

    rib = grav * zldis * dthv / (thv * um**2)


    zeta =rib
    zeta =np.where(rib >= 0.0,rib * np.log(zldis / z0m) / (1.0 - 5.0 * np.minimum(rib, 0.19)),rib * np.log(zldis / z0m))
    zeta =np.where(rib >= 0.0, np.minimum(2.0, np.maximum(zeta, 1e-6)),np.maximum(-100.0, np.minimum(zeta, -1e-6)))
    


    '''
    if rib >= 0.0:  # neutral or stable
        zeta = rib * np.log(zldis / z0m) / (1.0 - 5.0 * min(rib, 0.19))
        zeta = min(2.0, max(zeta, 1e-6))
    else:  # unstable
        zeta = rib * np.log(zldis / z0m)
        zeta = max(-100.0, min(zeta, -1e-6))
    '''
    obu = zldis / zeta

    return um, obu

def dewfraction_2(RH):
    '''
    Shao, R., Shao, W., Gu, C., & Zhang, B. (2022). 
    Increased Interception Induced by Vegetation Restoration Counters Ecosystem Carbon and Water Exchange Efficiency in China. 
    Earth’s Future, 10(2). https://doi.org/10.1029/2021EF002464
    '''
    fwet=RH**4.0

    return fwet

def dewfraction_1(sigf, lai, sai, dewmx, ldew, ldew_rain, ldew_snow):
    """
    Determine fraction of foliage covered by water and fraction of foliage that is dry and transpiring.

    Args:
        sigf (float): fraction of veg cover, excluding snow-covered veg [-]
        lai (float): leaf area index [-]
        sai (float): stem area index [-]
        dewmx (float): maximum allowed dew [0.1 mm]
        ldew (float): depth of water on foliage [kg/m2/s]
        ldew_rain (float): depth of rain on foliage [kg/m2/s]
        ldew_snow (float): depth of snow on foliage [kg/m2/s]

    Returns:
        fwet (float): fraction of foliage covered by water [-]
        fdry (float): fraction of foliage that is green and dry [-]
    """
    # Constants
    lsai = lai + sai
    dewmxi = 1.0 / dewmx
    vegt = lsai

    # Calculate fwet
    fwet = 0
    if ldew > 0:
        fwet = ((dewmxi / vegt) * ldew) ** 0.666666666666
        fwet = np.minimum(fwet, 1.0)

    # Calculate fdry
    fdry = (1 - fwet) * lai / lsai
    return fwet, fdry

def qair2rh(qair,temp,press):
  #' Convert specific humidity to relative humidity
  #'
  #' converting specific humidity into relative humidity
  #' NCEP surface flux data does not have RH
  #' from Bolton 1980 The computation of Equivalent Potential Temperature 
  #' \url{http://www.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html}
  #' @title qair2rh
  #' @param qair specific humidity, dimensionless (e.g. kg/kg) ratio of water mass / total air mass
  #' @param temp degrees C
  #' @param press pressure in mb
  #' @return rh relative humidity, ratio of actual water mixing ratio to saturation mixing ratio
  #' @export
  #' @author David LeBauer
  qair,temp,press  =  map(np.asarray, (qair,temp,press))
  es = 6.112 * np.exp((17.67 * temp)/(temp + 243.5))
  e  = qair * press / (0.378 * qair + 0.622)
  rh = e / es
  rh =  np.where(rh <= 1.0, rh, 1.0)
  rh =  np.where(rh > 0.0,  rh, 0.1)

  return rh
