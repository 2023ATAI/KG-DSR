import numpy as np
from . import metlib as metlib
from . import resistance as reslib
def calc_G_time_diff(R_n, G_param=[12.0, 0.35, 3.0, 24.0]):
    ''' Estimates Soil Heat Flux as function of time and net radiation.

    Parameters
    ----------
    R_n : float
        Net radiation (W m-2).
    G_param : tuple(float,float,float,float)
        tuple with parameters required (time, Amplitude,phase_shift,shape).

            time: float
                time of interest (decimal hours).
            Amplitude : float
                maximum value of G/Rn, amplitude, default=0.35.
            phase_shift : float
                shift of peak G relative to solar noon (default 3hrs before noon).
            shape : float
                shape of G/Rn, default 24 hrs.

    Returns
    -------
    G : float
        Soil heat flux (W m-2).

    References
    ----------
    .. [Santanello2003] Joseph A. Santanello Jr. and Mark A. Friedl, 2003: Diurnal Covariation in
        Soil Heat Flux and Net Radiation. J. Appl. Meteor., 42, 851-862,
        http://dx.doi.org/10.1175/1520-0450(2003)042<0851:DCISHF>2.0.CO;2.'''

    # Get parameters
    time = G_param[0] - 12.0
    A = G_param[1]
    phase_shift = G_param[2]
    B = G_param[3]
    G_ratio = A * np.cos(2.0 * np.pi * (time + phase_shift) / B)
    G = R_n * G_ratio
    return np.asarray(G, dtype=np.float32)

def calc_G_Rn_ratio(Rn, G_ratio=0.2):
    '''Estimates Soil Heat Flux as ratio of net  radiation.

    Parameters
    ----------
    Rn : float
        Net  radiation (W m-2).
    G_ratio : float, optional
        G/Rn ratio, default=0.2.

    Returns
    -------
    G : float
        Soil heat flux (W m-2).

    References
    ----------
    .. [Choudhury1987] B.J. Choudhury, S.B. Idso, R.J. Reginato, Analysis of an empirical model
        for soil heat flux under a growing wheat crop for estimating evaporation by an
        infrared-temperature based energy balance equation, Agricultural and Forest Meteorology,
        Volume 39, Issue 4, 1987, Pages 283-297,
        http://dx.doi.org/10.1016/0168-1923(87)90021-9.
    '''

    G = G_ratio * Rn
    return G #np.asarray(G, dtype=np.float32)

def calc_difuse_ratio(S_dn, sza, press=1013.25, SOLAR_CONSTANT=1320):
    """Fraction of difuse shortwave radiation.
    Partitions the incoming solar radiation into PAR and non-PR and
    diffuse and direct beam component of the solar spectrum.
    Parameters
    ----------
    S_dn : float
         Incoming shortwave radiation (W m-2).
    sza : float
        Solar Zenith Angle (degrees).
    Wv : float, optional
        Total column precipitable water vapour (g cm-2), default 1 g cm-2.
    press : float, optional
        atmospheric pressure (mb), default at sea level (1013mb).

    Returns
    -------
    difvis : float
        diffuse fraction in the visible region.
    difnir : float
        diffuse fraction in the NIR region.
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.

    References
    ----------
    .. [Weiss1985] Weiss and Norman (1985) Partitioning solar radiation into direct and diffuse,
        visible and near-infrared components, Agricultural and Forest Meteorology,
        Volume 34, Issue 2, Pages 205-213,
        http://dx.doi.org/10.1016/0168-1923(85)90020-6.
    """

    # Convert input scalars to numpy arrays
    #S_dn, sza, press = map(np.asarray, (S_dn, sza, press))
    difvis, difnir, fvis, fnir = [np.zeros(S_dn.shape) for i in range(4)]
    fvis = fvis + 0.6
    fnir = fnir + 0.4

    # Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = calc_potential_irradiance_weiss(
        sza, press=press, SOLAR_CONSTANT=SOLAR_CONSTANT,fnir_ini=fnir)

    # Potential total solar radiation
    potvis = np.asarray(Rdirvis + Rdifvis)
    potvis[potvis <= 0] = 1e-6
    potnir = np.asarray(Rdirnir + Rdifnir)
    potnir[potnir <= 0] = 1e-6
    fclear = S_dn / (potvis + potnir)
    fclear = np.minimum(1.0, fclear)

    # Partition S_dn into VIS and NIR
    fvis = potvis / (potvis + potnir)  # Eq. 7
    fnir = potnir / (potvis + potnir)  # Eq. 8
    fvis = np.clip(fvis, 0.0, 1.0)
    fnir = 1.0 - fvis

    # Estimate direct beam and diffuse fractions in VIS and NIR wavebands
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.9] = 0.9
    dirvis = (Rdirvis / potvis) * (1. - ((.9 - ratiox) / .7)**.6667)  # Eq. 11
    ratiox = np.asarray(fclear)
    ratiox[fclear > 0.88] = 0.88
    dirnir = (Rdirnir / potnir) * \
            (1. - ((.88 - ratiox) / .68)**.6667)  # Eq. 12

    dirvis = np.clip(dirvis, 0.0, 1.0)
    dirnir = np.clip(dirnir, 0.0, 1.0)
    difvis = 1.0 - dirvis
    difnir = 1.0 - dirnir

    return difvis,difnir,fvis,fnir 

def calc_net_radiation_FAO_refcrop(Rsd, Rld, Tair,albedo):
    ''' 
    Calculates the net radiation using the FAO Penman-Monteith equation
    '''
    # Stephan Boltzmann constant (W m-2 K-4)
    sb = 5.670373e-8
    Rn=(1.0-albedo)*Rsd+0.9*(Rld-sb*Tair**4)

    return Rn

def calc_net_radiation_FAO(DOY, Hour,T, RH, Rs,lat,lon,z,albedo):
    #TODO: need to check,bug may exist
    ''' 
    Calculates the net radiation using the FAO equation
    Input:
        isodate: iso datetime (in UTC?)
        T: hourly air temperature at 2m [Celsius]
        RH: hourly relative air humidity [%]
        u2: hourly wind speed at 2 m [m/s]
        Rs: hourly incoming solar radiation [W/m2]
        lat: latitude of the mesurement point [decimal degree]
        z: altitude above sea level of the measurement point [m]
        albedo: hourly air pressure [Pa] (Opzional)
    Output:
        Rn: hourly net radiation [W/m2]
    Examples::
        >>> (isodate="2012-10-01T14:00Z",T=38,RH=52,u2=3.3,Rs=2.450,lat=16.21,z=8,albedo=0.23)
    '''
    # convert from  W/m2 to MJ/m2/hour 
    Rs=Rs * 0.0036 #need to check this
    # get datetime object
    #dt = isodate #parser.parse(isodate)
    #print("dt: %s" % dt)
    # air pressure (calculation)
    #if not P:
    #    P = 101.3 * ((293-0.0065*z)/293)**5.26
    # saturation vapour pressure (eq.11) [kPa]
    e0T = 0.6108 * np.exp((17.27*T)/(T+237.3))
    print("e0T: %s" % e0T)

    ea = e0T * RH / 100
    print("ea: %s" % ea)

    # Pressure deficit [kPa]
    #ed = e0T - ea
    #print("ed: %s" % ed)

    # Extraterrestrial radiation
    # ============================
    # solar costant [ MJ / m2 * min ]
    Gsc = 0.0820
    #print("Gsc: %s" % Gsc)
    
    # Convert latitude [degrees] to radians
    j = lat * np.pi / 180.0
    #print("j: %s" % j)


    # day of the year [-]
    #tt = dt.timetuple()
    J  = DOY - 1
    #print("J: %s" % J)

    # inverse relative distance Earth-Sun (eq.23) [-]
    dr = 1.0 + 0.033 * np.cos(J * ((2*np.pi)/365) )
    #print("dr: %s" % dr)

    # solar declination (eq.24) [rad]
    d = 0.409 * np.sin( (2*np.pi/365)*J - 1.39)
    #print("d: %s" % d)

    # solar correction (eq.33) [-]
    b = 2 * np.pi * (J-81) / 364
    #print("b: %s" % b)

    # Seasonal correction for solar time
    Sc = (0.1645 * np.sin(2*b)) - (0.1255 * np.cos(b)) - (0.025 * np.sin(b))
    #print("Sc: %s" % Sc)

    # longitude of the centre of the local time zone
    Lz = round(lon/15) * 15
    #print("Lz: %s" % Lz)

    # Solartime angle
    t = Hour + 0.5
    #print("t: %s" % t)

    # standard clock time at the midpoint of the perion [hour]
    w = (np.pi/12) * ((t + 0.06667*(np.fabs(Lz)-np.fabs(lon)) + Sc) - 12)
    #print("w: %s" % w)

    # time interval [hour]
    ti = 1

    # solar time angle at begin of the period
    w1 = w - (np.pi*ti)/24
    #print("w1: %s" % w1)

    # solar time angle at end of the period
    w2 = w + (np.pi*ti)/24
    #print("w2: %s" % w2)

    # Sunset hour angle
    ws = np.arccos(-np.tan(j) * np.tan(d))
    #print("ws: %s" % ws)

    # Extraterrestrial radiation (eq.28) [MJ m-2 * hour-1 ]
    Ra = (12*60/np.pi) * Gsc * dr * ((w2-w1)*np.sin(j)*np.sin(d) + np.cos(j)*np.cos(d)*(np.sin(w2)-np.sin(w1)) )
    mask = np.logical_or(w < -ws, w > ws)

    Ra = np.where(mask,0.0, Ra)

    # Net solar radiation
    # ============================
    # for reference hypotetical grass reference crop [-]
    #albedo = 0.23
    # net solar radiation [ MJ * m-2 * hour-1 ]
    Rns = (1 - albedo) * Rs
    Rns = np.where(mask,0.0, Rns)

    # clear sky solar radiation [ MJ * m-2 * hour-1 ]
    Rso = (0.75 + 2 * (10**-5) * z) * Ra

    # Steffan-Boltzman constant [ MJ * K-4 * m-2 * hour-1 ]
    sbc = 2.04 * (10**-10)

    # cloudiness factor
    f = Rso
    f = np.where((Rso!=0.0), 1.35 * Rs/Rso - 0.35, Rso)

    # net emissivity of the surface
    nes = 0.34 - 0.14 * np.sqrt(ea)
    nes = np.where((ea < 0),0.34, nes)
    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    k=1-nes * f
    # Net longwave Radiation
    Rnl = nes * f * (sbc * (T + 273.15)**4)
    
    # Net Radiation
    # ============================
    Rn = Rns - Rnl
    #print("Rn: %s" % Rn)
    Rn=Rn/0.036 # convert to W/m2

    '''
    # if w < -ws or w > ws:
    #     Ra = 0
    # else:
    #     Ra = (12*60/np.pi) * Gsc * dr * ((w2-w1)*np.sin(j)*np.sin(d) + np.cos(j)*np.cos(d)*(np.sin(w2)-np.sin(w1)) )
    
    # print("Ra: %s" % Ra)
    #----------------------------------------------------------------
    # Solar radiation [ MJ * m-2 * hour-1 ]
    if w < -ws or w > ws:
        # solar radiation
        Rs = 0
        print("Rs: %s" % Rs)

        # clear sky solar radiation [ MJ * m-2 * hour-1 ]
        Rso = 0
        print("Rso: %s" % Rso)

        # net solar radiation [ MJ * m-2 * hour-1 ]
        Rns = 0
        print("Rns: %s" % Rns)
    else:
        # solar radiation
        Rs = 2.450
        print("Rs: %s" % Rs)

        # clear sky solar radiation [ MJ * m-2 * hour-1 ]
        Rso = (0.75 + 2 * (10**-5) * z) * Ra
        print("Rso: %s" % Rso)

        # for reference hypotetical grass reference crop [-]
        albedo = 0.23

        # net solar radiation [ MJ * m-2 * hour-1 ]
        Rns = (1 - albedo) * Rs
        print("Rns: %s" % Rns)
    
    # Net longwave Radiation
    # ============================

    # Steffan-Boltzman constant [ MJ * K-4 * m-2 * hour-1 ]
    sbc = 2.04 * (10**-10)

    # cloudiness factor
    if Rso is 0:
        f = 1.35 * 0.8 - 0.35
    else:
        f = 1.35 * (Rs/Rso) - 0.35
    print("f: %s" % f)

    # net emissivity of the surface
    if ea < 0:
        nes = 0.34
    else:
        nes = 0.34 - 0.14 * np.sqrt(ea)
    print("nes: %s" % nes)

    # Net longwave Radiation
    Rnl = nes * f * (sbc * (T + 273)**4)
    print("Rnl: %s" % Rnl)

    # Net Radiation
    # ============================
    Rn = Rns - Rnl
    print("Rn: %s" % Rn)
    '''
    return Rn

def calc_net_radiation_TESB_SW(LWdown,SWdown,SZA,press,Tair_K,LAI,f_c,IGBP_CLASS, z0_soil,z0m,Zc, d0, x_LAD       =    1.0):
    print('TESB_SW is used for net radiation calculation')
    #leaf angle distribution parameter
    ladf       =     np.full_like(LAI,IGBP_CLASS)

    # Spectral Properties
    rho_vis_C = np.full_like(LAI,0.07)  #Leaf PAR Reflectance
    rho_nir_C = np.full_like(LAI,0.32)  #Leaf NIR Reflectance
    tau_vis_C = np.full_like(LAI,0.08)  #Leaf PAR Transmitance
    tau_nir_C = np.full_like(LAI,0.33)  #Canopy NIR Transmitance
    rho_vis_S = np.full_like(LAI,0.15)  #Soil PAR Reflectance
    rho_nir_S = np.full_like(LAI,0.25)  #Soil NIR Reflectance

    emis_C    = np.full_like(LAI,0.98)  #Leaf Emissivity
    emis_S    = np.full_like(LAI,0.95)  #Soil Emissivity
    Rnet      = np.full_like(LAI,0.0)
  
    IGBP_ladf  = {
                0:   0.010,   # water
                1:   0.010 ,  # evergreen needleleaf forest
                2:   0.100 ,  # evergreen broadleaf forest
                3:   0.010 ,  # deciduous needleleaf forest
                4:   0.250 ,  # deciduous broadleaf forest
                5:   0.125 ,  # mixed forests
                6:   0.010 ,   # closed shrubland
                7:   0.010 ,   # open shrublands
                8:   0.010 ,   # woody savannas
                9:   0.010 ,   # savannas
                10:  -0.300 ,  # grasslands 
                11:  0.100 ,  # permanent wetlands
                12:  -0.300 ,  # croplands
                13:  0.010 ,  # urban and built-up
                14:  -0.300 ,  # cropland natural vegetation mosaic
                15:  0.010 ,    # snow and ice
                16:  0.010   # barren or sparsely vegetated
                }
    #TODO: need to check          
    #map_ladf       =    np.vectorize(lambda x: IGBP_ladf[x])
    #ladf           =    map_ladf(ladf)
    ladf = np.vectorize(lambda x: IGBP_ladf[x])(IGBP_CLASS)

    Sn             =    np.full_like(LAI, -9999.0)

    _, _, _, taudl = calc_spectra_Cambpell(LAI,
                                                    np.zeros(Sn.shape),
                                                    1.0 - emis_C,
                                                    np.zeros( Sn.shape),
                                                    1.0 -  emis_S,
                                                    x_lad=x_LAD,
                                                    lai_eff=None)
                
    emiss       =    taudl * emis_S + (1 - taudl) * emis_C
    Ln          =    emiss * (LWdown - metlib.calc_stephan_boltzmann(Tair_K))
    Ln_C        =    (1. - taudl) * Ln
    Ln_S        =    taudl * Ln


    #fvis: fration of total visible radiation.
    #fnir: fration of total near-infrared radiation.
    #Skyl: Broadband diffuse skylight (W m-2).
    #S_dn_dir: Broadband direct solar radiation (W m-2).
    #S_dn_dif: Broadband incoming diffuse shortwave radiation (W m-2).
    #difvis: Broadband diffuse visible radiation (W m-2).
    #difnir: Broadband diffuse near-infrared radiation (W m-2).

    #mask 
    mask = np.ones(LAI.shape, np.int32)

    difvis, difnir, fvis, fnir =    calc_difuse_ratio(SWdown, SZA, press) #press in mb
    Skyl                       =    difvis * fvis +  difnir *  fnir  
    S_dn_dir                   =    SWdown * (1.0- Skyl)  
    S_dn_dif                   =    SWdown * Skyl  
    w_C                        =    np.full_like(LAI,1.0)
    
    # ======================================
    # bare soil cases
    noVegPixels = LAI <= 0
    noVegPixels = np.logical_or.reduce(
                    (f_c <= 0.01,
                    LAI <= 0,
                    np.isnan(LAI)))
    # in_data['LAI'][noVegPixels] = 0
    # in_data['f_c'][noVegPixels] = 0
    i = np.array(np.logical_and(noVegPixels, mask == 1))

    # Calculate roughness
    z0m[i] = z0_soil[i]
    d0[i]  = 0.0

    # Net shortwave radition for bare soil
    spectraGrdOSEB = fvis *  rho_vis_S +  fnir *  rho_nir_S
    Sn[i] = (1. - spectraGrdOSEB[i]) * (S_dn_dir[i] + S_dn_dif[i])
    Rnet[i] = Sn[i] + Ln[i]

    # ======================================
    # canopy surface cases
    i = np.array(np.logical_and(~noVegPixels, mask == 1))

    # Calculate roughness
    z0m[i],d0[i] = reslib.calc_roughness_Schaudt2000(LAI[i],
                               Zc[i],
                               w_C[i],
                               landcover=IGBP_CLASS[i],
                               f_c=f_c[i])

    # Net shortwave radiation for vegetation
    F    = np.zeros(LAI.shape, np.float32)
    F[i] = LAI[i] /f_c[i]

    # Clumping index
    omega0    = np.zeros(LAI.shape, np.float32)
    Omega     = np.zeros(LAI.shape, np.float32)
    omega0[i] = calc_omega0_Kustas(LAI[i], f_c[i],x_LAD=x_LAD,isLAIeff=True)
    Omega[i]  = calc_omega_Kustas(omega0[i], SZA[i], w_C=1.0)
    LAI_eff = F * Omega

    Sn_C= np.ones(LAI.shape, np.float32)
    Sn_S= np.ones(LAI.shape, np.float32)

    [Sn_C[i],Sn_S[i]] = calc_Sn_Campbell(LAI[i],SZA[i],S_dn_dir[i],
                                        S_dn_dif[i], fvis[i], fnir[i],
                                        rho_vis_C[i], tau_vis_C[i],
                                        rho_nir_C[i], tau_nir_C[i],
                                        rho_vis_S[i], rho_nir_S[i],
                                        x_LAD=1.0, LAI_eff=LAI_eff[i])
                
    Sn[i] = Sn_C[i] + Sn_S[i]

    Rnet[i] = Sn[i] + Ln[i]
    return Rnet

def calc_longwave_irradiance(ea, t_a_k, p=1013.25, z_T=2.0, h_C=2.0):
    '''Longwave irradiance

    Estimates longwave atmospheric irradiance from clear sky.
    By default there is no lapse rate correction unless air temperature
    measurement height is considerably different than canopy height, (e.g. when
    using NWP gridded meteo data at blending height)

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (K).
    p : float
        air pressure (mb)
    z_T: float
        air temperature measurement height (m), default 2 m.
    h_C: float
        canopy height (m), default 2 m,

    Returns
    -------
    L_dn : float
        Longwave atmospheric irradiance (W m-2) above the canopy
    '''

    lapse_rate = metlib.calc_lapse_rate_moist(t_a_k, ea, p)
    t_a_surface = t_a_k - lapse_rate * (h_C - z_T)
    emisAtm = calc_emiss_atm(ea, t_a_surface)
    L_dn = emisAtm * metlib.calc_stephan_boltzmann(t_a_surface)
    return np.asarray(L_dn)

def calc_potential_irradiance_weiss(
        sza,
        press=1013.25,
        SOLAR_CONSTANT=1320,
        fnir_ini=0.5455):
    ''' Estimates the potential visible and NIR irradiance at the surface

    Parameters
    ----------
    sza : float
        Solar Zenith Angle (degrees)
    press : Optional[float]
        atmospheric pressure (mb)

    Returns
    -------
    Rdirvis : float
        Potential direct visible irradiance at the surface (W m-2)
    Rdifvis : float
        Potential diffuse visible irradiance at the surface (W m-2)
    Rdirnir : float
        Potential direct NIR irradiance at the surface (W m-2)
    Rdifnir : float
        Potential diffuse NIR irradiance at the surface (W m-2)

    based on Weiss & Normat 1985, following same strategy in Cupid's RADIN4 subroutine.
    '''

    # Convert input scalars to numpy arrays
    sza, press = map(np.asarray, (sza, press))
    # Set defaout ouput values
    Rdirvis, Rdifvis, Rdirnir, Rdifnir, w = [
        np.zeros(sza.shape) for i in range(5)]

    coszen = np.cos(np.radians(sza))
    # Calculate potential (clear-sky) visible and NIR solar components
    # Weiss & Norman 1985
    # Correct for curvature of atmos in airmas (Kasten and Young,1989)
    i = sza < 90
    airmas = 1.0 / coszen
    # Visible PAR/NIR direct beam radiation
    Sco_vis = SOLAR_CONSTANT * (1.0 - fnir_ini)
    Sco_nir = SOLAR_CONSTANT * fnir_ini
    # Directional trasnmissivity
    # Calculate water vapour absorbance (Wang et al 1976)
    # A=10**(-1.195+.4459*np.log10(1)-.0345*np.log10(1)**2)
    # opticalDepth=np.log(10.)*A
    # T=np.exp(-opticalDepth/coszen)
    # Asssume that most absortion of WV is at the NIR
    Rdirvis[i] = (Sco_vis[i] * np.exp(-.185 * (press[i] / 1313.25) * airmas[i])- w[i]) * coszen[i]  # Modified Eq1 assuming water vapor absorption
    # Rdirvis=(Sco_vis*exp(-.185*(press/1313.25)*airmas))*coszen
    # #Eq. 1
    Rdirvis = np.maximum(0, Rdirvis)
    # Potential diffuse radiation
    # Eq 3                                      #Eq. 3
    Rdifvis[i] = 0.4 * (Sco_vis[i] * coszen[i] - Rdirvis[i])
    Rdifvis = np.maximum(0, Rdifvis)

    # Same for NIR
    # w=SOLAR_CONSTANT*(1.0-T)
    w = SOLAR_CONSTANT * \
        10**(-1.195 + .4459 * np.log10(coszen[i]) - .0345 * np.log10(coszen[i])**2)  # Eq. .6
    Rdirnir[i] = (Sco_nir[i] * np.exp(-0.06 * (press[i] / 1313.25)
                                   * airmas[i]) - w) * coszen[i]  # Eq. 4
    Rdirnir = np.maximum(0, Rdirnir)

    # Potential diffuse radiation
    Rdifnir[i] = 0.6 * (Sco_nir[i] * coszen[i] - Rdirvis[i] - w)  # Eq. 5
    Rdifnir = np.maximum(0, Rdifnir)
    Rdirvis, Rdifvis, Rdirnir, Rdifnir = map(
        np.asarray, (Rdirvis, Rdifvis, Rdirnir, Rdifnir))
    return Rdirvis, Rdifvis, Rdirnir, Rdifnir

def calc_emiss_atm(ea, t_a_k):
    '''Atmospheric emissivity

    Estimates the effective atmospheric emissivity for clear sky.

    Parameters
    ----------
    ea : float
        atmospheric vapour pressure (mb).
    t_a_k : float
        air temperature (Kelvin).

    Returns
    -------
    emiss_air : float
        effective atmospheric emissivity.

    References
    ----------
    .. [Brutsaert1975] Brutsaert, W. (1975) On a derivable formula for long-wave radiation
        from clear skies, Water Resour. Res., 11(5), 742-744,
        htpp://dx.doi.org/10.1029/WR011i005p00742.'''

    emiss_air = 1.24 * (ea / t_a_k)**(1. / 7.)  # Eq. 11 in [Brutsaert1975]_

    return np.asarray(emiss_air)

def calc_K_be_Campbell(theta, x_lad=1, radians=False):
    ''' Beam extinction coefficient

    Calculates the beam extinction coefficient based on [Campbell1998]_ ellipsoidal
    leaf inclination distribution function.

    Parameters
    ----------
    theta : float
        incidence zenith angle.
    x_lad : float, optional
        Chi parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_lad=1 for a spherical LAD.
    radians : bool, optional
        Should be True if theta is in radians.
        Default is False.

    Returns
    -------
    K_be : float
        beam extinction coefficient.
    x_lad: float, optional
        x parameter for the ellipsoidal Leaf Angle Distribution function,
        use x_lad=1 for a spherical LAD.

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    '''

    if not radians:
        theta = np.radians(theta)

    K_be = (np.sqrt(x_lad**2 + np.tan(theta)**2)
            / (x_lad + 1.774 * (x_lad + 1.182)**-0.733))

    return K_be

# def calc_L_n_Kustas(T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1):
#     ''' Net longwave radiation for soil and canopy layers
#
#     Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
#     from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on
#     [Campbell1998]_
#
#     Parameters
#     ----------
#     T_C : float
#         Canopy temperature (K).
#     T_S : float
#         Soil temperature (K).
#     L_dn : float
#         Downwelling atmospheric longwave radiation (w m-2).
#     lai : float
#         Effective Leaf (Plant) Area Index.
#     emisVeg : float
#         Broadband emissivity of vegetation cover.
#     emisGrd : float
#         Broadband emissivity of soil.
#     x_lad: float, optional
#         x parameter for the ellipsoidal Leaf Angle Distribution function,
#         use x_lad=1 for a spherical LAD.
#
#     Returns
#     -------
#     L_nC : float
#         Net longwave radiation of canopy (W m-2).
#     L_nS : float
#         Net longwave radiation of soil (W m-2).
#
#     References
#     ----------
#     .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
#         flux predictions using a simple two-source model with radiometric temperatures for
#         partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
#         Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
#     '''
#
#     # Get the diffuse transmitance
#     _, _, _, taudl = calc_spectra_Cambpell(lai,
#                                           np.zeros(emisVeg.shape),
#                                           1.0 - emisVeg,
#                                           np.zeros(emisVeg.shape),
#                                           1.0 - emisGrd,
#                                           x_lad=x_LAD,
#                                           lai_eff=None)
#
#     # calculate long wave emissions from canopy, soil and sky
#     L_C = emisVeg * met.calc_stephan_boltzmann(T_C)
#     L_S = emisGrd * met.calc_stephan_boltzmann(T_S)
#
#     # calculate net longwave radiation divergence of the soil
#     L_nS = taudl * L_dn + (1.0 - taudl) * L_C - L_S
#     L_nC = (1.0 - taudl) * (L_dn + L_S - 2.0 * L_C)
#     return np.asarray(L_nC), np.asarray(L_nS)
#
# def calc_L_n_Campbell(T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD=1):
#     ''' Net longwave radiation for soil and canopy layers
#
#     Estimates the net longwave radiation for soil and canopy layers unisg based on equation 2a
#     from [Kustas1999]_ and incorporated the effect of the Leaf Angle Distribution based on [Campbell1998]_
#
#     Parameters
#     ----------
#     T_C : float
#         Canopy temperature (K).
#     T_S : float
#         Soil temperature (K).
#     L_dn : float
#         Downwelling atmospheric longwave radiation (w m-2).
#     lai : float
#         Effective Leaf (Plant) Area Index.
#     emisVeg : float
#         Broadband emissivity of vegetation cover.
#     emisGrd : float
#         Broadband emissivity of soil.
#     x_LAD: float, optional
#         x parameter for the ellipsoidal Leaf Angle Distribution function,
#         use x_LAD=1 for a spherical LAD.
#
#     Returns
#     -------
#     L_nC : float
#         Net longwave radiation of canopy (W m-2).
#     L_nS : float
#         Net longwave radiation of soil (W m-2).
#
#     References
#     ----------
#     .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
#         flux predictions using a simple two-source model with radiometric temperatures for
#         partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
#         Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
#     '''
#     T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD = map(np.asarray, (T_C, T_S, L_dn, lai, emisVeg, emisGrd, x_LAD))
#
#     # calculate long wave emissions from canopy, soil and sky
#     L_C = emisVeg * met.calc_stephan_boltzmann(T_C)
#     L_C[np.isnan(L_C)] = 0
#     L_S = emisGrd * met.calc_stephan_boltzmann(T_S)
#     L_S[np.isnan(L_S)] = 0
#     # Calculate the canopy spectral properties
#     _, albl, _, taudl = calc_spectra_Cambpell(lai,
#                                               np.zeros(emisVeg.shape),
#                                               1.0 - emisVeg,
#                                               np.zeros(emisVeg.shape),
#                                               1.0 - emisGrd,
#                                               x_lad=x_LAD,
#                                               lai_eff=None)
#
#     # calculate net longwave radiation divergence of the soil
#     L_nS = emisGrd * taudl * L_dn + emisGrd * (1.0 - taudl) * L_C - L_S
#     L_nC = (1 - albl) * (1.0 - taudl) * (L_dn + L_S) - 2.0 * (1.0 - taudl) * L_C
#     L_nC[np.isnan(L_nC)] = 0
#     L_nS[np.isnan(L_nS)] = 0
#     return np.asarray(L_nC), np.asarray(L_nS)


def calc_taud(x_lad, lai):
    TAUD_STEP_SIZE_DEG = 5
    taud = 0
    for angle in range(0, 90, TAUD_STEP_SIZE_DEG):
        angle = np.radians(angle)
        akd = calc_K_be_Campbell(angle, x_lad, radians=True)
        taub = np.exp(-akd * lai)
        taud += taub * np.cos(angle) * np.sin(angle) * np.radians(TAUD_STEP_SIZE_DEG)
    return 2.0 * taud

def calc_spectra_Cambpell(lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad=1, lai_eff=None):
    """ Canopy spectra

    Estimate canopy spectral using the [Campbell1998]_
    Radiative Transfer Model

    Parameters
    ----------
    lai : float
        Effective Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    rho_leaf : float, or array_like
        Leaf bihemispherical reflectance
    tau_leaf : float, or array_like
        Leaf bihemispherical transmittance
    rho_soil : float
        Soil bihemispherical reflectance
    x_lad : float,  optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    lai_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    albb : float or array_like
        Beam (black sky) canopy albedo
    albd : float or array_like
        Diffuse (white sky) canopy albedo
    taubt : float or array_like
        Beam (black sky) canopy transmittance
    taudt : float or array_like
        Beam (white sky) canopy transmittance

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    """
    lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad = map(np.asarray, (lai, sza, rho_leaf, tau_leaf, rho_soil, x_lad))

    # calculate aborprtivity
    amean = 1.0 - rho_leaf - tau_leaf
    amean_sqrt = np.sqrt(amean)
    del rho_leaf, tau_leaf, amean

    # Calculate canopy beam extinction coefficient
    # Modification to include other LADs
    if lai_eff is None:
        lai_eff = np.asarray(lai)
    else:
        lai_eff = np.asarray(lai_eff)

    # D I F F U S E   C O M P O N E N T S
    # Integrate to get the diffuse transmitance
    taud = calc_taud(x_lad, lai)

    # Diffuse light canopy reflection coefficients  for a deep canopy
    akd = -np.log(taud) / lai
    rcpy= (1.0 - amean_sqrt) / (1.0 + amean_sqrt)  # Eq 15.7
    rdcpy = 2.0 * akd * rcpy / (akd + 1.0)  # Eq 15.8

    # Diffuse canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akd * lai
    del akd
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    xnum = (rdcpy * rdcpy - 1.0) * neg_exp
    xden = (rdcpy * rho_soil - 1.0) + rdcpy * (rdcpy - rho_soil) * d_neg_exp
    taudt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rdcpy - rho_soil) / (rdcpy * rho_soil - 1.0)) * d_neg_exp
    albd = (rdcpy + fact) / (1.0 + rdcpy * fact)  # Eq 15.9
    del rdcpy, fact

    # B E A M   C O M P O N E N T S
    # Direct beam extinction coeff (spher. LAD)
    akb = calc_K_be_Campbell(sza, x_lad)  # Eq. 15.4

    # Direct beam canopy reflection coefficients for a deep canopy
    rbcpy = 2.0 * akb * rcpy / (akb + 1.0)  # Eq 15.8
    del rcpy, sza, x_lad
    # Beam canopy transmission and albedo coeff for a generic canopy (visible)
    expfac = amean_sqrt * akb * lai_eff
    neg_exp, d_neg_exp = np.exp(-expfac), np.exp(-2.0 * expfac)
    del amean_sqrt, akb, lai_eff
    xnum = (rbcpy * rbcpy - 1.0) * neg_exp
    xden = (rbcpy * rho_soil - 1.0) + rbcpy * (rbcpy - rho_soil) * d_neg_exp
    taubt = xnum / xden  # Eq 15.11
    del xnum, xden
    fact = ((rbcpy - rho_soil) / (rbcpy * rho_soil - 1.0)) * d_neg_exp
    del expfac
    albb = (rbcpy + fact) / (1.0 + rbcpy * fact)  # Eq 15.9
    del rbcpy, fact

    taubt[np.isnan(taubt)] = 1
    taudt[np.isnan(taudt)] = 1
    albb[np.isnan(albb)] = rho_soil[np.isnan(albb)]
    albd[np.isnan(albd)] = rho_soil[np.isnan(albd)]


    return albb, albd, taubt, taudt

def calc_Sn_Campbell(lai, sza, S_dn_dir, S_dn_dif, fvis, fnir, rho_leaf_vis,
                     tau_leaf_vis, rho_leaf_nir, tau_leaf_nir, rsoilv, rsoiln,
                     x_LAD=1, LAI_eff=None):
    ''' Net shortwave radiation

    Estimate net shorwave radiation for soil and canopy below a canopy using the [Campbell1998]_
    Radiative Transfer Model, and implemented in [Kustas1999]_

    Parameters
    ----------
    lai : float
        Effecive Leaf (Plant) Area Index.
    sza : float
        Sun Zenith Angle (degrees).
    S_dn_dir : float
        Broadband incoming beam shortwave radiation (W m-2).
    S_dn_dif : float
        Broadband incoming diffuse shortwave radiation (W m-2).
    fvis : float
        fration of total visible radiation.
    fnir : float
        fraction of total NIR radiation.
    rho_leaf_vis : float
        Broadband leaf bihemispherical reflectance in the visible region (400-700nm).
    tau_leaf_vis : float
        Broadband leaf bihemispherical transmittance in the visible region (400-700nm).
    rho_leaf_nir : float
        Broadband leaf bihemispherical reflectance in the NIR region (700-2500nm).
    tau_leaf_nir : float
        Broadband leaf bihemispherical transmittance in the NIR region (700-2500nm).
    rsoilv : float
        Broadband soil bihemispherical reflectance in the visible region (400-700nm).
    rsoiln : float
        Broadband soil bihemispherical reflectance in the NIR region (700-2500nm).
    x_lad : float, optional
        x parameter for the ellipsoildal Leaf Angle Distribution function of
        Campbell 1988 [default=1, spherical LIDF].
    LAI_eff : float or None, optional
        if set, its value is the directional effective LAI
        to be used in the beam radiation, if set to None we assume homogeneous canopies.

    Returns
    -------
    Sn_C : float
        Canopy net shortwave radiation (W m-2).
    Sn_S : float
        Soil net shortwave radiation (W m-2).

    References
    ----------
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
    .. [Kustas1999] Kustas and Norman (1999) Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    rho_leaf = np.array((rho_leaf_vis, rho_leaf_nir))
    tau_leaf = np.array((tau_leaf_vis, tau_leaf_nir))
    rho_soil = np.array((rsoilv, rsoiln))
    albb, albd, taubt, taudt = calc_spectra_Cambpell(lai,
                                                     sza,
                                                     rho_leaf,
                                                     tau_leaf,
                                                     rho_soil,
                                                     x_lad=x_LAD,
                                                     lai_eff=LAI_eff)

    Sn_C = ((1.0 - taubt[0]) * (1.0- albb[0]) * S_dn_dir*fvis
            + (1.0 - taubt[1]) * (1.0- albb[1]) * S_dn_dir*fnir
            + (1.0 - taudt[0]) * (1.0- albd[0]) * S_dn_dif*fvis
            + (1.0 - taudt[1]) * (1.0- albd[1]) * S_dn_dif*fnir)
            
    Sn_S = (taubt[0] * (1.0 - rsoilv) * S_dn_dir*fvis
            + taubt[1] * (1.0 - rsoiln) * S_dn_dir*fnir
            + taudt[0] * (1.0 - rsoilv) * S_dn_dif*fvis
            + taudt[1] * (1.0 - rsoiln) * S_dn_dif*fnir)
    
    return np.asarray(Sn_C), np.asarray(Sn_S)

def leafangle_2_chi(alpha):
    alpha = np.radians(alpha)
    x_lad = ((alpha / 9.65) ** (1. / -1.65)) - 3.
    return x_lad

def chi_2_leafangle(x_lad):
    alpha = 9.65 * (3. + x_lad) ** -1.65
    alpha = np.degrees(alpha)
    return alpha

def calc_omega0_Kustas(LAI, f_C, x_LAD=1, isLAIeff=True):
    ''' Nadir viewing clmping factor

    Estimates the clumping factor forcing equal gap fraction between the real canopy
    and the homogeneous case, after [Kustas1999]_.

    Parameters
    ----------
    LAI : float
        Leaf Area Index, it can be either the effective LAI or the real LAI
        , default input LAI is effective.
    f_C : float
        Apparent fractional cover, estimated from large gaps, means that
        are still gaps within the canopy to be quantified.
    x_LAD : float, optional
        Chi parameter for the ellipsoildal Leaf Angle Distribution function of
        [Campbell1988]_ [default=1, spherical LIDF].
    isLAIeff :  bool, optional
        Defines whether the input LAI is effective or local.

    Returns
    -------
    omega0 : float
        clumping index at nadir.

    References
    ----------
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    .. [Campbell1998] Campbell, G. S. & Norman, J. M. (1998), An introduction to environmental
        biophysics. Springer, New York
        https://archive.org/details/AnIntroductionToEnvironmentalBiophysics.
 '''

    # Convert input scalars to numpy array
    LAI, f_C, x_LAD = map(np.asarray, (LAI, f_C, x_LAD))
    theta = np.zeros(LAI.shape)
    # Estimate the beam extinction coefficient based on a ellipsoidal LAD function
    # Eq. 15.4 of Campbell and Norman (1998)
    K_be = np.sqrt(x_LAD**2 + np.tan(theta)**2) / \
        (x_LAD + 1.774 * (x_LAD + 1.182)**-0.733)
    if isLAIeff:
        F = LAI / f_C
    else:  # The input LAI is actually the real LAI
        F = np.array(LAI)
    # Calculate the gap fraction of our canopy
    trans = np.asarray(f_C * np.exp(-K_be * F) + (1.0 - f_C))
    trans[trans <= 0] = 1e-36
    # and then the nadir clumping factor
    omega0 = -np.log(trans) / (F * K_be)
    return omega0

def calc_omega_Kustas(omega0, theta, w_C=1):
    ''' Clumping index at an incidence angle.

    Estimates the clumping index for a given incidence angle assuming randomnly placed canopies.

    Parameters
    ----------
    omega0 : float
        clumping index at nadir, estimated for instance by :func:`calc_omega0_Kustas`.
    theta : float
        incidence angle (degrees).
    w_C :  float, optional
        canopy witdth to height ratio, [default = 1].

    Returns
    -------
    Omega : float
        Clumping index at an incidenc angle.

    References
    ----------
    .. [Kustas1999] William P Kustas, John M Norman, Evaluation of soil and vegetation heat
        flux predictions using a simple two-source model with radiometric temperatures for
        partial canopy cover, Agricultural and Forest Meteorology, Volume 94, Issue 1,
        Pages 13-29, http://dx.doi.org/10.1016/S0168-1923(99)00005-2.
    '''

    w_C = 1.0 / w_C
    omega = omega0 / (omega0 + (1.0 - omega0) *
                      np.exp(-2.2 * (np.radians(theta))**(3.8 - 0.46 * w_C)))
    return omega

