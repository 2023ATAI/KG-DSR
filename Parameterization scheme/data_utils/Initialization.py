# This file is part PySiLSMV3, consisting of of high level pySiLSMV3 scripting
# Copyright 2022 Zhongwang Wei@SYSU, zhongwang007@gmail.com 
#
import numpy as np
import xarray as xr
import sys
from . import metlib
from . import energy as ene
from .timelib import get_local_etc_timezone,calc_sun_angles
import pandas as pd
from . import resistance as res
# Stephan Boltzmann constant (W m-2 K-4)
sb = 5.670373e-8

class basic_vars:
    def __init__(self,df):
        self.name        = 'Basic variables'
        self.version     = '0.1'
        self.release     = '0.1'
        self.date        = 'May 2023'
        self.author      = "Zhongwang Wei"
        self.affiliation = "Sun Yat-sen University"
        self.description = "This is basic variables for pySiLSMV3"    
        print("---------------------------------------------------------------------------------------")
        print("basic data preprocessing") 
        try:
            self.z_T           =      np.full_like(df['LAI'],df['reference_height'].values)  
            self.z_U           =      np.full_like(df['LAI'],df['reference_height'].values)  
            self.elevation     =      np.full_like(df['LAI'].values,df['elevation'].values)
            self.dt            =      df['dt'].values
            self.IGBP_CLASS    =      np.full_like(df['LAI'].values,df['IGBP_CLASS'].values) 
            self.Year          =      df["time.year"].values
            self.Month         =      df["time.month"].values
            self.Day           =      df["time.day"].values
            self.Hour          =      df["time.hour"].values
            self.Doy           =      df["time.dayofyear"].values
            self.lon           =      df["lon"].values
            self.lat           =      df["lat"].values
            self.time          =      df["time"]
        except:
            print('missing basic attributes in inputdata, please check your inputdata')
            sys.exit(1)

class met_vars:
    def __init__(self,df):
        self.name        = 'Meteorological variables'
        self.version     = '0.1'
        self.release     = '0.1'
        self.date        = 'May 2023'
        self.author      = "Zhongwang Wei"
        self.affiliation = "Sun Yat-sen University"
        self.description = "This is meteorological variables for pySiLSMV3"

        if (df['Tair'].mean()>100.):
            self.Tair_C    =     df['Tair'].values-273.15  #K
            self.Tair_K    =     df['Tair'].values         #C
        else:
            self.Tair_C    =     df['Tair'].values        #C
            self.Tair_K    =     df['Tair'].values+273.15  #K
            
        self.Wind          =     df['Wind'].values        # wind speed (m/s)
        self.Precip        =     df['Precip'].values      # precipitation (mm/s)??
        self.Qair          =     df['Qair'].values  #kg/kg
        try:
            self.Psurf     =     df['Psurf'].values        #kpa
        except:
            print('missing air pressure variables in inputdata, 101.325 kpa is used as default')
            self.Psurf     =      101.325           #kpa
        
        try:
            self.RH        =     df['RH'].values
            if (df['RH'].mean()>1.):
                self.RH        =     self.RH/100.         #relative humidity  ---> to fraction
        except:
            print('missing relative humidity variables in inputdata')
            print("try to calculate it from specific humidity, air temperature and air pressure")
            try:
                self.RH                  =    metlib.qair2rh(self.Qair,self.Tair_C, self.Psurf*10.0)
            except:
                print('missing both relative humidity and specific humidity variables in inputdata, at lease one of them should be provided, please check your inputdata (RH,Qair)')
                sys.exit(1)
               #get VPD
        #try:
        #    self.VPD                      =    df['VPD'].values #Kpa
        #except:
        #    print('missing VPD variables in inputdata, try to calculate it from relative humidity, air temperature and air pressure')
        self.VPD                      =    metlib.calc_vpd(self.RH,self.Tair_C) #Kpa
        
        #Calculate the (saturation) water vapour pressure
        self.svp,self.vp                  =    metlib.calc_vapor_pressure(self.Tair_K,self.RH) #kpa     

        #get other meteorological variables
        self.Cp                           =    metlib.calc_c_p(self.Psurf*10.0,self.vp*10.0)  #  Calculates the heat capacity of air at constant pressure (J kg-1 K-1).
        self.rho                          =    metlib.calc_rho(self.Psurf*10.0,self.vp*10.0,self.Tair_K)  #  Calculates the density of air (kg m-3).
        self.delta                        =    metlib.calc_delta_vapor_pressure(self.Tair_K) #slope of the saturation water vapour pressure (kPa K-1)
        self.lhv                          =    metlib.calc_lambda(self.Tair_K)   # Latent heat of vaporisation (J kg-1).
        self.Psy                          =    metlib.calc_psicr(self.Cp,self.Psurf*10.0,self.lhv)/10.0 #  Psychrometric constant (kPa K-1).

        #get fraction of wet and dry canopy
        self.fwet                         =    metlib.dewfraction_2(self.RH) #fraction of wet canopy

        try:
            self.Ustar                    =    df['Ustar'].values #friction velocity (m/s)
            print(self.Ustar)
        except:
            print('missing friction velocity variables in inputdata, try to calculate it from wind speed and air temperature')
        try:
            self.ObukhovLength             =    df['ObukhovLength'].values #friction velocity (m/s)
        except:
            self.ObukhovLength             =  np.full_like( df['Tair'].values,np.inf)

        Zc                                 =    np.full_like(df['Tair'].values,df['canopy_height'].values)

        self.cd, self.z0c                  =    res.calc_cd_Shuttleworth1990(Zc)
        self.d0                            =    res.calc_d_0_Shuttleworth1990(Zc,self.z0c,self.cd,df['LAI'].values)
        self.Km                            =    res.calc_Km(Zc)
        
        #TODO:need to be checked
        self.z0_soil                       =    np.full_like( df['Tair'].values,0.01) 

        self.z0m                           =    res.calc_z_0m_Shuttleworth1990(df['LAI'].values,Zc,self.z0_soil ,self.cd,self.d0)
        self.z0h                           =    self.z0m*0.1
        self.beta                          =    np.full_like(self.Tair_K, -9999.0)
        self.rac                           =    np.full_like(self.Tair_K, -9999.0)
        self.raa                           =    np.full_like(self.Tair_K, -9999.0)
        self.ras                           =    np.full_like(self.Tair_K, -9999.0)
        self.rsc                           =    np.full_like(self.Tair_K, -9999.0)
        self.rss                           =    np.full_like(self.Tair_K, -9999.0)
        self.rav                           =    np.full_like(self.Tair_K, -9999.0)
        self.R_d                           =    np.full_like(self.Tair_K, -9999.0)
        self.rss_v                         =    np.full_like(self.Tair_K, 1.0)

        #TODO:need to check the unit of CO2air
        try:
            self.CO2air       =     df['CO2air'].values*1.8   #CO2 concentration in air (ppm)-->umol/mol  ???
        except:
            print('missing CO2 variables in inputdata, 400 ppm is used as default')
            self.CO2air       =     np.full_like(self.df['Tair'].values, 400.0*1.8)

        print('Meteorological variables are ready')

class soil_vars:
    def __init__(self,df):
        self.name        = 'Soil variables'
        self.version     = '0.1'
        self.release     = '0.1'
        self.date        = 'May 2023'
        self.author      = "Zhongwang Wei"
        self.affiliation = "Sun Yat-sen University"
        self.description = "This is soil variables for pySiLSMV3"    
        print("---------------------------------------------------------------------------------------")
        print("Soil data preprocessing") 
        try:
            self.swc_surf    =     df['swc_surf'].values
            self.swc_root    =     df['swc_root'].values
        except:
            print('missing soil moisture variables in inputdata, 0.5 is used as default')
            sys.exit(1)
        
        try:
            self.WP                   =      np.full_like(df['Tair'].values,df['wiltingpoint_mean'].values)
            self.Ks_mean              =      np.full_like(df['Tair'].values,df['Ks_mean'].values)
            self.sfc_mean             =      np.full_like(df['Tair'].values,df['sfc_mean'].values)
            self.sh_mean              =      np.full_like(df['Tair'].values,df['sh_mean'].values)
            self.n_mean               =      np.full_like(df['Tair'].values,df['n_mean'].values)
            self.FC                   =      self.sfc_mean
        except:
            print('missing soil information variables in inputdata, please check your inputdata (wiltingpoint_mean,FC,Ks_mean,sfc_mean,sh_mean,n_mean)')
        
        print('')
        print('')
        print('Soil variables are ready')
        print("---------------------------------------------------------------------------------------")

class veg_vars:
    def __init__(self,df,bas,met):
        self.name         = 'Vegetable variables'
        self.version      = '0.1'
        self.release      = '0.1'
        self.date         = 'May 2023'
        self.author       = "Zhongwang Wei"
        self.affiliation  = "Sun Yat-sen University"
        self.description  = "This is vegetable variables for pySiLSMV3"    
        print("---------------------------------------------------------------------------------------")
        print("vegetation data preprocessing") 
        self.LAI          =     df['LAI'].values
        self.Kr           =     np.full_like(self.LAI , df['Kr'].values)  
        self.dl           =     np.full_like(self.LAI, df['dl'].values)  
        self.rsc          =     np.full_like(self.LAI, -9999.0)
        self.Tc_K         =     df['Tair'].values
        self.D_s          =     np.full_like(self.LAI, -9999.0)
        self.C3           =     np.full_like(self.LAI, df['C3'].values)   
        self.D0           =     np.full_like(self.LAI, df['D0'].values)   
        self.g_min_c      =     np.full_like(self.LAI, df['g_min_c'].values) 
        self.Zc           =    np.full_like(df['Tair'].values,df['canopy_height'].values)

        self.leaf_width   =     np.full_like(self.LAI, 0.1    )   
        #mesophyll conductance
        ###C3
        self.g_m_298      =     np.full_like(self.LAI, 7.0    )   
        self.Q_10_g_m     =     np.full_like(self.LAI, 2.0    )
        self.T_1_g_m      =     np.full_like(self.LAI, 278.0  )
        self.T_2_g_m      =     np.full_like(self.LAI, 301.0  )
        ###C4
        self.g_m_298      =     np.where(self.C3< 1.0, 17.5,  self.g_m_298 )  
        self.Q_10_g_m     =     np.where(self.C3< 1.0, 2.0,   self.Q_10_g_m )
        self.T_1_g_m      =     np.where(self.C3< 1.0, 286.0, self.T_1_g_m  )
        self.T_2_g_m      =     np.where(self.C3< 1.0, 309.0, self.T_2_g_m  )

        #CO2 compensation point
        #TODO:Broadcast the 2D array to the same shape as the 3D array ---> ds_2d_expanded = ds_2d.expand_dims(dim='time'); ds_2d_broadcast = ds_2d_expanded.broadcast_like(ds_3d)

        self.Gamma_298    =     np.full_like(self.LAI,  68.5*1.23) # CO2 compensation point at 298 degK (68.5*rho_a mg/m3; air density in kg/m3)
        self.Gamma_298    =     np.where(self.C3< 1.0, 4.3*1.23, self.Gamma_298)
        
        # light use efficiency
        self.alpha_0      =     np.full_like(self.LAI, 0.017) #for C3
        self.alpha_0      =     np.where(self.C3 < 1.0, 0.014,self.alpha_0) #for C4

        #primary productivity
        self.A_m_max_298  =     np.full_like(self.LAI, 2.2) #for C3
        self.A_m_max_298  =     np.where(self.C3< 1.0, 1.7, self.A_m_max_298) #for C4

        self.Q_10_A_m     =     np.full_like(self.LAI, 2.0) #for C3
        self.Q_10_A_m     =     np.where(self.C3< 1.0, 2.0, self.Q_10_A_m) #for C4

        self.T_1_A_m      =     np.full_like(self.LAI, 281.0) #for C3
        self.T_1_A_m      =     np.where(self.C3< 1.0, 286.0, self.T_1_A_m) #for C4

        self.T_2_A_m      =     np.full_like(self.LAI, 311.0) #for C3
        self.T_2_A_m      =     np.where(self.C3< 1.0, 311.0, self.T_2_A_m) #for C4

        self.f0           =     np.full_like(self.LAI, 0.89) #for C3
        self.f0           =     np.where(self.C3< 1.0, 0.85, self.f0) #for C4
        
        self.a_1          =     1.0 / (1.0 - self.f0) #for C3

        self.A_g          =     np.full_like(self.LAI, -9999.0)
        self.A_n          =     np.full_like(self.LAI, -9999.0)
        self.A_m          =     np.full_like(self.LAI, -9999.0)
        self.g_m          =     np.full_like(self.LAI, -9999.0)
        self.g_cc         =     np.full_like(self.LAI, -9999.0)
        self.g_cw         =     np.full_like(self.LAI, -9999.0)
        self.Gamma        =     np.full_like(self.LAI, -9999.0)
        self.d1           =     np.full_like(self.LAI, -9999.0)
        self.d2           =     np.full_like(self.LAI, -9999.0)
        #1-14:Houldcroft, Caroline J., et al. "New vegetation albedo parameters and global fields of
        # soil background albedo derived from MODIS for use in a climate model."
        #  Journal of Hydrometeorology 10.1 (2009): 183-198.
        self.albedo       =     np.full_like(self.LAI,bas.IGBP_CLASS)
        print (self.albedo)

        print (self.albedo.shape)
        IGBP_albedo = {
        0: 0.08,   # water
        1: 0.092,  # evergreen needleleaf forest
        2: 0.139,  # evergreen broadleaf forest
        3: 0.103,  # deciduous needleleaf forest
        4: 0.133,  # deciduous broadleaf forest
        5: 0.112,  # mixed forests
        6: 0.134,   # closed shrubland
        7: 0.161,   # open shrublands
        8: 0.131,   # woody savannas
        9: 0.155,   # savannas
        10: 0.168,  # grasslands 
        11: 0.102,  # permanent wetlands
        12: 0.165,  # croplands
        13: 0.149,  # urban and built-up
        14: 0.158,  # cropland natural vegetation mosaic
        15: 0.8,    # snow and ice
        16: 0.35    # barren or sparsely vegetated
        }
        map_albedo = np.vectorize(lambda x: IGBP_albedo[x])
        self.albedo = map_albedo(self.albedo)

        self.f_c =  1.0-np.exp(-(self.Kr)*df['LAI'])    #canopy cover fraction


        self.C_s              =     met.CO2air
        self.C_i              =     met.CO2air
        self.C_i_min          =     np.full_like(self.LAI, -9999.0)
        #self.albedo = xr.apply_ufunc(lambda x: IGBP_albedo[x], self.albedo)


        self.lmda       =     np.full_like(self.LAI,bas.IGBP_CLASS)
        IGBP_lmda = {
        0: 1500.,   # water
        1: 1500.,  # evergreen needleleaf forest
        2: 1500.,  # evergreen broadleaf forest
        3: 1500.,  # deciduous needleleaf forest
        4: 1500.,  # deciduous broadleaf forest
        5: 1500.,  # mixed forests
        6: 1500.,   # closed shrubland
        7: 1500.,   # open shrublands
        8: 1500.,   # woody savannas
        9: 1500.,   # savannas
        10: 1500.,  # grasslands 
        11: 1500.,  # permanent wetlands
        12: 1500.,  # croplands
        13: 1500.,  # urban and built-up
        14: 1500.,  # cropland natural vegetation mosaic
        15: 1500.,    # snow and ice
        16: 1500.    # barren or sparsely vegetated
        }
        map_lmda = np.vectorize(lambda x: IGBP_lmda[x])
        self.lmda = map_lmda(self.lmda)
        print('')
        print('')
        print('vegetation variables are ready')
        print("---------------------------------------------------------------------------------------")

class energy_vars:
    def __init__(self,df,met,bas,veg):
        self.name         = 'Co2 related variables'
        self.version      = '0.1'
        self.release      = '0.1'
        self.date         = 'May 2023'
        self.author       = "Zhongwang Wei"
        self.affiliation  = "Sun Yat-sen University"
        self.description  = "This is Energy related variables for pySiLSMV3"   
        
        try:
            self.SWdown       = df['SWdown'].values
        except:
            print('missing necessary Meteorological variables in inputdata, please check your SWdown inputdata')
            sys.exit(1)
        
        try:
            self.LWdown    = df['LWdown'].values
        except:
            # Incoming long wave radiation
            # If longwave irradiance was not provided then estimate it based on air temperature and humidity
            print('missing necessary Meteorological variables in inputdata, please check your LWdown inputdata')
            print('try to calculate it from air temperature and humidity')
            self.LWdown    =  ene.calc_longwave_irradiance(met.vp*10.0, met.Tair_K,met.Psurf*10.0, bas.z_T)
        
        try:
            self.SZA       = df['SZA'].values
        except:
            #TODO: need to check the calculation of SZA
            print('missing necessary Meteorological variables in inputdata, please check your SZA inputdata')
            print('try to calculate it from lat, lon, UTC Doy and hour')
            self.SZA      =  calc_sun_angles(bas.lat, bas.lon, 0.0, bas.Doy,  bas.Hour)
            print('')
            print('')
            print('SZA are ready')
            print("---------------------------------------------------------------------------------------")

        #get net radiation
        # if nl['ModSet']['net_radiation_formula'] == 'observation':
        self.Rnet = df['Rnet'].values
        try:
            self.LWup = df['LWup'].values
            self.SWup = df['SWup'].values
        except:
            self.SWup = (1.0 - veg.albedo) * self.SWdown
            self.LWup = 0.9 * (self.LWdown - sb * met.Tair_K ** 4)



        # elif nl['ModSet']['net_radiation_formula'] == 'FAO_refcrop':
        #     self.Rnet             =    ene.calc_net_radiation_FAO_refcrop(self.SWdown, self.LWdown,met.Tair_K,veg.albedo)
        # elif nl['ModSet']['net_radiation_formula'] == 'FAO':
        #     #TODO:bug here!!!!
        #     #time_index             =    pd.to_datetime(df['time'].values)
        #     lat1D                  =    bas.lat
        #     lon1D                  =    bas.lon
        #     #TODO:check the order of time, lat and lon
        #     # Broadcast the 1D coordinates to 3D
        #     #lon_3d, lat_3d, time_3d = np.broadcast_arrays(lon1D[:, np.newaxis, np.newaxis],
        #     #                                              lat1D[np.newaxis, :, np.newaxis],
        #     #                                              time_index[np.newaxis, np.newaxis, :])
        #     #time3d=time_index.strftime('%Y-%m-%dT%H:%MZ')
        #     self.Rnet             =    ene.calc_net_radiation_FAO(bas.Doy, bas.Hour,met.Tair_C, met.RH*100.,
        #                                                           self.SWdown,lat1D,lon1D,bas.elevation,veg.albedo)
        #
        # elif nl['ModSet']['net_radiation_formula'] == 'TESB_SW':
        #     self.Rnet              =   ene.calc_net_radiation_TESB_SW(self.LWdown,self.SWdown,self.SZA,met.Psurf*10.,
        #                                                               met.Tair_K,veg.LAI,veg.f_c,bas.IGBP_CLASS,
        #                                                               met.z0_soil,met.z0m,met.Zc, met.d0, x_LAD = 1.0)
        # get ground heat flux
        try:
            self.Qg             =     df['Qg'].values
        except:
            print('missing Qg variables in inputdata, try to initial it from Rnet')
            self.Qg             =    ene.calc_G_Rn_ratio(self.Rnet)



        # get total available energy [w/m2]
        self.A                  =     self.Rnet- self.Qg

        # get Radiation reaching the soil surface  [w/m2]
        #todo: Kr can be calculated in a more accurate way
        self.Rns                =     self.Rnet*np.exp(-(veg.Kr)*veg.LAI)

        self.Rnc                =     self.Rnet - self.Rns

        #available energy for the soil surface  [w/m2]
        self.As                 =     self.Rns - self.Qg

        #; % shortwave radiation in PAR waveband, [W/m2]
        #todo: relationship between par and  beam/diffuse shortwave radiation
        #todo: check the calculation of PAR
        self.PAR_t              =    self.SWdown * 0.45
        self.LEs                =    np.full_like(df['Tair'], -9999.0)#soil latent heat flux
        self.LEc                =    np.full_like(df['Tair'], -9999.0) #canopy latent heat flux
        self.LEi                =    np.full_like(df['Tair'], -9999.0) #canopy latent heat flux

        self.LE                 =    np.full_like(df['Tair'], -9999.0)#total latent heat flux
        self.HEs                =    np.full_like(df['Tair'], -9999.0)#soil sensible heat flux
        self.HEc                =    np.full_like(df['Tair'], -9999.0)#canopy sensible heat flux
        self.HE                 =    np.full_like(df['Tair'], -9999.0)#total sensible heat flux
        self.Qle_cor            =    df['Qle_cor'].values






