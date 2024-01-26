import random

import numpy as np
import pandas as pd
import xarray as xr

from .Initialization import met_vars,soil_vars,veg_vars,basic_vars,energy_vars
from .resistance import calc_R_A_Norman1995
class canopy_conductance_Jarvis1976:
    def __init__(self, Swin, wfc, w2, wwilt, gD, VPD, rsmin, theta_T, LAI):
        self.name = 'canopy_conductance_Liang2022'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        self.Swin = Swin
        self.wfc = wfc
        self.w2 = w2
        self.wwilt = wwilt
        self.gD = gD
        self.VPD = VPD
        self.rsmin = rsmin  # minimum resistance transpiration [s m-1]
        self.theta_T = theta_T
        self.LAI = LAI

    def canopy_conductance(self):
        '''
        # calculate surface resistances using Jarvis-Stewart model
        Parameters:
        - Swin :  incoming short wave radiation [W m-2]
        - wfc  :  volumetric water content field capacity[-]
        - w2   :  volumetric water content deeper soil layer [m3 m-3]
        - wwilt:  volumetric water content wilting point  [-]
        - gD   :  correction factor transpiration for VPD [-]
        - VPD  :  saturated vapor pressure [KPa]
        - rsmin:  minimum resistance transpiration [s m-1]
        - theta_T:  potential temperature [K]?
        - LAI  :  leaf area index [-]
        - f1   :  factors accounting for the influence of shortwave radiation [-]
        - f2   :  factors accounting for the influence of air vapor deficit [-]
        - f3   :  factors accounting for the influence of air temperature [-]
        - f4   :  factors accounting for the influence of root zone soil moisture on stomatal resistance [-]
        Returns:
        - R_c  :  resistance transpiration [s m-1]

        Returns
        -------

        Reference
        Jarvis, P. G. [1976].
        The interpretation of the variations in leaf water potential and stomatal conductance found in canopies in the field.
        Philosophical Transactions of the Royal Society of London. Series B, Biological Sciences, 593-610. Chicago

        Stewart, J. B. [1988].
        Modelling surface conductance of pine forest.
        Agricultural and Forest meteorology, 43(1), 19-35.

        Alfieri, J. G., D. Niyogi, P. D. Blanken, F. Chen, M. A. LeMone, K. E. Mitchell, M. B. Ek, and A. Kumar, [2008].
        Estimation of the Minimum Canopy Resistance for Croplands and Grasslands Using Data from the 2002 International H2O Project.
        Mon. Wea. Rev., 136, 4452–4469, https://doi.org/10.1175/2008MWR2524.1.
        '''
        f = 0.55 * self.Swin / 100 * 2.0 / self.LAI  # 0.55 is the fraction of PAR in SWin, see Alfieri et al. 2008
        # r_cmin = 72. # sm-1  see Alfieri et al. 2008 abstract
        r_cmax = 5000.
        F1 = (f + self.rsmin / r_cmax) / (f + 1.0)
        F1 = np.where(F1 > 1.0, 1, F1)
        F1 = np.where(F1 < 0.0, 0.0001, F1)

        F2 = (self.w2 - self.wwilt) / (self.wfc - self.wwilt)
        F2 = np.where(F2 > 1.0, 1, F2)
        F2 = np.where(F2 < 0.0, 0.0001, F2)

        self.VPD = np.where(self.VPD < 0.0, 0.0, self.VPD)
        # F3 = 1.0/ (1.0+ HS * self.VPD)
        F3 = 1.0 - self.gD * self.VPD
        # F3     = 1. / np.exp(- self.gD * self.VPD)
        F3 = np.where(F3 > 1.0, 1, F3)
        F3 = np.where(F3 < 0.01, 0.01, F3)

        F4 = 1. / (1. - 0.0016 * (298.0 - self.theta_T) ** 2.)  # **4.0
        F4 = np.where(F4 > 1.0, 1, F4)
        F4 = np.where(F4 < 0.0, 0.0001, F4)

        R_c = self.rsmin / (self.LAI * F1 * F2 * F3 * F4)

        print(max(F1))
        print(max(F2))
        print(max(F3))
        print(max(F4))
        return R_c,F1,F2,F3,F4

def split_data(station_list,input_dir):
    station_len = len(station_list['use_Syear'])

    for i in np.arange(station_len):
        indata = input_dir + f"input_{station_list['SiteName'][i]}" + f"_{station_list['use_Syear'][i]}" + f"_{station_list['use_Eyear'][i]}.nc"
        df = xr.open_dataset(indata, engine='netcdf4')  # .squeeze()
        sinfo = station_list.iloc[i]
        syear = station_list['use_Syear']
        eyear = station_list['use_Eyear']
        years = eyear-syear
        t = 365*2*24
        if years[i]>1:
            t = 365 * 2 * 24
            train = xr.Dataset({
                'CO2air': (('time', 'lat', 'lon'), df['CO2air'].values[:t]),
                'LAI': (('time', 'lat', 'lon'), df['LAI'].values[:t]),
                'LAI_alternative': (('time', 'lat', 'lon'), df['LAI_alternative'].values[:t]),
                'LWdown': (('time', 'lat', 'lon'), df['LWdown'].values[:t]),
                'Precip': (('time', 'lat', 'lon'), df['Precip'].values[:t]),
                'Psurf': (('time', 'lat', 'lon'), df['Psurf'].values[:t] ),  # Kpa
                'Qair': (('time', 'lat', 'lon'), df['Qair'].values[:t]),
                'RH': (('time', 'lat', 'lon'), df['RH'].values[:t]),
                'SWdown': (('time', 'lat', 'lon'), df['SWdown'].values[:t]),
                'Tair': (('time', 'lat', 'lon'), df['Tair'].values[:t]),
                'VPD': (('time', 'lat', 'lon'), df['VPD'].values[:t] ),  # Kpa
                'esat': (('time', 'lat', 'lon'), df['esat'].values[:t]),  # Kpa
                'eair': (('time', 'lat', 'lon'), df['eair'].values[:t]),  # Kpa

                'Wind': (('time', 'lat', 'lon'), df['Wind'].values[:t]),
                'GPP': (('time', 'lat', 'lon'), df['GPP'].values[:t]),
                'Qg': (('time', 'lat', 'lon'), df['Qg'].values[:t]),
                'Qh': (('time', 'lat', 'lon'), df['Qh'].values[:t]),
                'Qh_cor': (('time', 'lat', 'lon'), df['Qh_cor'].values[:t]),
                'Qle': (('time', 'lat', 'lon'), df['Qle'].values[:t]),
                'Qle_cor': (('time', 'lat', 'lon'), df['Qle_cor'].values[:t]),
                'Ustar': (('time', 'lat', 'lon'), df['Ustar'].values[:t]),
                'Rnet': (('time', 'lat', 'lon'), df['Rnet'].values[:t]),
                'Resp': (('time', 'lat', 'lon'), df['Resp'].values[:t]),
                'NEE': (('time', 'lat', 'lon'), df['NEE'].values[:t]),
                # soil
                'swc1': (('time', 'lat', 'lon'), df['swc1'].values[:t]),
                'swc2': (('time', 'lat', 'lon'), df['swc2'].values[:t]),
                'swc3': (('time', 'lat', 'lon'), df['swc3'].values[:t]),
                'swc4': (('time', 'lat', 'lon'), df['swc4'].values[:t]),
                'swc_root': (('time', 'lat', 'lon'), df['swc_root'].values[:t]),
                'swc_surf': (('time', 'lat', 'lon'), df['swc_surf'].values[:t]),
                'alp_surf': (('lat', 'lon'), df['alp_surf'].values),
                'alp_mean': (('lat', 'lon'), df['alp_mean'].values),
                'bld_surf': (('lat', 'lon'), df['bld_surf'].values),
                'bld_mean': (('lat', 'lon'), df['bld_mean'].values),
                'Ks_surf': (('lat', 'lon'), df['Ks_surf'].values),
                'Ks_mean': (('lat', 'lon'), df['Ks_mean'].values),
                'n_surf': (('lat', 'lon'), df['n_surf'].values),
                'n_mean': (('lat', 'lon'), df['n_mean'].values),
                'omega_mean': (('lat', 'lon'), df['omega_mean'].values),
                'omega_zero': (('lat', 'lon'), df['omega_zero'].values),
                'sfc_surf': (('lat', 'lon'), df['sfc_surf'].values),
                'sfc_mean': (('lat', 'lon'), df['sfc_mean'].values),  # filed capacity of soil saturation
                'sh_surf': (('lat', 'lon'), df['sh_surf'].values),
                'sh_mean': (('lat', 'lon'), df['sh_mean'].values),
                'slt_surf': (('lat', 'lon'), df['slt_surf'].values),
                'slt_mean': (('lat', 'lon'), df['slt_mean'].values),
                'sn_surf': (('lat', 'lon'), df['sn_surf'].values),
                'sn_mean': (('lat', 'lon'), df['sn_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'wiltingpoint_surf': (('lat', 'lon'), df['wiltingpoint_surf'].values),
                'wiltingpoint_mean': (('lat', 'lon'), df['wiltingpoint_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'canopy_height': (df['canopy_height'].values),
                'elevation': (df['elevation'].values),
                'reference_height': (df['reference_height'].values),
                'IGBP_CLASS': (df['IGBP_CLASS'].values),
                'z_T': (sinfo['z_T']),
                'z_U': (sinfo['z_U']),
                'dt': (sinfo['dt']),
                'C3': (sinfo['C3']),
                'D0': (sinfo['D0']),
                'rss_v': (sinfo['rss_v']),
                'g_min_c': (sinfo['g_min_c']),
                'Kr': (sinfo['Kr']),
                'dl': (sinfo['dl']),
                'z0_soil': (sinfo['z0_soil']),
                'Km': (sinfo['Km']),
                'cd': (sinfo['cd']),
            },
                coords={'time': (('time'), df['time'].values[:t]),
                        'lat': (('lat'), [sinfo['lat']]),
                        'lon': (('lon'), [sinfo['lon']]),
                        })
            test  = xr.Dataset({
                'CO2air': (('time', 'lat', 'lon'), df['CO2air'].values[t:]),
                'LAI': (('time', 'lat', 'lon'), df['LAI'].values[t:]),
                'LAI_alternative': (('time', 'lat', 'lon'), df['LAI_alternative'].values[t:]),
                'LWdown': (('time', 'lat', 'lon'), df['LWdown'].values[t:]),
                'Precip': (('time', 'lat', 'lon'), df['Precip'].values[t:]),
                'Psurf': (('time', 'lat', 'lon'), df['Psurf'].values[t:] ),  # Kpa
                'Qair': (('time', 'lat', 'lon'), df['Qair'].values[t:]),
                'RH': (('time', 'lat', 'lon'), df['RH'].values[t:]),
                'SWdown': (('time', 'lat', 'lon'), df['SWdown'].values[t:]),
                'Tair': (('time', 'lat', 'lon'), df['Tair'].values[t:]),
                'VPD': (('time', 'lat', 'lon'), df['VPD'].values[t:] ),  # Kpa
                'esat': (('time', 'lat', 'lon'), df['esat'].values[t:]),  # Kpa
                'eair': (('time', 'lat', 'lon'), df['eair'].values[t:]),  # Kpa

                'Wind': (('time', 'lat', 'lon'), df['Wind'].values[t:]),
                'GPP': (('time', 'lat', 'lon'), df['GPP'].values[t:]),
                'Qg': (('time', 'lat', 'lon'), df['Qg'].values[t:]),
                'Qh': (('time', 'lat', 'lon'), df['Qh'].values[t:]),
                'Qh_cor': (('time', 'lat', 'lon'), df['Qh_cor'].values[t:]),
                'Qle': (('time', 'lat', 'lon'), df['Qle'].values[t:]),
                'Qle_cor': (('time', 'lat', 'lon'), df['Qle_cor'].values[t:]),
                'Ustar': (('time', 'lat', 'lon'), df['Ustar'].values[t:]),
                'Rnet': (('time', 'lat', 'lon'), df['Rnet'].values[t:]),
                'Resp': (('time', 'lat', 'lon'), df['Resp'].values[t:]),
                'NEE': (('time', 'lat', 'lon'), df['NEE'].values[t:]),
                # soil
                'swc1': (('time', 'lat', 'lon'), df['swc1'].values[t:]),
                'swc2': (('time', 'lat', 'lon'), df['swc2'].values[t:]),
                'swc3': (('time', 'lat', 'lon'), df['swc3'].values[t:]),
                'swc4': (('time', 'lat', 'lon'), df['swc4'].values[t:]),
                'swc_root': (('time', 'lat', 'lon'), df['swc_root'].values[t:]),
                'swc_surf': (('time', 'lat', 'lon'), df['swc_surf'].values[t:]),
                'alp_surf': (('lat', 'lon'), df['alp_surf'].values),
                'alp_mean': (('lat', 'lon'), df['alp_mean'].values),
                'bld_surf': (('lat', 'lon'), df['bld_surf'].values),
                'bld_mean': (('lat', 'lon'), df['bld_mean'].values),
                'Ks_surf': (('lat', 'lon'), df['Ks_surf'].values),
                'Ks_mean': (('lat', 'lon'), df['Ks_mean'].values),
                'n_surf': (('lat', 'lon'), df['n_surf'].values),
                'n_mean': (('lat', 'lon'), df['n_mean'].values),
                'omega_mean': (('lat', 'lon'), df['omega_mean'].values),
                'omega_zero': (('lat', 'lon'), df['omega_zero'].values),
                'sfc_surf': (('lat', 'lon'), df['sfc_surf'].values),
                'sfc_mean': (('lat', 'lon'), df['sfc_mean'].values),  # filed capacity of soil saturation
                'sh_surf': (('lat', 'lon'), df['sh_surf'].values),
                'sh_mean': (('lat', 'lon'), df['sh_mean'].values),
                'slt_surf': (('lat', 'lon'), df['slt_surf'].values),
                'slt_mean': (('lat', 'lon'), df['slt_mean'].values),
                'sn_surf': (('lat', 'lon'), df['sn_surf'].values),
                'sn_mean': (('lat', 'lon'), df['sn_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'wiltingpoint_surf': (('lat', 'lon'), df['wiltingpoint_surf'].values),
                'wiltingpoint_mean': (('lat', 'lon'), df['wiltingpoint_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'canopy_height': (df['canopy_height'].values),
                'elevation': (df['elevation'].values),
                'reference_height': (df['reference_height'].values),
                'IGBP_CLASS': (df['IGBP_CLASS'].values),
                'z_T': (sinfo['z_T']),
                'z_U': (sinfo['z_U']),
                'dt': (sinfo['dt']),
                'C3': (sinfo['C3']),
                'D0': (sinfo['D0']),
                'rss_v': (sinfo['rss_v']),
                'g_min_c': (sinfo['g_min_c']),
                'Kr': (sinfo['Kr']),
                'dl': (sinfo['dl']),
                'z0_soil': (sinfo['z0_soil']),
                'Km': (sinfo['Km']),
                'cd': (sinfo['cd']),
            },
                coords={'time': (('time'), df['time'].values[t:]),
                        'lat': (('lat'), [sinfo['lat']]),
                        'lon': (('lon'), [sinfo['lon']]),
                        })
        else:
            t = 365*24
            train = xr.Dataset({
                'CO2air': (('time', 'lat', 'lon'), df['CO2air'].values[:t]),
                'LAI': (('time', 'lat', 'lon'), df['LAI'].values[:t]),
                'LAI_alternative': (('time', 'lat', 'lon'), df['LAI_alternative'].values[:t]),
                'LWdown': (('time', 'lat', 'lon'), df['LWdown'].values[:t]),
                'Precip': (('time', 'lat', 'lon'), df['Precip'].values[:t]),
                'Psurf': (('time', 'lat', 'lon'), df['Psurf'].values[:t] ),  # Kpa
                'Qair': (('time', 'lat', 'lon'), df['Qair'].values[:t]),
                'RH': (('time', 'lat', 'lon'), df['RH'].values[:t]),
                'SWdown': (('time', 'lat', 'lon'), df['SWdown'].values[:t]),
                'Tair': (('time', 'lat', 'lon'), df['Tair'].values[:t]),
                'VPD': (('time', 'lat', 'lon'), df['VPD'].values[:t] ),  # Kpa
                'esat': (('time', 'lat', 'lon'), df['esat'].values[:t]),  # Kpa
                'eair': (('time', 'lat', 'lon'), df['eair'].values[:t]),  # Kpa

                'Wind': (('time', 'lat', 'lon'), df['Wind'].values[:t]),
                'GPP': (('time', 'lat', 'lon'), df['GPP'].values[:t]),
                'Qg': (('time', 'lat', 'lon'), df['Qg'].values[:t]),
                'Qh': (('time', 'lat', 'lon'), df['Qh'].values[:t]),
                'Qh_cor': (('time', 'lat', 'lon'), df['Qh_cor'].values[:t]),
                'Qle': (('time', 'lat', 'lon'), df['Qle'].values[:t]),
                'Qle_cor': (('time', 'lat', 'lon'), df['Qle_cor'].values[:t]),
                'Ustar': (('time', 'lat', 'lon'), df['Ustar'].values[:t]),
                'Rnet': (('time', 'lat', 'lon'), df['Rnet'].values[:t]),
                'Resp': (('time', 'lat', 'lon'), df['Resp'].values[:t]),
                'NEE': (('time', 'lat', 'lon'), df['NEE'].values[:t]),
                # soil
                'swc1': (('time', 'lat', 'lon'), df['swc1'].values[:t]),
                'swc2': (('time', 'lat', 'lon'), df['swc2'].values[:t]),
                'swc3': (('time', 'lat', 'lon'), df['swc3'].values[:t]),
                'swc4': (('time', 'lat', 'lon'), df['swc4'].values[:t]),
                'swc_root': (('time', 'lat', 'lon'), df['swc_root'].values[:t]),
                'swc_surf': (('time', 'lat', 'lon'), df['swc_surf'].values[:t]),
                'alp_surf': (('lat', 'lon'), df['alp_surf'].values),
                'alp_mean': (('lat', 'lon'), df['alp_mean'].values),
                'bld_surf': (('lat', 'lon'), df['bld_surf'].values),
                'bld_mean': (('lat', 'lon'), df['bld_mean'].values),
                'Ks_surf': (('lat', 'lon'), df['Ks_surf'].values),
                'Ks_mean': (('lat', 'lon'), df['Ks_mean'].values),
                'n_surf': (('lat', 'lon'), df['n_surf'].values),
                'n_mean': (('lat', 'lon'), df['n_mean'].values),
                'omega_mean': (('lat', 'lon'), df['omega_mean'].values),
                'omega_zero': (('lat', 'lon'), df['omega_zero'].values),
                'sfc_surf': (('lat', 'lon'), df['sfc_surf'].values),
                'sfc_mean': (('lat', 'lon'), df['sfc_mean'].values),  # filed capacity of soil saturation
                'sh_surf': (('lat', 'lon'), df['sh_surf'].values),
                'sh_mean': (('lat', 'lon'), df['sh_mean'].values),
                'slt_surf': (('lat', 'lon'), df['slt_surf'].values),
                'slt_mean': (('lat', 'lon'), df['slt_mean'].values),
                'sn_surf': (('lat', 'lon'), df['sn_surf'].values),
                'sn_mean': (('lat', 'lon'), df['sn_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'wiltingpoint_surf': (('lat', 'lon'), df['wiltingpoint_surf'].values),
                'wiltingpoint_mean': (('lat', 'lon'), df['wiltingpoint_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'canopy_height': (df['canopy_height'].values),
                'elevation': (df['elevation'].values),
                'reference_height': (df['reference_height'].values),
                'IGBP_CLASS': (df['IGBP_CLASS'].values),
                'z_T': (sinfo['z_T']),
                'z_U': (sinfo['z_U']),
                'dt': (sinfo['dt']),
                'C3': (sinfo['C3']),
                'D0': (sinfo['D0']),
                'rss_v': (sinfo['rss_v']),
                'g_min_c': (sinfo['g_min_c']),
                'Kr': (sinfo['Kr']),
                'dl': (sinfo['dl']),
                'z0_soil': (sinfo['z0_soil']),
                'Km': (sinfo['Km']),
                'cd': (sinfo['cd']),
            },
                coords={'time': (('time'), df['time'].values[:t]),
                        'lat': (('lat'), [sinfo['lat']]),
                        'lon': (('lon'), [sinfo['lon']]),
                        })
            test = xr.Dataset({
                'CO2air': (('time', 'lat', 'lon'), df['CO2air'].values[t:]),
                'LAI': (('time', 'lat', 'lon'), df['LAI'].values[t:]),
                'LAI_alternative': (('time', 'lat', 'lon'), df['LAI_alternative'].values[t:]),
                'LWdown': (('time', 'lat', 'lon'), df['LWdown'].values[t:]),
                'Precip': (('time', 'lat', 'lon'), df['Precip'].values[t:]),
                'Psurf': (('time', 'lat', 'lon'), df['Psurf'].values[t:] ),  # Kpa
                'Qair': (('time', 'lat', 'lon'), df['Qair'].values[t:]),
                'RH': (('time', 'lat', 'lon'), df['RH'].values[t:]),
                'SWdown': (('time', 'lat', 'lon'), df['SWdown'].values[t:]),
                'Tair': (('time', 'lat', 'lon'), df['Tair'].values[t:]),
                'VPD': (('time', 'lat', 'lon'), df['VPD'].values[t:] ),  # Kpa
                'esat': (('time', 'lat', 'lon'), df['esat'].values[t:]),  # Kpa
                'eair': (('time', 'lat', 'lon'), df['eair'].values[t:]),  # Kpa

                'Wind': (('time', 'lat', 'lon'), df['Wind'].values[t:]),
                'GPP': (('time', 'lat', 'lon'), df['GPP'].values[t:]),
                'Qg': (('time', 'lat', 'lon'), df['Qg'].values[t:]),
                'Qh': (('time', 'lat', 'lon'), df['Qh'].values[t:]),
                'Qh_cor': (('time', 'lat', 'lon'), df['Qh_cor'].values[t:]),
                'Qle': (('time', 'lat', 'lon'), df['Qle'].values[t:]),
                'Qle_cor': (('time', 'lat', 'lon'), df['Qle_cor'].values[t:]),
                'Ustar': (('time', 'lat', 'lon'), df['Ustar'].values[t:]),
                'Rnet': (('time', 'lat', 'lon'), df['Rnet'].values[t:]),
                'Resp': (('time', 'lat', 'lon'), df['Resp'].values[t:]),
                'NEE': (('time', 'lat', 'lon'), df['NEE'].values[t:]),
                # soil
                'swc1': (('time', 'lat', 'lon'), df['swc1'].values[t:]),
                'swc2': (('time', 'lat', 'lon'), df['swc2'].values[t:]),
                'swc3': (('time', 'lat', 'lon'), df['swc3'].values[t:]),
                'swc4': (('time', 'lat', 'lon'), df['swc4'].values[t:]),
                'swc_root': (('time', 'lat', 'lon'), df['swc_root'].values[t:]),
                'swc_surf': (('time', 'lat', 'lon'), df['swc_surf'].values[t:]),
                'alp_surf': (('lat', 'lon'), df['alp_surf'].values),
                'alp_mean': (('lat', 'lon'), df['alp_mean'].values),
                'bld_surf': (('lat', 'lon'), df['bld_surf'].values),
                'bld_mean': (('lat', 'lon'), df['bld_mean'].values),
                'Ks_surf': (('lat', 'lon'), df['Ks_surf'].values),
                'Ks_mean': (('lat', 'lon'), df['Ks_mean'].values),
                'n_surf': (('lat', 'lon'), df['n_surf'].values),
                'n_mean': (('lat', 'lon'), df['n_mean'].values),
                'omega_mean': (('lat', 'lon'), df['omega_mean'].values),
                'omega_zero': (('lat', 'lon'), df['omega_zero'].values),
                'sfc_surf': (('lat', 'lon'), df['sfc_surf'].values),
                'sfc_mean': (('lat', 'lon'), df['sfc_mean'].values),  # filed capacity of soil saturation
                'sh_surf': (('lat', 'lon'), df['sh_surf'].values),
                'sh_mean': (('lat', 'lon'), df['sh_mean'].values),
                'slt_surf': (('lat', 'lon'), df['slt_surf'].values),
                'slt_mean': (('lat', 'lon'), df['slt_mean'].values),
                'sn_surf': (('lat', 'lon'), df['sn_surf'].values),
                'sn_mean': (('lat', 'lon'), df['sn_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'wiltingpoint_surf': (('lat', 'lon'), df['wiltingpoint_surf'].values),
                'wiltingpoint_mean': (('lat', 'lon'), df['wiltingpoint_mean'].values),
                'snd_surf': (('lat', 'lon'), df['snd_surf'].values),
                'snd_mean': (('lat', 'lon'), df['snd_mean'].values),
                'canopy_height': (df['canopy_height'].values),
                'elevation': (df['elevation'].values),
                'reference_height': (df['reference_height'].values),
                'IGBP_CLASS': (df['IGBP_CLASS'].values),
                'z_T': (sinfo['z_T']),
                'z_U': (sinfo['z_U']),
                'dt': (sinfo['dt']),
                'C3': (sinfo['C3']),
                'D0': (sinfo['D0']),
                'rss_v': (sinfo['rss_v']),
                'g_min_c': (sinfo['g_min_c']),
                'Kr': (sinfo['Kr']),
                'dl': (sinfo['dl']),
                'z0_soil': (sinfo['z0_soil']),
                'Km': (sinfo['Km']),
                'cd': (sinfo['cd']),
            },
                coords={'time': (('time'), df['time'].values[t:]),
                        'lat': (('lat'), [sinfo['lat']]),
                        'lon': (('lon'), [sinfo['lon']]),
                        })

        train_output_path = input_dir + f"input_train_{station_list['SiteName'][i]}" + f"_{station_list['use_Syear'][i]}" + f"_{station_list['use_Eyear'][i]}.nc"
        test_output_path = input_dir + f"input_test_{station_list['SiteName'][i]}" + f"_{station_list['use_Syear'][i]}" + f"_{station_list['use_Eyear'][i]}.nc"
        train.to_netcdf(train_output_path)
        test.to_netcdf(test_output_path)


    return None

def data_gen(input_dir):
    x_train = None
    y_train = None
    # x_test = None
    # y_test = None
    stnlist = input_dir + "selected_list.txt"
    print(stnlist)
    station_list = pd.read_csv(stnlist, header=0)
    station_len = len(station_list['use_Syear'])
    split_data(station_list,input_dir)
    for i in np.arange(station_len):
        indata = input_dir + f"input_train_{station_list['SiteName'][i]}" + f"_{station_list['use_Syear'][i]}" + f"_{station_list['use_Eyear'][i]}.nc"
        df = xr.open_dataset(indata,engine='netcdf4')#.squeeze()
        bas = basic_vars(df)
        met = met_vars(df)
        soil = soil_vars(df)
        veg = veg_vars(df, bas, met)
        ene = energy_vars(df, met, bas, veg)
        met.rav = calc_R_A_Norman1995(bas.z_T, met.Ustar,met.ObukhovLength, met.d0,met.z0m)
        Psy = met.Psy
        delta = met.delta
        Rn = ene.Rnet
        Q = ene.Qg
        rho  = met.rho
        Cp = met.Cp
        VPD = met.VPD
        Ga = 1./met.rav
        LAI = veg.LAI
        SWdown = ene.SWdown
        swc_root = soil.swc_root
        WP = soil.WP
        FC = soil.FC
        Tair_K = met.Tair_K
        rsmin = 72.  # sm-1  see Alfieri et al. 2008 abstract
        gD = 0.1914  # Kpa  see Alfieri et al. 2008
        p1 = canopy_conductance_Jarvis1976(ene.SWdown, soil.FC, soil.swc_root, soil.WP, gD,
                                           met.VPD, rsmin, met.Tair_K, veg.LAI)
        rc, F1, F2, F3, F4 = p1.canopy_conductance()
        Wind = met.Wind #风速 m/s
        RH = met.RH #相对湿度 %
        CO2air = met.CO2air #空气中的CO2浓度%
        # SZA = ene.SZA #太阳角度
        albedo = veg.albedo #土壤反照率 %
        X = np.stack((Rn,Q,delta,rho,Cp,VPD,Ga,Psy,LAI,SWdown,swc_root,WP,FC,Tair_K,F1,F2,F3,F4,Wind,RH,CO2air,albedo))#Rn,Q,delta,rho,Cp,VPD,Ga,Psy,LAI,SWdown,swc_root,WP,FC,Tair_K
        LE = df['Qle_cor'].values
        Y = LE
        if i==0:
            x_train = X
            y_train = Y
        else:
            x_train = np.concatenate((x_train,X),axis = 1)
            y_train = np.concatenate((y_train, Y), axis=0)



    return x_train[:,:,0,0],y_train[:,0,0]