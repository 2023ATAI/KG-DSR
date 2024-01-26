# This file is part PySiLSMV3, consisting of of high level pySiLSMV3 scripting
# Copyright 2022 Zhongwang Wei@SYSU, zhongwang007@gmail.com 
class Unit_Conversions:
    def __init__(self):
        ## Conversion constants
        self.Kelvin       = 273.15        # conversion degree Celsius to Kelvin
        self.DwDc         = 1.6           # Ratio of the molecular diffusivities for water vapor and CO2
        self.days2seconds = 86400.        # seconds per day
        self.kPa2Pa       = 1000.         # conversion kilopascal (kPa) to pascal (Pa)
        self.Pa2kPa       = 0.001         # conversion pascal (Pa) to kilopascal (kPa)
        self.umol2mol     = 1e-06         # conversion micromole (umol) to mole (mol)
        self.mol2umol     = 1e06          # conversion mole (mol) to micromole (umol)
        self.kg2g         = 1000.         # conversion kilogram (kg) to gram (g)
        self.g2kg         = 0.001         # conversion gram (g) to kilogram (kg)
        self.kJ2J         = 1000.         # conversion kilojoule (kJ) to joule (J)
        self.J2kJ         = 0.001         # conversion joule (J) to kilojoule (kJ)
        self.se_median    = 1.253         # conversion standard error (SE) of the mean to SE of the median (http://influentialpoints.com/Training/standard_error_of_median.htm)
        self.frac2percent = 100.          # conversion between fraction and percent
        ## Physical constants
        self.cp         = 1004.834        # specific heat of air for constant pressure (J K-1 kg-1)
        self.Rgas       = 8.31451         # universal gas constant (J mol-1 K-1)
        self.Rv         = 461.5           # gas constant of water vapor (J kg-1 K-1) (Stull 1988 p.641)
        self.Rd         = 287.0586        # gas constant of dry air (J kg-1 K-1) (Foken 2008 p. 245)
        self.Md         = 0.0289645       # molar mass of dry air (kg mol-1)
        self.Mw         = 0.0180153       # molar mass of water vapor (kg mol-1)
        self.eps        = 0.622           # ratio of the molecular weight of water vapor to dry air (=Mw/Md)
        self.g          = 9.81            # gravitational acceleration (m s-2)
        self.solar_constant = 1366.1      # solar constant, i.e. solar radation at earth distance from the sun (W m-2)
        self.pressure0  = 101325.         # reference atmospheric pressure at sea level (Pa)
        self.Tair0      = 273.15          # reference air temperature (K)
        self.k          = 0.41            # von Karman constant
        self.Cmol       = 0.012011        # molar mass of carbon (kg mol-1)
        self.Omol       = 0.0159994       # molar mass of oxygen (kg mol-1)
        self.H2Omol     = 18.01528/1000.  # molar mass of water (kg mol-1)
        self.sigma      = 5.670367e-08    # Stefan-Boltzmann constant (W m-2 K-4)
        self.Pr         = 0.71            # Prandtl number
        self.Sc_CO2     = 1.07            # Schmidt number for CO2 (Hicks et al. 1987)
    
    def LE_to_ET(self,LE,Tair_C):
        #
        '''
        converts evaporative water flux from energy (LE=latent heat flux) to mass (ET=evapotranspiration) units
        
        reference:
            Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
                             Kluwer Academic Publishers, Dordrecht, Netherlands
            Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany. 
  
        input:
            LE       :  latent heat flux (Wm-2)
            Tair_C    :  air temperature  (C)

        output:
            ET       :  evapotranspiration (kg m-2 s-1 or mm/s)

        '''
        k1 = 2.501
        k2 = 0.00237
        lambda_LE2ET = ( k1 - k2 * Tair_C ) * 1e+06 #the latent heat of vaporization (J kg-1) (Stull 1988 p. 641)
        ET=LE/lambda_LE2ET      
        return ET
    def ET_to_LE(self,ET,Tair_C):
        '''
        converts evaporative water flux from mass (ET=evapotranspiration) to energy (LE=latent heat flux) units
        
        reference:
            Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
                             Kluwer Academic Publishers, Dordrecht, Netherlands
            Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany. 
  
        input:
            ET       :  evapotranspiration (kg m-2 s-1 or mm/s)
            TairC    :  air temperature  (C)

        output:
            LE       :  latent heat flux (Wm-2)
        '''
        k1 = 2.501
        k2 = 0.00237
        lambda_ET2LE = ( k1 - k2 * Tair_C ) * 1e+06   # the latent heat of vaporization (J kg-1) (Stull 1988 p. 641)
        LE=ET*lambda_ET2LE
        return LE

    def Conductance_ms_to_mol(self,G_ms,Tair_C,Pa_K):
        '''
        converts conductance from (m s-1) to (mol m-2 s-1)
        reference: 
            Jones, H.G. 1992. Plants and microclimate: a quantitative approach to environmental plant physiology.
            2nd Edition., Cambridge University Press, Cambridge. 428 p
        input:
            Tair_C   : air temperature (C)
            Pa_K     : atmospheric pressure (kPa) 
            G_ms     : Conductance (m s-1)   
        output:
            G_mol    : Conductance (mol m-2 s-1)
        '''
        G_mol  = G_ms * Pa_K / (self.Rgas * (Tair_C+273.15))
        return G_mol

    def Conductance_mol_to_ms(self,G_mol,Tair_C,Pa_K):
        '''
        converts conductance from  (mol m-2 s-1) to (m s-1)
        reference: 
            Jones, H.G. 1992. Plants and microclimate: a quantitative approach to environmental plant physiology.
            2nd Edition., Cambridge University Press, Cambridge. 428 p
        input:
            Tair_C   : air temperature (C)
            Pa_K     : atmospheric pressure (kPa) 
            G_mol    : Conductance (mol m-2 s-1)
        output:
            G_ms     : Conductance (m s-1)
        '''
        G_ms  = G_mol *  (self.Rgas * (Tair_C+273.15)) /Pa_K 
        return G_ms
    
    


