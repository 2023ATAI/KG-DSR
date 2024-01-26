
import numpy as np
import pandas as pd
import xarray as xr
import scipy.optimize as opt
from scipy.optimize import root
from  Unit_Conversions import Unit_Conversions # Conductance_mol_to_ms, Conductance_ms_to_mol
import sys
import math
from joblib import Parallel, delayed
n_jobs = -1 # Number of CPUs to use (-1 for all available CPUs)
import numpy as np
    
class canopy_conductance_Liang2022:
    def __init__(self,Tleaf,RH,PAR,Lambda_c,ca,ps,LAI,f5,carbon_fixation='C3'):
        self.name = 'canopy_conductance_Liang2022'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"

        self.Unit_Convert=Unit_Conversions()
        if carbon_fixation == 'C3':
            self.Vc_25                   =   50.0
            self.Jm_25                   =   2.3 * 50. 
            self.gamma_25                =   45.0 
            self.Q1                      =   0.999
            self.Q2                      =   0.86
            self.Kc_25                   =   272.0     # (#ppm) Michaelis constant of Rubisco for CO2  or /10 Pa at leaf temperature of 25 degree
            self.Ko_25                   =   16.58      # (#kPa) Michaelis constant of Rubisco for O2 at leaf temperature of 25 degree

        elif carbon_fixation == 'C4':
            #need to be updated
            self.Vc_25                   =   50.
            self.Jm_25                   =   2.3 * 50.
            self.gamma_25                =   45.0
            self.Q1                      =   0.999
            self.Q2                      =   0.86
        else:
            print('carbon_fixation must be C3 or C4')
            sys.exit(1)
        
        self.Ii                      =   PAR *4.6
        self.ps                      =   ps*10.0 #convert to hpa
        self.Tleaf                   =   Tleaf 
        self.LAI                     =   LAI
        self.phi                     =   0.425  # effective quantum yield of electrons from incident irradiance (photosythetic photon flux density, PPFD)

        ###~~~~~temperature correction~~~~~~~~~~~~~~~~~~~~~~~~~
        self.gamma_p = self.gamma_25 * np.exp(37.83 / 8.31 * (1. / 298 - 1. / (273.15 + self.Tleaf)) * 1000)  # ppm about 4.5Pa,Pa*10 convert to ppm or umol mol-1
        # effective Michaelis constant for carboxylation #ppm
        self.Km =  self.Kc_25 * np.exp(59400.0 * (self.Tleaf - 25.0) / (298.0 * 8.31 * (self.Tleaf + 273.))) * (
                1 + 20.5 * 1e+3 / (self.Ko_25 * 1e+3 * np.exp(36000. * (self.Tleaf - 25) / (298 * 8.31 * (self.Tleaf + 273)))))
        # carboxylation capacity #umol m-2 s-1
        self.Vc = self.Vc_25 * np.exp(64800 * (self.Tleaf - 25) / (298 * 8.31 * (self.Tleaf + 273)))
        # electron transport capacity #umol m-2 s-1
        self.Jm = self.Jm_25 * np.exp(37000 * (self.Tleaf - 25) / (298 * 8.31 * (self.Tleaf + 273))) * (
                1 + np.exp((710 * 298 - 220000) / 8.31 / 298)) / (
                     1 + np.exp((710 * (self.Tleaf + 273) - 220000) / 8.31 / (self.Tleaf + 273)))
        #leaf to air vapor pressure difference, #mmol mol-1 or mbar/hPa
        self.Dl = 6.107*np.exp(17.27*self.Tleaf/(237.3+self.Tleaf))*(1-RH) #6.107*exp(17.27*Tleaf/(237.3+Tleaf))*(1-0.2)*(1-0.1*sin((Time_hour-3)/2))
        if self.Dl <= 0.0:
            self.Dl = 0.0001
        # potntial elerctron transport capacity and its value is the lesser root of Q2*J**2-(Jm+phi*I)*J+Jm*phi*I=0

        #self.Ja = min(self.QuadRoot_result(self.Q2, -self.Jm - self.phi * self.Ii, self.Jm * self.phi * self.Ii))  # QuadRoot(c(Q2,b,c))
        #print(self.Q2[:], -self.Jm - self.phi * self.Ii, self.Jm * self.phi * self.Ii)
        #self.Ja= np.minimum(aa)

        self.Ja = min(self.QuadRoot_result(self.Q2, -self.Jm - self.phi * self.Ii, self.Jm * self.phi * self.Ii))  # QuadRoot(c(Q2,b,c))

        #n_jobs = -1 # Number of CPUs to use (-1 for all available CPUs)
        #results = Parallel(n_jobs=n_jobs)(
        #delayed(self.QuadRoot_result)(self.Q2[i], -self.Jm[i] - self.phi[i] * self.Ii[i], self.Jm[i] * self.phi[i] * self.Ii[i])
        #for i in range(len(self.Q2))
        #    )
        
        ##self.Ja = np.minimum(results)
        ##mesphyll conductance #mol m-2 s-1
        self.gm = self.mesophyll_conductance(carbon_fixation='C3') 

        self.Lambda_c= Lambda_c  
        self.ca=ca/1.8

        self.f5=f5

    def QuadRoot_result(self,a, b, c):
        ### Solve Quadratic Formula
        delta = b ** 2 - 4 * a * c
        if delta > 0:  # first case D>0
            x_1 = (-b + np.sqrt(delta)) / (2 * a)
            x_2 = (-b - np.sqrt(delta)) / (2 * a)
            return [x_1, x_2]
        if delta == 0:  # second case D=0
            x = -b / (2 * a)
            return x
        if delta < 0:  # third case D<0
            return np.nan
           
    def mesophyll_conductance(self,carbon_fixation='C3'):
        '''
        mesophyll conductance
    
        input:
            carbon_fixation   : C3 or C4  

        output:
            g_m  : mesophyll conductance (mol m-2 s-1)
        '''
        #caution: large different between Ronda2001 and Liang2022 scheme need to check
        if carbon_fixation == 'C3':
                g_m          =      0.004 * self.Vc_25 * np.exp(20.03 - 49600. / (8.31 * (self.Tleaf + 273.15))) 
        elif carbon_fixation == 'C4':
                g_m          =      0.004 * self.Vc_25 * np.exp(20.03 - 49600. / (8.31 * (self.Tleaf + 273.15))) 
        else:
            print('carbon_fixation is not defined correctly')
            sys.exit(1)
        return g_m

    def gs_upscaling(self,gs,LAI):
        '''
        upscaling stomatal conductance to canopy scale

        reference:
            Ding, R., Kang, S., Du, T., Hao, X., & Zhang, Y. (2014). 
            Scaling Up Stomatal Conductance from Leaf to Canopy Using a Dual-Leaf Model
            for Estimating Crop Evapotranspiration. PLoS ONE, 9(4), e95584.
            https://doi.org/10.1371/journal.pone.0095584

        input:
            gs  : stomatal conductance (mol m-2 s-1)
            LAI : leaf area index (m2 m-2)

        output:
            g_s  : stomatal conductance (mol m-2 s-1)
        '''
        if LAI <= 2.0:
            eLAI=LAI
        elif LAI > 2.0 and LAI <= 4.0:
            eLAI=LAI/2.0
        else:
            eLAI=2.0
        g_s = gs * eLAI
        return g_s

    def A_Farquhar_gm(self,):
        def f1_gm(Km, Vc, gamma_p, gm, Lambda_c, Dl, ca):
            f = lambda x: (1 / (((x + Km) ** 2) / Vc / (Km + gamma_p) + 1 / gm) - Vc * (x - gamma_p) / (x + Km) / (
                ((ca - x - Vc * (x - gamma_p) / (x + Km) / gm) ** 2) * Lambda_c / 1.6 / (Dl * 1000) - ca + x + Vc * (x - gamma_p) / (
                x + Km) / gm))
            try:
                sol1 = opt.brentq(f, 50, ca, disp=True)
            except ValueError as err:
                if str(err) == "f(a) and f(b) must have different signs":
                    sol1 = root(f, 50).x
            except RuntimeWarning as err:
                if str(err) == "divide by zero encountered in double_scalars":
                    sol1 = root(f, 50).x
            m = f(sol1)
            data_top = math.floor(sol1)
            n=0
            while (m > 0.5) | (m < -0.5):
                try:
                    sol1 = opt.brentq(f, 50, data_top, disp=True)  # disp=True
                    data_top = data_top - 10
                except ValueError as err:
                    if str(err) == "f(a) and f(b) must have different signs":
                        data_top = data_top - 10
                m = f(sol1)
                n=n+1
                if n>10:
                    break
            return sol1

        def f2_gm(gamma_p, Ja, gm, Lambda_c, ca, Dl):
            f = lambda x: (1 / (4 * (x + 2 * gamma_p) ** 2 / 3 / gamma_p / Ja + 1 / gm) - Ja / 4 * (x - gamma_p) / (x + 2 * gamma_p) / (
                (ca - x - Ja / 4 * (x - gamma_p) / (x + 2 * gamma_p) / gm) ** 2 * Lambda_c / 1.6 / (Dl * 1000) - ca + x + Ja / 4 * (
                x - gamma_p) / (x + 2 * gamma_p) / gm))
            try:
                sol2 = opt.brentq(f, 50, ca, disp=True)  # disp=True
            except ValueError as err:
                if str(err) == "f(a) and f(b) must have different signs":
                    sol2 = root(f, 50).x
            except RuntimeWarning as err:
                if str(err) == "divide by zero encountered in double_scalars":
                    sol2 = root(f, 50).x
            m = f(sol2)
            data_top = math.floor(sol2)
            n=0
            while (m > 0.5) | (m < -0.5):
                try:
                    sol2 = opt.brentq(f, 50, data_top, disp=True)  # disp=True
                    data_top = data_top - 10
                except ValueError as err:
                    if str(err) == "f(a) and f(b) must have different signs":
                        data_top = data_top - 10
                m = f(sol2)
                n=n+1
                if n>10:
                    break
            return sol2

        ##~~~~~caculate Cc when A meets 1/(dcc/dA+1/gm)-A/[(ca-cc-A/gm)**2*lambda/1.6D-(ca-cc-A/gm)]=0 where units of D,ca are ppm or umol mol-1~~~~~~~~~``
        # Robisco limiting
        # --------------------------------------------------------------------------------
        a=f1_gm(self.Km, self.Vc, self.gamma_p, self.gm, self.Lambda_c, self.Dl, self.ca)
       # print(a,elf.Km, self.Vc, self.gamma_p, self.gm, self.Lambda_c, self.Dl, self.ca)

        Cc_Rubisco = a
        A_Rubisco = self.Vc * (Cc_Rubisco - self.gamma_p) / (Cc_Rubisco + self.Km)  # umol m-2 s-1
        Ci_Rubisco = Cc_Rubisco + A_Rubisco / self.gm  # intercellular [CO2] #ppm
        del a
        # RuBP regeneration limiting
        # ================================================================================
        b=f2_gm(self.gamma_p, self.Ja, self.gm, self.Lambda_c, self.ca, self.Dl)
        Cc_RuBP = b  # chloroplast [CO2]
        A_RuBP = self.Ja / 4.0 * (Cc_RuBP - self.gamma_p) / (Cc_RuBP + 2.0 * self.gamma_p)
        Ci_RuBP = Cc_RuBP + A_RuBP / self.gm  # intercellular [CO2]
        del b

        # -----------------------------------------------------------------------------
        # A value is the lesser root of Q2*A^2-(A_Rubisco+A_RuBP)*A+A_Rubisco*A_RuBP=0
        if np.isnan(A_Rubisco):
            Am = A_RuBP
        elif np.isnan(A_RuBP):
            Am = A_Rubisco
        else:
            Am=min(self.QuadRoot_result(self.Q1, -A_Rubisco - A_RuBP, A_Rubisco * A_RuBP))  # QuadRoot(c(Q2,b,c))

        WUE_Rubisco = (self.ca - Ci_Rubisco) / 1.6 / self.Dl
        WUE_Rubisco = np.maximum(WUE_Rubisco,0.0)
        WUE_RuBP = (self.ca - Ci_RuBP) / 1.6 / self.Dl

        # intrinsic water use efficiency #umol mol-1
        WUEi_Rubisco = WUE_Rubisco * self.Dl
        WUEi_RuBP = WUE_RuBP * self.Dl
        gs_Rubisco = A_Rubisco / (self.ca - Ci_Rubisco) * 1000.0 * 1.6
        gs_Rubisco = np.maximum(gs_Rubisco,0.0)
        if np.isnan(A_Rubisco):
            gs_Rubisco = 0.0
            A_Rubisco  = 0.0
        gs_RuBP = A_RuBP / (self.ca - Ci_RuBP) * 1000 * 1.6
        if np.isnan(A_RuBP):
            gs_RuBP = 0.0
            A_RuBP  = 0.0
        gs_RuBP = np.maximum(gs_RuBP,0.0)

        ##select values when min(A_Robisco,A_RuBP) where A_Rubisco = 1, A_RuBP = 2
        if (A_Rubisco == min(A_Rubisco, A_RuBP)):
            A_flag  = 1
            WUE     = WUE_Rubisco
            WUEi    = WUEi_Rubisco
            Ci      = Ci_Rubisco
            ga      = gs_Rubisco
        else:
            A_flag  = 2
            WUE     = WUE_RuBP
            WUEi    = WUEi_RuBP
            Ci      = Ci_RuBP
            ga      = gs_RuBP

        # select values when >WUE_max
        WUE_max = (self.ca - self.gamma_p) / 1.6 / self.Dl
        if ((WUE - WUE_max > 0)|(ga<0.00000001)):
            Am = 0
            ga = 0.00000001 #mmol m-2 s-1
            WUE = np.nan
            WUEi = np.nan
        # transpiration rate #mmol m-2 s-1
        Er = ga * self.Dl / 1000.
        E_Rubisco = gs_Rubisco * self.Dl / 1000.
        E_RuBP = gs_RuBP * self.Dl / 1000.
        ga=self.Unit_Convert.Conductance_mol_to_ms(ga/1000.,self.Tleaf,self.ps/10.)
        try:
            rc=1./(self.gs_upscaling(ga,self.LAI)*self.f5)#*f5
        except:
            rc=[10000.]
        return np.asarray(rc) 

class canopy_conductance_Leuning1995:
    def __init__(self,PAR_t,swc2,LAI,C_s,WP,FC,Kr,g_m_298,Q_10_g_m,T_1_g_m,T_2_g_m,Gamma_298,alpha_0,
                 A_m_max_298,Q_10_A_m,T_1_A_m,T_2_A_m,g_min_c,a_1,D0,Tc_k,C_i, D_s,f5):
        self.name = 'canopy_conductance_Leuning1995'
        self.version = '0.1'
        self.release = '0.1'
        self.date = 'Mar 2023'
        self.author = "Zhongwang Wei / zhongwang007@gmail.com"
        
        self.PAR_t=PAR_t
        self.swc2=swc2
        self.LAI=LAI
        self.C_s=C_s
        self.WP=WP
        self.FC=FC
        self.Kr=Kr
        self.g_m_298=g_m_298
        self.Q_10_g_m=Q_10_g_m
        self.T_1_g_m=T_1_g_m
        self.T_2_g_m=T_2_g_m
        self.Gamma_298=Gamma_298
        self.alpha_0=alpha_0
        self.A_m_max_298=A_m_max_298
        self.Q_10_A_m=Q_10_A_m
        self.T_1_A_m=T_1_A_m
        self.T_2_A_m=T_2_A_m
        self.g_min_c=g_min_c
        self.a_1=a_1
        self.D0=D0
        self.Tc_k=Tc_k
        self.C_i=C_i
        self.D_s=D_s
        self.f5=f5

    def prim_prod_Leuning1995(self,g_m,Gamma):
        '''
        primary productivity
        input:
        Tc_k   : canopy temperature (K)
        g_m    : mesophyll conductance (mm/s)
        Gamma  : CO2 compensation point (mg/m3)
        C_i  : the intercellular CO2 concentration (mg/m3)
        
        output
        A_m    :  primary productivity (mg/J)
        '''
        A_m_max                       =    self.A_m_max_298 * ( self.Q_10_A_m ** (( self.Tc_k -298.)/10.))      
        A_m_max                       =    A_m_max / ((1.0 + np.exp(0.3*( self.T_1_A_m- self.Tc_k))) * (1.0+ np.exp(0.3*( self.Tc_k- self.T_2_A_m)))) 
        A_m                           =    A_m_max * (1.0 - np.exp(- g_m*1e-3 * ( self.C_i- Gamma) /  A_m_max))       
        return A_m
    
    def mesophyll_conductance_Leuning1995(self,):
        '''
        mesophyll conductance
    
        input:
            Tc_k : skin temperature (degK)
            C3   : C3 or C4 (=1 C3; =0 C4)

        output:
            g_m  : mesophyll conductance (mm/s)
        '''
        g_m          =      self.g_m_298*(self.Q_10_g_m**((self.Tc_k-298.0)/10.0))
        g_m          =      g_m/((1.0+np.exp(0.3*(self.T_1_g_m-self.Tc_k)))*(1.0+np.exp(0.3*(self.Tc_k-self.T_2_g_m))))
        return g_m
    
    def light_use_efficiency_Leuning1995(self,Gamma):
        '''
        light use efficiency

        input:
            Gamma  : CO2 compensation point (mg/m3)
            C_s    : CO2 concentration at the leaf surface (mg/m3)
        
        output
            alpha: mg/J
        '''
        alpha                     =     self.alpha_0*(self.C_s-Gamma)/(self.C_s+2.0*Gamma) 
        return alpha
   
    def CO2_compensation_point_Leuning1995(self,):
        '''
        CO2 compensation point

        input:
            Tc_k   : skin temperature (degK)
            C3     : C3 or C4 (=1 C3; =0 C4)

        output:
            Gamma  : CO2 compensation point (mg/m3)

        '''
        Gamma                =      self.Gamma_298*(1.5**((self.Tc_k-298.0)/10.0))
        return Gamma

    def canopy_conductance(self,):
        import scipy.special as sc
   
        '''
        canopy_conductance
    
        input:
            T_sk       :  canopy temperature (K)
            PAR_t      :  photosynthetically active radiation ()
            C_s        :
            C_i        :  CO2 concentration at the leaf surface
            LAI        :  leaf area index (m2/m2)
            theta      :  soil moisture ()
            C3         :  C3 or C4 (=1 C3; =0 C4)
            Kr         :
            g_min_c    :  the cuticular conductance
            D0         :  a tunable empirical parameter
        

        
        output:
            A_g    :  the gross assimilation rate (?)
            A_n    :  the net flow of carbon dioxide into the plant
            A_m    :  primary productivity (mg/J)
            g_m    :  the mesophyll conductance for CO2
            g_cc   :  the canopy (c) conductance to carbon dioxide flow
            Gamma  :  CO2 compensation point (mg/m3)
            d1     :  analytic method to scale conductance parameter
            d2     :  analytic method to scale conductance parameter
            R_d    :  the dark respiration ()
        '''
    
        #########soil moisture stress#################################
        #beta            =      (self.swc2-self.WP)/(self.FC-self.WP)
        #beta            =      np.where(beta > 1.0, 1.0, beta  )  #for C4
        #beta            =      np.where(beta < 0.0, 0.0, beta   )  #for C4
        #f5              =      2.0 * beta  - beta  * beta  
        ###################################################################
        g_m             =      self.mesophyll_conductance_Leuning1995()
        Gamma           =      self.CO2_compensation_point_Leuning1995()
        alpha           =      self.light_use_efficiency_Leuning1995(Gamma)
        A_m             =      self.prim_prod_Leuning1995(g_m,Gamma)

        R_d             =      0.11*A_m

        d1              =      sc.exp1(alpha*self.Kr * self.PAR_t / (A_m+R_d)*np.exp(-self.Kr*self.LAI))
        d2              =      sc.exp1(alpha*self.Kr * self.PAR_t / (A_m+R_d))
        d1              =      np.where(self.PAR_t <= 0.001, 0.0, d1   )  
        d2              =      np.where(self.PAR_t <= 0.001, 0.0, d2   )  

        d1              =      np.where(d1>10000.001, 0.0, d1   )  
        d2              =      np.where(d1>10000.001, 0.0, d2   )  

        A_g             =      (A_m+R_d)*(self.LAI-(d1-d2)/(self.Kr))*self.f5
        A_g             =      np.where(self.PAR_t <= 0.001, 0.0, A_g   )  

        A_n             =      ( A_g- R_d* self.LAI)

        g_cc            =      (self.g_min_c*self.LAI +1000.* self.a_1*(A_m+R_d)*self.f5/((self.C_s-Gamma)*(1.0 + self.D_s/(self.D0)))*(self.LAI-(d1-d2)/(self.Kr)))
        g_cc            =      np.where(self.PAR_t <= 0.001, self.g_min_c*self.LAI, g_cc   )  

        return A_g,A_n,A_m,g_m,g_cc,Gamma,d1,d2,R_d

class canopy_conductance_Jarvis1976:
    def __init__(self,Swin, wfc, w2, wwilt, gD, VPD, rsmin, theta_T, LAI):
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
        f      = 0.55*self.Swin/100*2.0/self.LAI # 0.55 is the fraction of PAR in SWin, see Alfieri et al. 2008
        # r_cmin = 72. # sm-1  see Alfieri et al. 2008 abstract
        r_cmax = 5000.
        F1     = (f+self.rsmin/r_cmax)/(f+1.0)
        F1     = np.where(F1>1.0, 1 ,F1 )  
        F1     = np.where(F1<0.0, 0.0001, F1)  

        F2     =  (self.w2 - self.wwilt)/(self.wfc - self.wwilt)
        F2     = np.where(F2>1.0, 1,F2)  
        F2     = np.where(F2<0.0, 0.0001, F2)  

        self.VPD = np.where(self.VPD<0.0, 0.0, self.VPD)  
        # F3 = 1.0/ (1.0+ HS * self.VPD)
        F3     =  1.0 - self.gD * self.VPD
        #F3     = 1. / np.exp(- self.gD * self.VPD)
        F3     = np.where(F3>1.0,1, F3)  
        F3     = np.where(F3<0.01, 0.01, F3)     

        F4     = 1. / (1. - 0.0016 * (298.0 - self.theta_T) ** 2.)#**4.0
        F4     = np.where(F4>1.0, 1, F4)  
        F4     = np.where(F4<0.0, 0.0001, F4)     

        R_c    = self.rsmin / (self.LAI * F1 * F2 * F3 * F4)


        print(max(F1))
        print( max(F2))
        print(max(F3))
        print( max(F4))
        return R_c
