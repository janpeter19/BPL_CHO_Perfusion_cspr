# setup applicateion data BPL_CHO_Perfusion_cspr_openloop.py 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-27 - Created
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import matplotlib.pyplot as plt 
from pyfmi import load_fmu

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   flag_vendor = 'JM'
   flag_type = 'CS'
   fmu_model ='BPL_CHO_Perfusion_cspr_openloop_windows_jm_cs.fmu'        
   model = load_fmu(fmu_model, log_level=0)  
elif platform.system() == 'Linux':
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-compiled OpenModelica') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_CHO_Perfusion_cspr_openloop_linux_om_cs.fmu'    
         model = load_fmu(fmu_model, log_level=0) 
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_CHO_Perfusion_cspr_openloop_linux_om_me.fmu'    
         model = load_fmu(fmu_model, log_level=0)
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = model.simulate_options()
   opts_std['silent_mode'] = True
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'     
elif flag_type in ['ME', 'me']:
   opts_std = model.simulate_options()
   opts_std["CVode_options"]["verbosity"] = 50 
   opts_std['ncp'] = 500 
   opts_std['result_handling'] = 'binary'  
else:    
   print('There is no FMU for this platform')
  
# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   MSL_usage = model.get('MSL.usage')[0]
   MSL_version = model.get('MSL.version')[0]
   BPL_version = model.get('BPL.version')[0]
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '4.1.0 - used components: RealInput, RealOutput' 
   MSL_version = '4.1.0'
   BPL_version = 'Bioprocess Library version 2.3.2' 
else:    
   print('There is no FMU for this platform')

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: parValue, parLocation, parCheck, diagrams, ax, lines
#------------------------------------------------------------------------------------------------------------------

# Simulation time
simulationTime = 1000.0
prevFinalTime = 0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Create stateValue that later will be used to store final state and used for initialization in 'cont':
stateValue =  {}
stateValue = model.get_states_list()
stateValue.update(timeDiscreteStates)

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = ['bioreactor', 'bioreactor.culture', 'bioreactor.broth_decay']

# Process diagram
fmu_process_diagram ='BPL_CHO_Perfusion_cspr_openloop_process_diagram_om.png'
   
# Create parValue
parValue = {}
parValue['V_start']   = 0.35          # L
parValue['VXv_start'] = 0.35*0.2       
parValue['VXd_start'] = 0.0 
parValue['VXl_start'] = 0.0              
parValue['VG_start'] = 0.35*18.0       
parValue['VGn_start'] = 0.35*10.0      
parValue['VL_start'] = 0.0             
parValue['VN_start'] = 0.0             

parValue['qG_max1'] = 0.2971
parValue['qG_max2'] = 0.0384
parValue['qGn_max1'] = 0.1238
parValue['qGn_max2'] = 0.0218
parValue['mu_d_max'] = 0.1302
parValue['k_toxic'] = 0.0
parValue['alpha'] = 0
parValue['beta'] = 10.0/24

parValue['k_lysis_v'] = 0.0
parValue['k_lysis_d'] = 0.0

eps = 0.10
parValue['eps'] = eps                  # Fraction filtrate flow
parValue['alpha_Xv'] = 0.03            # Fraction Xv in filtrate flow
parValue['alpha_Xd'] = 0.03            # Fraction Xd in filtrate flow
parValue['alpha_Xl'] = eps             # Fraction Xl in filtrate flow
parValue['alpha_G'] = eps              # Fraction G in filtrate flow
parValue['alpha_Gn'] = eps             # Fraction Gn in filtrate flow
parValue['alpha_L'] = eps              # Fraction L in filtrate flow
parValue['alpha_N'] = eps              # Fraction N in filtrate flow
parValue['alpha_Pr'] = eps             # Fraction Pr in filtrate flow

parValue['G_in']  =  15.0              # mM
parValue['Gn_in']  =  11.0             # mM

parValue['samplePeriod'] = 1           # h 
parValue['mu_ref'] = 0.030             # 1/h 
parValue['t1'] = 70.0                  # h      
parValue['F1'] = 0.0020                # L/h
parValue['t2'] = 500.0                 # h      
parValue['F2'] = 0.0300                # L/h

parLocation = {}
parLocation['V_start'] = 'bioreactor.V_start'
parLocation['VXv_start'] = 'bioreactor.m_start[1]'
parLocation['VXd_start'] = 'bioreactor.m_start[2]'
parLocation['VXl_start'] = 'bioreactor.m_start[3]'
parLocation['VG_start'] = 'bioreactor.m_start[4]'
parLocation['VGn_start'] = 'bioreactor.m_start[5]'
parLocation['VL_start'] = 'bioreactor.m_start[6]'
parLocation['VN_start'] = 'bioreactor.m_start[7]'

parLocation['qG_max1'] = 'bioreactor.culture.qG_max1'
parLocation['qG_max2'] = 'bioreactor.culture.qG_max2'
parLocation['qGn_max1'] = 'bioreactor.culture.qGn_max1'
parLocation['qGn_max2'] = 'bioreactor.culture.qGn_max2'
parLocation['mu_d_max'] = 'bioreactor.culture.mu_d_max'
parLocation['k_toxic'] = 'bioreactor.culture.k_toxic'
parLocation['alpha'] = 'bioreactor.culture.alpha'
parLocation['beta'] = 'bioreactor.culture.beta'

parLocation['k_lysis_v'] = 'bioreactor.broth_decay.k_lysis_v'
parLocation['k_lysis_d'] = 'bioreactor.broth_decay.k_lysis_d'

parLocation['eps'] = 'filter.eps' 
parLocation['alpha_Xv'] = 'filter.alpha[1]' 
parLocation['alpha_Xd'] = 'filter.alpha[2]'
parLocation['alpha_Xl'] = 'filter.alpha[3]'
parLocation['alpha_G'] = 'filter.alpha[4]'
parLocation['alpha_Gn'] = 'filter.alpha[5]'
parLocation['alpha_L'] = 'filter.alpha[6]'
parLocation['alpha_N'] = 'filter.alpha[7]'
parLocation['alpha_Pr'] = 'filter.alpha[8]'

parLocation['G_in'] = 'feedtank.c_in[4]'
parLocation['Gn_in'] = 'feedtank.c_in[5]'

parLocation['samplePeriod'] = 'cspr_openloop.samplePeriod'     
parLocation['mu_ref'] = 'cspr_openloop.mu_ref'       
parLocation['t1'] = 'cspr_openloop.t1'                
parLocation['F1'] = 'cspr_openloop.F1'         
parLocation['t2'] = 'cspr_openloop.t2'                
parLocation['F2'] = 'cspr_openloop.F2'     

# Extra only for describe()
parLocation['mu'] = 'bioreactor.culture.mu'
parLocation['mu_d'] = 'bioreactor.culture.mu_d'  

# Parameter value check - especially for hysteresis to avoid runtime error
parCheck = []
parCheck.append("parValue['V_start'] > 0")
parCheck.append("parValue['VXv_start'] >= 0")
parCheck.append("parValue['VG_start'] >= 0")
parCheck.append("parValue['VGn_start'] >= 0")
parCheck.append("parValue['VL_start'] >= 0")
parCheck.append("parValue['VN_start'] >= 0")
parCheck.append("parValue['t2'] >= parValue['t1']")

# Create list of diagrams to be plotted by simu()
diagrams = []

# Create an empty list axes to be defined in newplot() and plotted by simu() or show()
ax = []

# Create list of pens for the diagrams
lines = ['-','--',':','-.']

