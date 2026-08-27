# setup application functions BPL_CHO_Perfusion_cspr_openloop, dependent on previous import from fmu_explore 
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2026-08-27 - Created
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Specific application functions: newplot(), describe()
#------------------------------------------------------------------------------------------------------------------

def newplot(title='Perfusion cultivation',  plotType='TimeSeries'):
   """ Standard plot window,
        title = '' """
   
   # Reset pens
   resetPen()
   
   # Plot diagram 
   if plotType == 'TimeSeries':
      
      ax11 = plt.subplot(6,2,1);  ax12 = plt.subplot(6,2,2)
      ax21 = plt.subplot(6,2,3);  ax22 = plt.subplot(6,2,4)    
      ax31 = plt.subplot(6,2,5);  ax32 = plt.subplot(6,2,6) 
      ax41 = plt.subplot(6,2,7);  ax42 = plt.subplot(6,2,8) 
      ax51 = plt.subplot(6,2,9);  ax52 = plt.subplot(6,2,10) 
      ax61 = plt.subplot(6,2,11); ax62 = plt.subplot(6,2,12) 
      
      ax.clear()
      ax.append(ax11)  #  0
      ax.append(ax12)  #  1
      ax.append(ax21)  #  2
      ax.append(ax22)  #  3
      ax.append(ax31)  #  4
      ax.append(ax32)  #  5
      ax.append(ax41)  #  6
      ax.append(ax42)  #  7     
      ax.append(ax51)  #  6
      ax.append(ax52)  #  9         
      ax.append(ax61)  # 10
      ax.append(ax62)  # 11         
      
      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('G [mM]')

      ax[1].grid()
      ax[1].set_ylabel('L [mM]')

      ax[2].grid()
      ax[2].set_ylabel('Gn[mM]')

      ax[3].grid()
      ax[3].set_ylabel('N [mM]')

      ax[4].grid()
      ax[4].set_ylabel('Xv [1E6/mL]')

      ax[5].grid()
      ax[5].set_ylabel('Xd [1E6/mL]')

      ax[6].grid()
      ax[6].set_ylabel('mu [1/h]')

      ax[7].grid()
      ax[7].set_ylabel('mu_d [1/h]')

      ax[8].grid()
      ax[8].set_ylabel('Fh*Xvh [g/h]')

      ax[9].grid()
      ax[9].set_ylim([0,0.5])
      ax[9].set_ylabel('V reactor [L]')

      ax[10].grid()
      ax[10].set_ylabel('F [L/h]')
      ax[10].set_xlabel('Time [h]')

      ax[11].grid()
      ax[11].set_ylabel('V harvest [L]')
      ax[11].set_xlabel('Time [h]')

      diagrams.clear()
      diagrams.append("ax[0].plot(t,sim_res['bioreactor.c[4]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[1].plot(t,sim_res['bioreactor.c[6]'], color='r', linestyle=linetype)")   
      diagrams.append("ax[2].plot(t,sim_res['bioreactor.c[5]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[3].plot(t,sim_res['bioreactor.c[7]'], color='r', linestyle=linetype)")  
      diagrams.append("ax[4].plot(t,sim_res['bioreactor.c[1]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[5].plot(t,sim_res['bioreactor.c[2]'], color='r', linestyle=linetype)")  
      diagrams.append("ax[6].plot(t,sim_res['bioreactor.culture.q[1]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[7].plot(t,sim_res['bioreactor.culture.q[2]'], color='r', linestyle=linetype)")  
      diagrams.append("ax[8].plot(t,sim_res['harvesttank.inlet.F']*sim_res['harvesttank.inlet.c[1]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[9].plot(t,sim_res['bioreactor.V'], color='b', linestyle=linetype)")  
      diagrams.append("ax[10].step(t,sim_res['bioreactor.inlet[1].F'], color='b', where='post', linestyle=linetype)")       
      diagrams.append("ax[11].plot(t,sim_res['harvesttank.V'], color='b', linestyle=linetype)")  

   if plotType == 'Extended':

      ax11 = plt.subplot(8,2,1);  ax12 = plt.subplot(8,2,2)
      ax21 = plt.subplot(8,2,3);  ax22 = plt.subplot(8,2,4)    
      ax31 = plt.subplot(8,2,5);  ax32 = plt.subplot(8,2,6) 
      ax41 = plt.subplot(8,2,7);  ax42 = plt.subplot(8,2,8) 
      ax51 = plt.subplot(8,2,9);  ax52 = plt.subplot(8,2,10) 
      ax61 = plt.subplot(8,2,11); ax62 = plt.subplot(8,2,12) 
      ax71 = plt.subplot(8,2,13); ax72 = plt.subplot(8,2,14) 
      ax81 = plt.subplot(8,2,15); ax82 = plt.subplot(8,2,16)  

      ax.clear()
      ax.append(ax11)  #  0
      ax.append(ax12)  #  1
      ax.append(ax21)  #  2
      ax.append(ax22)  #  3
      ax.append(ax31)  #  4
      ax.append(ax32)  #  5
      ax.append(ax41)  #  6
      ax.append(ax42)  #  7     
      ax.append(ax51)  #  6
      ax.append(ax52)  #  9         
      ax.append(ax61)  # 10
      ax.append(ax62)  # 11 
      ax.append(ax71)  # 12
      ax.append(ax72)  # 13         
      ax.append(ax81)  # 14
      ax.append(ax82)  # 15                   

      ax[0].set_title(title)
      ax[0].grid()
      ax[0].set_ylabel('G [mM]')

      ax[1].grid()
      ax[1].set_ylabel('L [mM]')

      ax[2].grid()
      ax[2].set_ylabel('Gn[mM]')

      ax[3].grid()
      ax[3].set_ylabel('N [mM]')

      ax[4].grid()
      ax[4].set_ylabel('qG_ind_over')

      ax[5].grid()
      ax[5].set_ylabel('qGn_ind_over')

      ax[6].grid()
      ax[6].set_ylabel('Xv [1E6/mL]')

      ax[7].grid()
      ax[7].set_ylabel('Xd [1E6/mL]')

      ax[8].grid()
      ax[8].set_ylabel('mu_v [1/h]')

      ax[9].grid()
      ax[9].set_ylabel('mu_d [1/h]')

      ax[10].grid()
      ax[10].set_ylabel('Fh*Xvh [g/h]')

      ax[11].grid()
      ax[11].set_ylim([0,0.5])
      ax[11].set_ylabel('V reactor [L]')

      ax[12].grid()
      ax[12].set_ylabel('F [L/h]')

      ax[13].grid()
      ax[13].set_ylabel('V harvest [L]')

      ax[14].grid()
      ax[14].set_ylabel('CSPR |pL/cell/day')
      ax[14].set_xlabel('Time [h]')        

      ax[15].grid()
      ax[15].set_ylabel('CSPR')
      ax[15].set_xlabel('Time [h]')          

      # List of commands to be executed by simu() after a simulation  
      diagrams.clear()
      diagrams.append("ax[0].plot(t,sim_res['bioreactor.c[4]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[1].plot(t,sim_res['bioreactor.c[6]'], color='r', linestyle=linetype)")   
      diagrams.append("ax[2].plot(t,sim_res['bioreactor.c[5]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[3].plot(t,sim_res['bioreactor.c[7]'], color='r', linestyle=linetype)")
      diagrams.append("ax[4].plot(t,sim_res['bioreactor.culture.Ind_qG_over'], color='g', linestyle=linetype)")       
      diagrams.append("ax[5].plot(t,sim_res['bioreactor.culture.Ind_qGn_over'], color='g', linestyle=linetype)")    
      diagrams.append("ax[6].plot(t,sim_res['bioreactor.c[1]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[7].plot(t,sim_res['bioreactor.c[2]'], color='r', linestyle=linetype)")  
      diagrams.append("ax[8].plot(t,sim_res['bioreactor.culture.q[1]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[9].plot(t,sim_res['bioreactor.culture.q[2]'], color='r', linestyle=linetype)")  
      diagrams.append("ax[10].plot(t,sim_res['harvesttank.inlet.F']*sim_res['harvesttank.inlet.c[1]'], color='b', linestyle=linetype)")       
      diagrams.append("ax[11].plot(t,sim_res['bioreactor.V'], color='b', linestyle=linetype)")  
      diagrams.append("ax[12].step(t,sim_res['bioreactor.inlet[1].F'], color='b', linestyle=linetype)")       
      diagrams.append("ax[13].plot(t,sim_res['harvesttank.V'], color='b', linestyle=linetype)")  
      diagrams.append("ax[14].step(t,sim_res['CSPR'], color='g', linestyle=linetype)")       
      diagrams.append("ax[15].step(t,sim_res['CSPR'], color='g', linestyle=linetype)")  
      

   if plotType == 'Cytiva-18':
 
      # Plot diagram
      plt.figure()
      ax11 = plt.subplot(6,2,1);  ax12 = plt.subplot(6,2,2)
      ax21 = plt.subplot(6,2,3);  ax22 = plt.subplot(6,2,4)    
      ax31 = plt.subplot(6,2,5);  ax32 = plt.subplot(6,2,6) 
      ax41 = plt.subplot(6,2,7);  ax42 = plt.subplot(6,2,8) 
      ax51 = plt.subplot(6,2,9);  ax52 = plt.subplot(6,2,10) 
      ax61 = plt.subplot(6,2,11); ax62 = plt.subplot(6,2,12) 

      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('G [mM]')

      ax12.grid()
      ax12.set_ylabel('L [mM]')

      ax21.grid()
      ax21.set_ylabel('Gn[mM]')

      ax22.grid()
      ax22.set_ylabel('N [mM]')

      ax31.grid()
      ax31.set_ylabel('Xv [1E6/mL]')

      ax32.grid()
      ax32.set_ylabel('Xd [1E6/mL]')

      ax41.grid()
      ax41.set_ylim([0,0.9])
      ax41.set_ylabel('mu [1/d]')

      ax42.grid()
      ax42.set_ylabel('V*Xv [1E9]')

      ax[8].grid()
      ax[8].set_ylabel('CSPR [pL/cell/d]')

      ax52.grid()
      ax52.set_ylim([0,11])
      ax52.set_ylabel('V reactor [L]')

      ax61.grid()
      ax61.set_ylabel('F [L/h]')
      ax61.set_xlabel('Time [d]')

      ax62.grid()
      ax62.set_ylabel('V harvest [L]')
      ax62.set_xlabel('Time [d]')

   if plotType == 'Cytiva-24':

      # Plot diagram
      plt.figure()
      ax11 = plt.subplot(6,2,1);  ax12 = plt.subplot(6,2,2)
      ax21 = plt.subplot(6,2,3);  ax22 = plt.subplot(6,2,4)    
      ax31 = plt.subplot(6,2,5);  ax32 = plt.subplot(6,2,6) 
      ax41 = plt.subplot(6,2,7);  ax42 = plt.subplot(6,2,8) 
      ax51 = plt.subplot(6,2,9);  ax52 = plt.subplot(6,2,10) 
      ax61 = plt.subplot(6,2,11); ax62 = plt.subplot(6,2,12) 

      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('G [mM]')

      ax12.grid()
      ax12.set_ylabel('L [mM]')

      ax21.grid()
      ax21.set_ylabel('Gn[mM]')

      ax22.grid()
      ax22.set_ylabel('N [mM]')

      ax31.grid()
      ax31.set_ylabel('Xv [1E6/mL]')

      ax32.grid()
      ax32.set_ylabel('Xd [1E6/mL]')

      ax41.grid()
      ax41.set_ylim([0,0.9])
      ax41.set_ylabel('mu [1/d]')

      ax42.grid()
      ax42.set_ylabel('V*Xv [1E9]')

      ax[8].grid()
      ax[8].set_ylabel('CSPR [pL/cell/d]')

      ax[9].grid()
      ax[9].set_ylim([0,30])
      ax[9].set_ylabel('V reactor [L]')

      ax61.grid()
      ax61.set_ylim([0,2.5])
      ax61.set_ylabel('F [L/h]')
      ax61.set_xlabel('Time [d]')

      ax62.grid()
      ax62.set_ylabel('V harvest [L]')
      ax62.set_xlabel('Time [d]')

def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""

   if name == 'culture':
      print('Reactor culture CHO-MAb - cell line HB-58 American Culture Collection ATCC') 

   elif name in ['broth', 'liquidphase', 'liquid-phase''media']:

      Xv  = model.get('liquidphase.Xv')[0]; Xv_description = model.get_variable_description('liquidphase.Xv'); Xv_mw = model.get('liquidphase.mw[1]')[0]
      Xd = model.get('liquidphase.Xd')[0]; Xd_description = model.get_variable_description('liquidphase.Xd'); Xd_mw = model.get('liquidphase.mw[2]')[0]
      Xl = model.get('liquidphase.Xl')[0]; Xl_description = model.get_variable_description('liquidphase.Xl'); Xl_mw = model.get('liquidphase.mw[2]')[0]

      G = model.get('liquidphase.G')[0]; G_description = model.get_variable_description('liquidphase.G'); G_mw = model.get('liquidphase.mw[3]')[0]
      Gn = model.get('liquidphase.Gn')[0]; Gn_description = model.get_variable_description('liquidphase.Gn'); Gn_mw = model.get('liquidphase.mw[4]')[0]
      L = model.get('liquidphase.L')[0]; L_description = model.get_variable_description('liquidphase.L'); L_mw = model.get('liquidphase.mw[5]')[0]
      N = model.get('liquidphase.N')[0]; N_description = model.get_variable_description('liquidphase.N'); N_mw = model.get('liquidphase.mw[6]')[0]
      Pr = model.get('liquidphase.Pr')[0]; Pr_description = model.get_variable_description('liquidphase.Pr'); Pr_mw = model.get('liquidphase.mw[7]')[0]

      print('Reactor broth substances included in the model')
      print()
      print(Xv_description, 'index = ', Xv, 'molecular weight = ', Xv_mw, 'Da')
      print(Xd_description, '  index = ', Xd, 'molecular weight = ', Xd_mw, 'Da')
      print(Xl_description, ' index = ', Xl, 'molecular weight = ', Xl_mw, 'Da')
      print(G_description, '     index = ', G, 'molecular weight = ', G_mw, 'Da')
      print(Gn_description, '   index = ', Gn, 'molecular weight = ', Gn_mw, 'Da')
      print(L_description, '     index = ', L, 'molecular weight = ', L_mw, 'Da')
      print(N_description, '     index = ', N, 'molecular weight = ', N_mw, 'Da')
      print(Pr_description, '     index = ', Pr, 'molecular weight = ', Pr_mw, 'Da')

   elif name in ['parts']:
      describe_parts(component_list_minimum)

   elif name in ['MSL']:
      describe_MSL()

   else:
      describe_general(name, decimals)
      
#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

FMU_explore_info()


