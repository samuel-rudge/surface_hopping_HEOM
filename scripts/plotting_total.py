import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from config.input_parameters import *
from pathlib import Path
from mpmath import coth

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('axes', linewidth=2)
plt.rc('text.latex', preamble=r'\boldmath')

### LOADING INFORMATION ### 

repo = Path("/home/samuel/Documents/Postdoc/Projects/Project_SHEOM/Code/surface_hopping_HEOM/")
results_dir = repo / f"results/gamma_{Gamma_choice}eV/omega_{Vib_Freq_cl[0]}eV/prop_time_{(max_time * Vib_Freq_cl[0]):.2f}_Omegat/cgt_{(cgt * dt_init * Vib_Freq_cl[0]):.2f}_Omegat/traj_{n_trajectories}/"
results_dir.mkdir(parents=True, exist_ok=True)
transient_dir = results_dir / "transient"
transient_dir.mkdir(parents=True, exist_ok=True)
plots_dir = transient_dir / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)
transient_ensemble_av_dir = transient_dir / "ensemble_av"
transient_ensemble_av_dir.mkdir(parents=True, exist_ok=True)

mol_pops_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "mol_pops_cg_heom.dat")
mol_pops_tclme_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "mol_pops_tclme_cg_heom.dat")
ke_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "ke_cg_heom.dat")
pe_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "pe_cg_heom.dat")
ke_tclme_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "ke_tclme_cg_heom.dat")
pe_tclme_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "pe_tclme_cg_heom.dat")
transition_rate_tclme_heom_av = np.genfromtxt(transient_ensemble_av_dir / "transition_rate_tclme_heom.dat")
transition_rate_tclme_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "transition_rate_tclme_cg_heom.dat")
transition_rate_cg_heom_av = np.genfromtxt(transient_ensemble_av_dir / "transition_rate_tclme_cg_heom.dat")
tclme_cg_evidence_av  = np.genfromtxt(
        transient_ensemble_av_dir / "tclme_cg_evidence.dat"
)
cg_evidence_av  = np.genfromtxt(
        transient_ensemble_av_dir / "cg_evidence.dat"
)

mol_pops_cme_av = np.genfromtxt(transient_ensemble_av_dir / "mol_pops_cme.dat")
ke_cme_av = np.genfromtxt(transient_ensemble_av_dir / "ke_cme.dat")
pe_cme_av = np.genfromtxt(transient_ensemble_av_dir / "pe_cme.dat")
transition_rate_cme_av = np.genfromtxt(transient_ensemble_av_dir / "transition_rate_cme.dat")
transition_rate_tclme_cme_av = np.genfromtxt(transient_ensemble_av_dir / "transition_rate_tclme_cme.dat")

# Electronic populations #
fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle \rho_{ii}$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_cl[0]*mol_pops_cg_heom_av[:,0],mol_pops_tclme_cg_heom_av[:,1],color='red',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_cl[0]*mol_pops_cg_heom_av[:,0],mol_pops_tclme_cg_heom_av[:,2],color='red',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*mol_pops_cg_heom_av[:,0],mol_pops_cg_heom_av[:,1],color='green',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_cl[0]*mol_pops_cg_heom_av[:,0],mol_pops_cg_heom_av[:,2],color='green',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*mol_pops_cg_heom_av[:,0],mol_pops_cme_av[:,1],color='blue',linestyle='-',linewidth=2)
ax.plot(Vib_Freq_cl[0]*mol_pops_cg_heom_av[:,0],mol_pops_cme_av[:,2],color='blue',linestyle='--',linewidth=2)
occ_handles = [Line2D([0], [0], color='black', linestyle='-', label=r'$\displaystyle \rho_{00} $'),
                    Line2D([0], [0], color='black', linestyle='--', label=r'$\displaystyle \rho_{11} $')]
method_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \mbox{\textbf{CME}} $'),
                    Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \mbox{\textbf{TCL CG HEOM}} $'),
                    Line2D([0], [0], color='green', linestyle='-', label=r'$\displaystyle \mbox{\textbf{CG HEOM}} $')]
occ_legend = ax.legend(handles=occ_handles,loc='upper right',fontsize=18)
method_legend = ax.legend(handles=method_handles,loc='lower right',fontsize=18)
ax.add_artist(occ_legend)
ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(plots_dir / "mol_pops_heom.pdf")
# plt.show()

##### Vibrational Energies #######

fig, ax = plt.subplots(3,1,sharex=True,figsize=(8, 12),gridspec_kw={"hspace": 0}
    )
ax[0].set_ylabel(r"$\displaystyle \mbox{\textbf{CME}} \:\: E_{\mbox{\textbf{vib.}}} \: [\mbox{\textbf{eV}}]$",color='black',fontsize=18,fontweight='bold')
ax[1].set_ylabel(r"$\displaystyle \mbox{\textbf{TCL CG HEOM}} \:\: E_{\mbox{\textbf{vib.}}} \: [\mbox{\textbf{eV}}]$",color='black',fontsize=18,fontweight='bold')
ax[2].set_ylabel(r"$\displaystyle \mbox{\textbf{CG HEOM}} \:\: E_{\mbox{\textbf{vib.}}} \: [\mbox{\textbf{eV}}]$",color='black',fontsize=18,fontweight='bold')
ax[2].set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=18,fontweight='bold')
ax[0].tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax[0].tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax[1].tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax[1].tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax[2].tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax[2].tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax[0].plot(Vib_Freq_cl[0]*ke_cme_av[:,0],ke_cme_av[:,1],color='blue',linestyle='-',linewidth=2)
ax[0].plot(Vib_Freq_cl[0]*pe_cme_av[:,0],pe_cme_av[:,1],color='red',linestyle='--',linewidth=2)
ax[1].plot(Vib_Freq_cl[0]*ke_tclme_cg_heom_av[:,0],ke_tclme_cg_heom_av[:,1],color='blue',linestyle='-',linewidth=2)
ax[1].plot(Vib_Freq_cl[0]*pe_tclme_cg_heom_av[:,0],pe_tclme_cg_heom_av[:,1],color='red',linestyle='--',linewidth=2)
ax[2].plot(Vib_Freq_cl[0]*ke_cg_heom_av[:,0],ke_cg_heom_av[:,1],color='blue',linestyle='-',linewidth=2)
ax[2].plot(Vib_Freq_cl[0]*pe_cg_heom_av[:,0],pe_cg_heom_av[:,1],color='red',linestyle='--',linewidth=2)
ax[0].axhline(y = (Vib_Freq_cl[0]/4)*coth(Vib_Freq_cl[0]/(2*Temp)), color='g', linestyle='-')
ax[0].axhline(y=Temp / 2, color='g', linestyle='--')
ax[1].axhline(y = (Vib_Freq_cl[0]/4)*coth(Vib_Freq_cl[0]/(2*Temp)), color='g', linestyle='-')
ax[1].axhline(y=Temp / 2, color='g', linestyle='--')
ax[2].axhline(y = (Vib_Freq_cl[0]/4)*coth(Vib_Freq_cl[0]/(2*Temp)), color='g', linestyle='-')
ax[2].axhline(y=Temp / 2, color='g', linestyle='--')
energy_handles = [Line2D([0], [0], color='blue', linestyle='-', label=r'$\displaystyle \langle \mbox{\textbf{KE}} \rangle $'),
                Line2D([0], [0], color='red', linestyle='-', label=r'$\displaystyle \langle \mbox{\textbf{PE}} \rangle $'),
                Line2D([0], [0], color='green', linestyle='-', label=r'$\displaystyle \left(\frac{\Omega}{4}\right)\coth\left(\frac{\Omega}{2k_{B}T}\right) $'),
                Line2D([0], [0], color='green', linestyle='--', label=r'$\displaystyle k_{B}T/2 $')]
ax[0].legend(handles=energy_handles,loc='lower right',fontsize=18,frameon=False)
ax[0].set_xlim(0,Vib_Freq_cl[0]*max_time)
# ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(plots_dir / "vib_energy.pdf")
# plt.show()

##### Transition rates #####

fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle \langle T_{ab} \rangle(t)$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \omega t$",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_cl[0]*transition_rate_cme_av[:,0],transition_rate_cme_av[:,1],color='blue',linestyle='-',linewidth=2,
        label=r'$\displaystyle \mbox{\textbf{CME}} $')
ax.plot(Vib_Freq_cl[0]*transition_rate_cme_av[:,0],transition_rate_cme_av[:,2],color='blue',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*transition_rate_tclme_cg_heom_av[:,0],transition_rate_tclme_cg_heom_av[:,1],color='red',linestyle='-',linewidth=2,
        label=r'$\displaystyle \mbox{\textbf{TCL CG HEOM}} $')
ax.plot(Vib_Freq_cl[0]*transition_rate_tclme_cg_heom_av[:,0],transition_rate_tclme_cg_heom_av[:,2],color='red',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*transition_rate_tclme_heom_av[:,0],transition_rate_tclme_heom_av[:,1],color='green',linestyle='-',linewidth=2,
        label=r'$\displaystyle \mbox{\textbf{TCL HEOM}} $')
ax.plot(Vib_Freq_cl[0]*transition_rate_tclme_heom_av[:,0],transition_rate_tclme_heom_av[:,2],color='green',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*transition_rate_cg_heom_av[:,0],transition_rate_cg_heom_av[:,1],color='magenta',linestyle='-',linewidth=2,
        label=r'$\displaystyle \mbox{\textbf{TCL CG}} $')
ax.plot(Vib_Freq_cl[0]*transition_rate_cg_heom_av[:,0],transition_rate_cg_heom_av[:,2],color='magenta',linestyle='--',linewidth=2)
method_labels,method_handles = ax.get_legend_handles_labels()
method_legend = ax.legend(method_labels,method_handles,loc='lower left',fontsize=18)
rate_handles = [Line2D([0], [0], color='black', linestyle='-', label=r'$\displaystyle \langle T_{01} \rangle $'),
                    Line2D([0], [0], color='black', linestyle='--', label=r'$\displaystyle \langle T_{10} \rangle $')]
ax.legend(handles=rate_handles,loc='lower right',fontsize=18)
ax.add_artist(method_legend)
ax.set_xlim(0,Vib_Freq_cl[0]*max_time)
# ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(plots_dir / "transition_rate.pdf")
# plt.show()

#### COARSE-GRAINING EVIDENCE ####

fig, ax = plt.subplots()
ax.set_ylabel(r"$\displaystyle \langle T_{ab} \rangle(0,\tau)$",color='black',fontsize=24,fontweight='bold')
ax.set_xlabel(r"$\displaystyle \omega \tau$",color='black',fontsize=24,fontweight='bold')
ax.tick_params(axis='y', labelcolor='black',length=6, width=2,labelsize=20)
ax.tick_params(axis='x',labelcolor='black',length=6,width=2,labelsize=20)
ax.plot(Vib_Freq_cl[0]*tclme_cg_evidence_av[:,0],tclme_cg_evidence_av[:,1],color='red',linestyle='-',linewidth=2,
        label=r'$\displaystyle \mbox{\textbf{TCL CG HEOM}} $')
ax.plot(Vib_Freq_cl[0]*tclme_cg_evidence_av[:,0],tclme_cg_evidence_av[:,2],color='red',linestyle='--',linewidth=2)
ax.plot(Vib_Freq_cl[0]*cg_evidence_av[:,0],cg_evidence_av[:,1],color='green',linestyle='-',linewidth=2,
        label=r'$\displaystyle \mbox{\textbf{CG HEOM}} $')
ax.plot(Vib_Freq_cl[0]*cg_evidence_av[:,0],cg_evidence_av[:,2],color='green',linestyle='--',linewidth=2)
method_labels,method_handles = ax.get_legend_handles_labels()
method_legend = ax.legend(method_labels,method_handles,loc='center right',fontsize=18)
rate_handles = [Line2D([0], [0], color='black', linestyle='-', label=r'$\displaystyle \langle T_{01} \rangle $'),
                    Line2D([0], [0], color='black', linestyle='--', label=r'$\displaystyle \langle T_{10} \rangle $')]
ax.legend(handles=rate_handles,loc='lower right',fontsize=18)
ax.add_artist(method_legend)
ax.set_xlim(0,np.max(Vib_Freq_cl[0]*tclme_cg_evidence_av[:,0]))
# ax.set_ylim(0,1)
plt.tight_layout()
plt.savefig(plots_dir / "transition_rate.pdf")
plt.show()
