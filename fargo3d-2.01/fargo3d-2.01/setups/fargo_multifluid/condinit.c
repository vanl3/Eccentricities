#include "fargo3d.h"

void _CondInit() {
  
  int i,j,k;
  real r, omega;
  
  real *rho  = Density->field_cpu;
  real *vphi = Vx->field_cpu;
  real *vr   = Vy->field_cpu;
#ifdef ADIABATIC
  real *e   = Energy->field_cpu;
#endif
#ifdef ISOTHERMAL
  real *cs   = Energy->field_cpu;
#endif
 
  real rhog, rhod;
  real vk, soundspeed;
  
  i = j = k = 0;
  
  for (k=0; k<Nz+2*NGHZ; k++) {
    for (j=0; j<Ny+2*NGHY; j++) {
      for (i=0; i<Nx+2*NGHX; i++) {
	
	r     = Ymed(j);
	omega = sqrt(G*MSTAR/r/r/r);                       //Keplerian frequency
	rhog  = SIGMA0*pow(r/R0,-SIGMASLOPE);              //Gas surface density
        rhod  = rhog*EPSILON;                              //Dust surface density

	if (Fluidtype == GAS) {
	  rho[l]   = rhog;
	  vphi[l]  = omega*r*sqrt(1.0 + pow(ASPECTRATIO,2)*pow(r/R0,2*FLARINGINDEX)*
				  (2.0*FLARINGINDEX - 1.0 - SIGMASLOPE));
	  vr[l]    = 0.0;
          soundspeed  = ASPECTRATIO*pow(r/R0,FLARINGINDEX)*omega*r;
#ifdef ISOTHERMAL
	  cs[l]    = soundspeed;
#endif
#ifdef ADIABATIC
          e[l] = pow(soundspeed,2)*rho[l]/(GAMMA-1.0);
#endif
	}
	
	if (Fluidtype == DUST) {
	  rho[l]  = rhod;
	  vphi[l] = omega*r;
	  vr[l]   = 0.0;
#ifdef ISOTHERMAL
	  cs[l]   = 0.0;
#endif
#ifdef ADIABATIC
	  e[l]   = 0.0;
#endif
	}
	
	vphi[l] -= OMEGAFRAME*r;
	
      }
    }
  }
}

void CondInit() {
  
  int id_gas = 0;
  int feedback = NO;
  //We first create the gaseous fluid and store it in the array Fluids[]
  Fluids[id_gas] = CreateFluid("gas",GAS);

  //We now select the fluid
  SelectFluid(id_gas);

  //and fill its fields
  _CondInit();

  //We repeat the process for the dust fluids
  char dust_name[MAXNAMELENGTH];
  int id_dust;

  for(id_dust = 1; id_dust<NFLUIDS; id_dust++) {
    sprintf(dust_name,"dust%d",id_dust); //We assign different names to the dust fluids

    Fluids[id_dust]  = CreateFluid(dust_name, DUST);
    SelectFluid(id_dust);
    _CondInit();

  }

  /*We now fill the collision matrix (Feedback from dust included)
   Note: ColRate() moves the collision matrix to the device.
   If feedback=NO, gas does not feel the drag force.*/
  
  ColRate(INVSTOKES1, id_gas, 1, feedback);
  ColRate(INVSTOKES2, id_gas, 2, feedback);
  ColRate(INVSTOKES3, id_gas, 3, feedback);
  ColRate(INVSTOKES4, id_gas, 4, feedback);
  ColRate(INVSTOKES5, id_gas, 5, feedback);
  ColRate(INVSTOKES6, id_gas, 6, feedback);
  ColRate(INVSTOKES7, id_gas, 7, feedback);

}
