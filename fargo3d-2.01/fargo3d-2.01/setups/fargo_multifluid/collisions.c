//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void _collisions_cpu(real dt, int id1, int id2, int id3, int option) {
  
//<USER_DEFINED>

  real *rho[NFLUIDS];
//  real *rho0[NFLUIDS];
  real *velocities_input[NFLUIDS];
  real *velocities_output[NFLUIDS];

  int ii;

  for (ii=0; ii<NFLUIDS; ii++) {

    INPUT(Fluids[ii]->Density);
    rho[ii]  = Fluids[ii]->Density->field_cpu;

//    INPUT2D(Fluids[ii]->Density0);
//    rho0[ii]  = Fluids[ii]->Density0->field_cpu;
    
    //Collisions along X
    #ifdef X
    if (id1 == 1) {
      if (option == 1) {
	INPUT(Fluids[ii]->Vx_temp);
	OUTPUT(Fluids[ii]->Vx_temp);
	velocities_input[ii] = Fluids[ii]->Vx_temp->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vx_temp->field_cpu;
      }
      if (option == 0) {
	INPUT(Fluids[ii]->Vx);
	OUTPUT(Fluids[ii]->Vx_half);
	velocities_input[ii] = Fluids[ii]->Vx->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vx_half->field_cpu;
      }
    }
    #endif
    
    //Collisions along Y
    #ifdef Y
    if (id2 == 1) {
      if (option == 1) {
	INPUT(Fluids[ii]->Vy_temp);
	OUTPUT(Fluids[ii]->Vy_temp);
	velocities_input[ii] = Fluids[ii]->Vy_temp->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vy_temp->field_cpu;
      }
      if (option == 0) {
	INPUT(Fluids[ii]->Vy);
	OUTPUT(Fluids[ii]->Vy_half);
	velocities_input[ii] = Fluids[ii]->Vy->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vy_half->field_cpu;
      }
    }
    #endif
    
    //Collisions along Z
    #ifdef Z
    if (id3 == 1) {
      if (option == 1) {
	INPUT(Fluids[ii]->Vz_temp);
	OUTPUT(Fluids[ii]->Vz_temp);
	velocities_input[ii] = Fluids[ii]->Vz_temp->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vz_temp->field_cpu;
      }
      if (option == 0) {
	INPUT(Fluids[ii]->Vz);
	OUTPUT(Fluids[ii]->Vz_half);
	velocities_input[ii] = Fluids[ii]->Vz->field_cpu;
	velocities_output[ii] = Fluids[ii]->Vz_half->field_cpu;
      }
    }
    #endif
  }
//<\USER_DEFINED>

//<EXTERNAL>
  int pitch  = Pitch_cpu;
  int stride = Stride_cpu;
  int pitch2d = Pitch2D;
  int size_x = XIP; 
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
  real* alpha = Alpha;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll;
//  int l2D;
  int o;
  int p;
  int q;
  int ir;
  int ir2;
  int ir_max;
  int ic;
  real max_value;
  real factor;
  real big;
  real temp;
  real sum;
  int idm;
  int idm2D;
  int lxm2D;
  int lym2D;
  int lzm2D;
  real b[NFLUIDS];
  real m[NFLUIDS*NFLUIDS];  
  real omega;
  real rho_p;
  real rho_o;
  real rho_q;
  real rho0_p;
  real rho0_o;
  real rho0_q;
  real St;
  real InvSt;
//<\INTERNAL>

//<CONSTANT>
// real Alpha(NFLUIDS*NFLUIDS);
//<\CONSTANT>

  
//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for(k=1; k<size_z; k++) {
#endif
#ifdef Y
    for(j=1; j<size_y; j++) {
#endif
#ifdef X
      for(i=XIM; i<size_x; i++) {
#endif
//<#>
	
//#include  "collision_kernel.h"
//#include  "gauss.h"
        if (ii == 1) InvSt = INVSTOKES1;
        if (ii == 2) InvSt = INVSTOKES2;
        if (ii == 3) InvSt = INVSTOKES3;
        if (ii == 4) InvSt = INVSTOKES4;
        if (ii == 5) InvSt = INVSTOKES5;
        if (ii == 6) InvSt = INVSTOKES6;
        if (ii == 7) InvSt = INVSTOKES7;
//        if (ii == 8) InvSt = INVSTOKES8;
//        if (ii == 9) InvSt = INVSTOKES9;
//        if (ii == 10) InvSt = INVSTOKES10;
        St = 1./InvSt*SIGMA0/rho[0][l]; 
        omega = (id1+id3)*sqrt(G*MSTAR/(ymed(j)*ymed(j)*ymed(j))) +
                 id2*sqrt(G*MSTAR/(ymin(j)*ymin(j)*ymin(j)));	
	for (o=0; o<NFLUIDS; o++) {
          if (o==0) {
           velocities_output[o][l] = velocities_output[o][l];
          }else{
           velocities_output[o][l] = velocities_input[o][l] + dt*omega/St*velocities_output[0][l];
           velocities_output[o][l] /= 1.0 + dt*omega/St;
          }
	}

//<\#>
#ifdef X
      }
#endif
#ifdef Y
    }
#endif
#ifdef Z
  }
#endif
//<\MAIN_LOOP>
}

void Collisions(real dt, int option) {

  //Input and output velocities are the same Fields
  if (option == 1) {
#ifdef X
    //Collisions along the X direction
    FARGO_SAFE(_collisions(dt,1,0,0,option));
#endif
#ifdef Y
    //Collisions along the Y direction
    FARGO_SAFE(_collisions(dt,0,1,0,option));
#endif
#ifdef Z
    //Collisions along the Z direction
    FARGO_SAFE(_collisions(dt,0,0,1,option));
#endif
  }
  
  //Input and output velocities are not the same Fields
  if (option == 0) {
#ifdef X
    //Collisions along the X direction
    FARGO_SAFE(_collisions(dt,1,0,0,option));
#endif
#ifdef Y
    //Collisions along the Y direction
    FARGO_SAFE(_collisions(dt,0,1,0,option));
#endif
#ifdef Z
    //Collisions along the Z direction
    FARGO_SAFE(_collisions(dt,0,0,1,option));
#endif
  }
}
