//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void Floor_cpu() {

//<USER_DEFINED>
  INPUT(Density);
  INPUT2D(Density0);
  OUTPUT(Density);
#ifdef ADIABATIC
  INPUT(Energy);
  INPUT2D(Energy0);
  OUTPUT(Energy);
#endif
//<\USER_DEFINED>


//<EXTERNAL>
  real* dens = Density->field_cpu;
  real* dens0 = Density0->field_cpu;
#ifdef ADIABATIC
  real* eng = Energy->field_cpu;
  real* eng0 = Energy0->field_cpu;
#endif
  int pitch  = Pitch_cpu;
  int pitch2d = Pitch2D;
  int stride = Stride_cpu;
  int size_x = Nx+2*NGHX;
  int size_y = Ny+2*NGHY;
  int size_z = Nz+2*NGHZ;
//<\EXTERNAL>

//<INTERNAL>
  int i;
  int j;
  int k;
  int ll, ll2D;
  real dens_org;
//<\INTERNAL>

//<CONSTANT>
//  real DFLOOR(1);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {
#endif
#ifdef X
      for (i=0; i<size_x; i++ ) {
#endif
//<#>

        ll = l;
	ll2D = l2D;
//        ll2D = j; // initial midplane density at the same (spherical) radius
        dens_org = dens[ll];
        if (dens[ll] < dens0[ll2D]*DFLOOR){
          dens[ll] = dens0[ll2D]*DFLOOR;
//	if (dens[ll] < DFLOOR)
//	  dens[ll] = DFLOOR;
#ifdef ADIABATIC
          eng[ll]  = eng[ll]/dens_org*dens[ll];
#endif
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
