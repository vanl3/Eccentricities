//<FLAGS>
//#define __GPU
//#define __NOPROTO
//<\FLAGS>

//<INCLUDES>
#include "fargo3d.h"
//<\INCLUDES>

void UpdateDensityX_cpu(real dt, Field *Q, Field *Vx_t) {     ///void f(P) - Return value is absent

//<USER_DEFINED>
  INPUT(Q);         ///so Q = Vx_t?, input to copy Q from cpu to gpu, defining Q for later use
  INPUT(Vx_t);
  INPUT(DensStar);
  OUTPUT(Q);        ///why there are no output of Vx_t and DensStar? output to copy back to cpu
//<\USER_DEFINED>

//<EXTERNAL>
  real* qb = Q->field_cpu;            ///what is the -> for? are we defining qb here? A: give properties of field for example
  real* vx = Vx_t->field_cpu;         ///if we are defining values here then what is the diff between user define and external?
  real* rho_s = DensStar->field_cpu;  ///if run on gpu, cpu will auto change to gpu
  int pitch  = Pitch_cpu;             ///array
  int stride = Stride_cpu;
  int size_x = XIP;                   ///X Interval Plus (we need to address right neighbor in loop) as defined in define.h
  int size_y = Ny+2*NGHY;             ///dont know what NGHY is, does not say in define.h, use capital to know it was defined in .h
  int size_z = Nz+2*NGHZ;             ///why Y and Z are not defined the same as X?, GH is Ghost cells
//<\EXTERNAL>

//<INTERNAL>
  int i; //Variables reserved
  int j; //for the topology
  int k; //of the kernels
  int ll;
  int llxp;
//<\INTERNAL>

//<CONSTANT>
// real Sxj(Ny+2*NGHY);
// real Syj(Ny+2*NGHY);
// real Szj(Ny+2*NGHY);
// real Sxk(Nz+2*NGHZ);
// real Syk(Nz+2*NGHZ);
// real Szk(Nz+2*NGHZ);
// real InvVj(Ny+2*NGHY);
//<\CONSTANT>

//<MAIN_LOOP>

  i = j = k = 0;

#ifdef Z
  for (k=0; k<size_z; k++) {
#endif
#ifdef Y
    for (j=0; j<size_y; j++) {      ///Y is for radius
#endif
#ifdef X
      for (i=0; i<size_x; i++) {    ///cant define azimuthal as X, finite diff: taking derivative
#endif
//<#>               ///l: Theindex of the current zone
	ll = l;          ///if ll = l then why dont we just use ll or l?, maybe bc of the gpu code cant define l?
	llxp = lxp;      ///lxp,lxm:lxplus/lxminnus,theright/leftx-neighbor

	qb[ll] += ((vx[ll]*rho_s[ll] -	\     ///what is "\" sign for?
		   vx[llxp]*rho_s[llxp])*	\         ///what is this eq for?
		  SurfX(j,k)*dt*InvVol(j,k));       ///Surf for surface?  dont understand this
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
