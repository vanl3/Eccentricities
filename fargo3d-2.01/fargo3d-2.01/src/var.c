#define __LOCAL
#include "../src/fargo3d.h"
#undef __LOCAL

void InitVariables() {
  ///input b and lambda Variables
  ///init_var("B", (char*)&B, REAL, NO, "")
  ///init_var("LAMBDA", (char*)&LAMBDA, REAL, NO, "2.25")
  ///
  init_var("ALPHA", (char*)&ALPHA, REAL, NO, "0.0");
  init_var("AMBIPOLARDIFFUSIONCOEFF", (char*)&AMBIPOLARDIFFUSIONCOEFF, REAL, NO, "0.0");
  init_var("ASPECT", (char*)&ASPECT, STRING, NO, "auto");
  init_var("ASPECTRATIO", (char*)&ASPECTRATIO, REAL, NO, "0.1");
  init_var("AUTOCOLOR", (char*)&AUTOCOLOR, BOOL, NO, "1");
  init_var("BETA", (char*)&BETA, REAL, NO, "0.0");
  init_var("BETAPHI", (char*)&BETAPHI, REAL, NO, "35.0");
  init_var("CFL", (char*)&CFL, REAL, NO, "0.44");
  init_var("CMAP", (char*)&CMAP, STRING, NO, "magma");
  init_var("COLORBAR", (char*)&COLORBAR, BOOL, NO, "1");
  init_var("COORDINATES", (char*)&COORDINATES, STRING, NO, "standard");
  init_var("CS", (char*)&CS, REAL, NO, "1.0");
  init_var("DAMPINGZONE", (char*)&DAMPINGZONE, REAL, NO, "1.15");
  init_var("DISK", (char*)&DISK, BOOL, NO, "1");
  init_var("DT", (char*)&DT, REAL, NO, "0.314159265359");
  init_var("ECCENTRICITY", (char*)&ECCENTRICITY, REAL, NO, "0.0");
  init_var("ELSASSER", (char*)&ELSASSER, REAL, NO, "1.0e-4");
  init_var("ETAHAT", (char*)&ETAHAT, REAL, NO, "0.05");
  init_var("EXCLUDEHILL", (char*)&EXCLUDEHILL, BOOL, NO, "0");
  init_var("FIELD", (char*)&FIELD, STRING, NO, "gasdens");
  init_var("FLARINGINDEX", (char*)&FLARINGINDEX, REAL, NO, "0.0");
  init_var("FRAME", (char*)&FRAME, STRING, NO, "F");
  init_var("FUNCARCHFILE", (char*)&FUNCARCHFILE, STRING, NO, "std/func_arch.cfg");
  init_var("GAMMA", (char*)&GAMMA, REAL, NO, "1.66666667");
  init_var("HALLEFFECTCOEFF", (char*)&HALLEFFECTCOEFF, REAL, NO, "0.0");
  init_var("HDUST", (char*)&HDUST, REAL, NO, "1.0");
  init_var("INCLINATION", (char*)&INCLINATION, REAL, NO, "0.0");
  init_var("INDIRECTTERM", (char*)&INDIRECTTERM, BOOL, NO, "0");
  init_var("INNERDAMPINGZONE", (char*)&INNERDAMPINGZONE, REAL, NO, "-0.2");
  init_var("KILLINGBCCOLATITUDE", (char*)&KILLINGBCCOLATITUDE, REAL, NO, "-0.2");
  init_var("MASSTAPER", (char*)&MASSTAPER, REAL, NO, "0.0");
  init_var("METAL", (char*)&METAL, REAL, NO, "1.00001");
  init_var("NINTERM", (char*)&NINTERM, INT, NO, "200");
  init_var("NOISE", (char*)&NOISE, REAL, NO, "0.0");
  init_var("NSNAP", (char*)&NSNAP, INT, NO, "0");
  init_var("NTOT", (char*)&NTOT, INT, NO, "2001");
  init_var("NU", (char*)&NU, REAL, NO, "0.0");
  init_var("NX", (char*)&NX, INT, NO, "1");
  init_var("NY", (char*)&NY, INT, NO, "4096");
  init_var("NZ", (char*)&NZ, INT, NO, "1");
  init_var("OHMICDIFFUSIONCOEFF", (char*)&OHMICDIFFUSIONCOEFF, REAL, NO, "0.0");
  init_var("OMEGAFRAME", (char*)&OMEGAFRAME, REAL, NO, "1.0");
  init_var("OORTA", (char*)&OORTA, REAL, NO, "-0.75");
  init_var("ORBITALRADIUS", (char*)&ORBITALRADIUS, REAL, NO, "0.0");
  init_var("OUTERDAMPINGZONE", (char*)&OUTERDAMPINGZONE, REAL, NO, "-0.2");
  init_var("OUTPUTDIR", (char*)&OUTPUTDIR, STRING, NO, "@outputs/wind_unstrat");
  init_var("PERIODICY", (char*)&PERIODICY, BOOL, NO, "1");
  init_var("PERIODICZ", (char*)&PERIODICZ, BOOL, NO, "1");
  init_var("PLANETCONFIG", (char*)&PLANETCONFIG, STRING, NO, "planets/zero.cfg");
  init_var("PLANETMASS", (char*)&PLANETMASS, REAL, NO, "0.0");
  init_var("PLOTLINE", (char*)&PLOTLINE, STRING, NO, "field[:,:,0]");
  init_var("PLOTLOG", (char*)&PLOTLOG, BOOL, NO, "0");
  init_var("REALTYPE", (char*)&REALTYPE, STRING, NO, "Standard  #float or double");
  init_var("RELEASEDATE", (char*)&RELEASEDATE, REAL, NO, "0.0");
  init_var("RELEASERADIUS", (char*)&RELEASERADIUS, REAL, NO, "0.0");
  init_var("RESONANCE", (char*)&RESONANCE, REAL, NO, "0.5");
  init_var("RHOG0", (char*)&RHOG0, REAL, NO, "1.0");
  init_var("ROCHESMOOTHING", (char*)&ROCHESMOOTHING, REAL, NO, "0.0");
  init_var("SEMIMAJORAXIS", (char*)&SEMIMAJORAXIS, REAL, NO, "0.0");
  init_var("SETUP", (char*)&SETUP, STRING, NO, "wind_unstrat");
  init_var("SHEARPARAM", (char*)&SHEARPARAM, REAL, NO, "1.5");
  init_var("SIGMA0", (char*)&SIGMA0, REAL, NO, "6.3661977237e-4");
  init_var("SIGMASLOPE", (char*)&SIGMASLOPE, REAL, NO, "-0.5");
  init_var("SI_AMP", (char*)&SI_AMP, REAL, NO, "-1.0e-5");
  init_var("SI_KX", (char*)&SI_KX, REAL, NO, "30.0");
  init_var("SI_KZ", (char*)&SI_KZ, REAL, NO, "30.0");
  init_var("SI_SIG_IM", (char*)&SI_SIG_IM, REAL, NO, "0.42524459149582894");
  init_var("SI_SIG_RE", (char*)&SI_SIG_RE, REAL, NO, "0.36467805191244379");
  init_var("SPACING", (char*)&SPACING, STRING, NO, "uni");
  init_var("STOKES1", (char*)&STOKES1, REAL, NO, "0.1");
  init_var("TAUDAMP", (char*)&TAUDAMP, REAL, NO, "1.0e-2");
  init_var("THICKNESSSMOOTHING", (char*)&THICKNESSSMOOTHING, REAL, NO, "0.1");
  init_var("VERTICALDAMPING", (char*)&VERTICALDAMPING, REAL, NO, "0.0");
  init_var("VMAX", (char*)&VMAX, REAL, NO, "1.0");
  init_var("VMIN", (char*)&VMIN, REAL, NO, "0.0");
  init_var("VTK", (char*)&VTK, BOOL, NO, "0");
  init_var("WRITEBX", (char*)&WRITEBX, BOOL, NO, "0");
  init_var("WRITEBY", (char*)&WRITEBY, BOOL, NO, "0");
  init_var("WRITEBZ", (char*)&WRITEBZ, BOOL, NO, "0");
  init_var("WRITEDENSITY", (char*)&WRITEDENSITY, BOOL, NO, "0");
  init_var("WRITEDIVERGENCE", (char*)&WRITEDIVERGENCE, BOOL, NO, "0");
  init_var("WRITEENERGY", (char*)&WRITEENERGY, BOOL, NO, "0");
  init_var("WRITEENERGYRAD", (char*)&WRITEENERGYRAD, BOOL, NO, "0");
  init_var("WRITETAU", (char*)&WRITETAU, BOOL, NO, "0");
  init_var("WRITEVX", (char*)&WRITEVX, BOOL, NO, "0");
  init_var("WRITEVY", (char*)&WRITEVY, BOOL, NO, "0");
  init_var("WRITEVZ", (char*)&WRITEVZ, BOOL, NO, "0");
  init_var("XMAX", (char*)&XMAX, REAL, NO, "3.141592653589793");
  init_var("XMIN", (char*)&XMIN, REAL, NO, "-3.141592653589793");
  init_var("YMAX", (char*)&YMAX, REAL, NO, "1.6");
  init_var("YMIN", (char*)&YMIN, REAL, NO, "0.4");
  init_var("ZMAX", (char*)&ZMAX, REAL, NO, "0.05");
  init_var("ZMIN", (char*)&ZMIN, REAL, NO, "-0.05");
}
