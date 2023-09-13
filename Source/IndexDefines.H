#ifndef EBR_INDEX_H_
#define EBR_INDEX_H_

#define URHO  0
#define UMX   1
#define UMY   2
#define UMZ   3
#define UEDEN 4 // rho*(e + 1/2*U^2)
#define NCONS 5

#define QRHO   0
#define QU     1
#define QV     2
#define QW     3
#define QPRES  4
#define QT     5
#define QC     6
#ifdef CHEM
#define QGAMA  7
#define NPRIM  8
#else
#define NPRIM  7
#endif

#define NTHREADS 256 // not in use

#ifdef CHEM
#define NSPECS 9
#define NREACS 19
#define SPEC_START 0
#define SPEC_END NSPECS-1
#endif

#endif
