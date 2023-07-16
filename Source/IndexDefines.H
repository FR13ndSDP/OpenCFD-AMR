#ifndef EBR_INDEX_H_
#define EBR_INDEX_H_

#define URHO  0
#define UMX   1
#define UMY   2
#define UMZ   3
#define UEDEN 4
#define NCONS 5

#define QRHO   0
#define QU     1
#define QV     2
#define QW     3
#define QPRES  4
#define NPRIM  5

#define NTHREADS 256

#ifdef CHEM
#define NSPECS 4 
#define SPEC_START 0
#define SPEC_END NSPECS-1
#endif

#endif
