AMREX_HOME = ../../Submodules/amrex

DEBUG = FALSE
DIM = 3
COMP = gcc
PRECISION = DOUBLE
USE_MPI = FALSE
USE_OMP = TRUE
USE_CUDA = FALSE

BL_NO_FORT = TRUE

programs ?=

ifeq ($(strip $(programs)),)
  programs += fboxinfo
  programs += fcompare
  programs += fextract
  programs += fextrema
  programs += fgradient
  programs += fnan
  programs += fsnapshot
  programs += ftime
  programs += fvarnames
  programs += fvolumesum
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

multiple_executables = $(addsuffix .$(machineSuffix).ex, $(programs))
default: $(multiple_executables)

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

clean::
	$(SILENT) $(RM) $(multiple_executables)
