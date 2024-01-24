#
# Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include makefiles/common.mk

BUILDDIR ?= $(abspath build)
OBJDIR := $(BUILDDIR)/device

LIBSRCFILES := src/functions.cu

LIBSRCFILES += src/all_gather.cu 
LIBSRCFILES += src/all_reduce.cu 


DEPFILES   := $(patsubst src/%.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
DEPENDFILES:= $(DEPFILES:%.d=%.dep)
STATICLIB  := $(OBJDIR)/libcolldevice.a
DEVOBJ     := $(OBJDIR)/devlink.o
RULESFILE  := $(OBJDIR)/Makefile.rules

NVCUFLAGS  += -Isrc -Iinclude --compiler-options "-fPIC -fvisibility=hidden"


all: $(STATICLIB)

# Dummy rule so that the extra dependency (%.dep) files are preserved by make
all_deps: $(DEPENDFILES)

# Auto-generating the rules per op/reduction/datatype/algorithm
$(RULESFILE) : gen_rules.sh
	@printf "Generating %-35s > %s\n" rules $@
	@mkdir -p $(OBJDIR)
	@CUDA_MAJOR=${CUDA_MAJOR} CUDA_MINOR=${CUDA_MINOR} ./gen_rules.sh $(OBJDIR) > $@

-include $(RULESFILE)

LIBOBJ     := $(GENOBJS) $(OBJDIR)/functions.o 

-include $(DEPFILES)

$(STATICLIB): $(LIBOBJ) $(DEVOBJ)
	@printf "Archiving  %-35s > %s\n" objects $@
	ar cr $@ $^

# We do not want make to build *.d when running make clean.
# So we only provide targets for .dep which will produce .dep and .d,
# with only .d being included, and .dep keeping track of what needs to
# be regenerated.
$(OBJDIR)/%.dep : src/%.cu
	@mkdir -p $(OBJDIR)
	@$(NVCC) $(NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

# Compiled kernels and collectives with relocatable device code ...
$(OBJDIR)/functions.o : src/functions.cu $(OBJDIR)/functions.dep
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(NVCC) $(NVCUFLAGS) -dc $< -o $@

# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)
	$(NVCC) $(NVCUFLAGS) -dlink $^ -o $@

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(RULESFILE) $(STATICLIB)
