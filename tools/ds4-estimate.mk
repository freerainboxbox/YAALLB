# Build YAALLB's ds4 footprint estimator inside a *built* ds4 tree.
#
# The estimator links ds4's own objects, so it inherits the exact shape/estimator
# code of that build (see tools/ds4_estimate.c and the ds4 provider docs in
# README.md). YAALLB builds it by itself on startup, for every ds4 directory
# config.json mentions, from inside that directory with an absolute -f; the ds4
# Makefile is read first, so CORE_OBJS/CFLAGS/link flags all come from it.
#
#   cd /path/to/ds4 && make -f /path/to/yaallb/tools/ds4-estimate.mk
#
# Produces ds4-estimate in the ds4 directory (where ds4-server also lives). It
# is additive: only ds4-estimate and ds4_estimate.host.o are written, and make
# does nothing when the tree is already current.
#
# The working-directory form matters: this fragment's own `include Makefile`,
# and ds4's object rules, are relative to the tree, so passing an absolute -f
# while staying in the tree is what keeps them meaning that tree.
#
# Variants:
#   - a tree built with `make cpu` has no GPU objects to link against; the
#     CPU-only switch swaps in the CPU object list (a command-line CORE_OBJS=
#     override would have to quote a make expression, so the switch does it):
#
#       cd /path/to/ds4 && make -f /path/to/yaallb/tools/ds4-estimate.mk \
#            DS4_ESTIMATE_CPU_ONLY=1
#
#   - DS4_ESTIMATE_OUT=<name> changes the output name.

DS4_ESTIMATE ?= ds4-estimate

# Where this fragment lives (the directory holding ds4_estimate.c).
DS4_ESTIMATE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

include Makefile

# ds4's GPU/CUDA/ROCm trees link through their own toolchain (DS4_LINK), which
# is empty for the plain Metal/CPU host builds.
DS4_ESTIMATE_LINK ?= $(DS4_LINK)
ifeq ($(strip $(DS4_ESTIMATE_LINK)),)
DS4_ESTIMATE_LINK := $(CC)
endif
DS4_ESTIMATE_LIBS ?= $(DS4_LINK_LIBS)
ifeq ($(strip $(DS4_ESTIMATE_LIBS)),)
DS4_ESTIMATE_LIBS := $(METAL_LDLIBS)
endif
DS4_ESTIMATE_CFLAGS := $(filter-out -std=c99,$(CFLAGS)) -I.
ifneq ($(strip $(DS4_ESTIMATE_CPU_ONLY)),)
DS4_ESTIMATE_CFLAGS += -DDS4_ESTIMATE_CPU_ONLY
# No GPU objects to link, so link the CPU core instead of the default one.
CORE_OBJS := $(CPU_CORE_OBJS)
endif

# ds4's Makefile is included first, so its `all:` goal would otherwise win.
.DEFAULT_GOAL := $(DS4_ESTIMATE)

$(DS4_ESTIMATE): ds4_estimate.host.o $(CORE_OBJS)
	$(DS4_ESTIMATE_LINK) -o $@ $^ $(DS4_ESTIMATE_LIBS)

ds4_estimate.host.o: $(DS4_ESTIMATE_DIR)ds4_estimate.c ds4.h
	$(CC) $(DS4_ESTIMATE_CFLAGS) -c -o $@ $(DS4_ESTIMATE_DIR)ds4_estimate.c
