# SM=sm_86 SIMPLE_TS=16 REG_TS=64 MR=8 NR=8
# using nvcc to compile both .cpp and .cu files.
NVCC  ?= nvcc

SM ?= sm_80 # architecture dependent
SIMPLE_TS ?= 16
REG_TS ?= 64
MR ?= 8
NR ?= 8

NVCCFLAGS ?= -O3 -lineinfo -Xptxas -v -arch=$(SM)

INCDIR = include
SRCDIR = src
OBJDIR = obj
BINDIR = bin

SRCS_CPP = $(SRCDIR)/main.cpp $(SRCDIR)/utils.cpp
SRCS_CU  = $(SRCDIR)/gemm_tiled.cu $(SRCDIR)/gemm_reg.cu

OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS_CPP)) \
       $(patsubst $(SRCDIR)/%.cu,$(OBJDIR)/%.o,$(SRCS_CU))

.PHONY: all
all: $(BINDIR)/gemm

$(OBJDIR):
	@mkdir -p $(OBJDIR)

$(BINDIR):
	@mkdir -p $(BINDIR)

# compile macros
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -DREG_TS=$(REG_TS) -DMR=$(MR) -DNR=$(NR) -DSIMPLE_TS=$(SIMPLE_TS) -c $< -o $@

# compile
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -DREG_TS=$(REG_TS) -DMR=$(MR) -DNR=$(NR) -DSIMPLE_TS=$(SIMPLE_TS) -c $< -o $@

# link
$(BINDIR)/gemm: $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

.PHONY: run_simple
run_simple: all
	$(BINDIR)/gemm --mode simple --size 1024 --iters 10

.PHONY: run_reg
run_reg: all
	$(BINDIR)/gemm --mode reg --size 1024 --iters 10

.PHONY: debug
debug:
	@echo "NVCC = $(NVCC)"
	@echo "NVCCFLAGS = $(NVCCFLAGS)"
	@echo "SIMPLE_TS = $(SIMPLE_TS)  REG_TS = $(REG_TS) MR = $(MR) NR = $(NR)"

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: distclean
distclean: clean
	rm -f *~ 

