# SM=sm_86 SIMPLE_TS=16 REG_TS=64 MR=8 NR=8
# using nvcc to compile both .cpp and .cu files.
NVCC  ?= nvcc

SM ?= sm_80 # architecture dependent
BLOCK_SIZE ?= 16
SIMPLE_TS ?= 16 # 16 
REG_TS ?= 32 #64
MR ?= 4 #8
NR ?= 4 #8

NVCCFLAGS ?= -O3 -lineinfo -Xptxas -v -arch=$(SM)

INCDIR = include
SRCDIR = src
OBJDIR = obj
BINDIR = bin

SRCS_CPP = $(SRCDIR)/main.cpp $(SRCDIR)/utils.cpp
SRCS_CU  = $(SRCDIR)/gemm_naive.cu $(SRCDIR)/gemm_tiled.cu $(SRCDIR)/gemm_reg.cu $(SRCDIR)/gemm_reg_prefetch.cu

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
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -DBLOCK_SIZE=$(BLOCK_SIZE) -DREG_TS=$(REG_TS) -DMR=$(MR) -DNR=$(NR) -DSIMPLE_TS=$(SIMPLE_TS) -c $< -o $@

# compile
$(OBJDIR)/%.o: $(SRCDIR)/%.cu | $(OBJDIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) -DBLOCK_SIZE=$(BLOCK_SIZE) -DREG_TS=$(REG_TS) -DMR=$(MR) -DNR=$(NR) -DSIMPLE_TS=$(SIMPLE_TS) -c $< -o $@

# link
$(BINDIR)/gemm: $(OBJS) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@

.PHONY: run_naive
run_naive: all
	$(BINDIR)/gemm --mode naive --size 1024 --iters 10

.PHONY: run_shared
run_shared: all
	$(BINDIR)/gemm --mode shared --size 1024 --iters 10

.PHONY: run_reg
run_reg: all
	$(BINDIR)/gemm --mode reg --size 1024 --iters 10

.PHONY: run_reg_prefetch
run_reg_prefetch: all
	$(BINDIR)/gemm --mode regprefetch --size 1024 --iters 10

.PHONY: clean
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: distclean
distclean: clean
	rm -f *~ 

