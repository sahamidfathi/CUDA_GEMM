GEMM Optimization Methods Used: tiling, register blocking, double buffering.

================================================================================
GEMM Optimization Method 1: Block-Tiling 
================================================================================
The N×N input matrices are partitioned into TS×TS tiles.

A threadblock is launched to compute a single TS×TS tile of the output matrix C. 

For example, N=64, TS = 16. Multiply row A10..A13 (k elems) by col B01..B31 to get C11.
Matrix A (64x64)           Matrix B (64x64)           Matrix C (64x64)
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A00|A01|A02|A03|          |B00|B01|B02|B03|          |C00|C01|C02|C03|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A10|A11|A12|A13|          |B10|B11|B12|B13|          |C10|C11|C12|C13|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A20|A21|A22|A23|          |B20|B21|B22|B23|          |C20|C21|C22|C23|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+
|A30|A31|A32|A33|          |B30|B31|B32|B33|          |C30|C31|C32|C33|
+---+---+---+---+          +---+---+---+---+          +---+---+---+---+


Each thread within the TS×TS threadblock is responsible for calculating one corresponding element in that output tile C.

To achieve data reuse, the kernel iterates through the K dimension, loading the necessary TS×TS tiles of A and B from slow Global Memory into fast Shared Memory (As and Bs).

Once the tiles are in shared memory, they are reused by all TS×TS threads in the block to perform the dot product calculation for their single C element.

Each thread accumulates the partial products across all K iterations until the final C element is complete.




================================================================================
GEMM Optimization Method 2: Block-Tiling with Register Blocking
================================================================================
This is an implementation of a highly optimized General Matrix Multiplication (GEMM) CUDA kernel, utilizing a multi-level tiling strategy.

The goal is to keep the required data as close as possible to the execution units (CUDA Cores) by moving it through the GPU's memory hierarchy: Global Memory (slowest) → Shared Memory (fast) → Registers (fastest).

Level 1 Optimization: Block-Tiling with Shared Memory 
The first level of optimization uses threadblocks to cooperatively load and compute large tiles of the output matrix C.

Partitioning: The N×N matrices (A, B, and C) are divided into square tiles of size TS×TS (e.g., 64×64).

Staging: A threadblock is launched to compute one TS×TS tile of C. The threads in the block work together to load the required A and B tiles from slow Global Memory into fast, on-chip Shared Memory (e.g., __shared__ float As[TS][TS + 1];).

Conflict Avoidance: The shared memory arrays are padded (e.g., TS + 1) to ensure bank conflicts are avoided during read access, preserving the high bandwidth of shared memory.

Level 2 Optimization: Register Blocking (Micro-Tiling)
The second and more aggressive optimization focuses on data reuse within a single thread, moving the core computation away from shared memory and into the thread's registers (the fastest storage available).

Micro-Tiles: Each individual thread is assigned to compute a small sub-block of the final output, called a micro-tile (e.g., MR×NR, like 8×8).

Register Accumulation: The partial results of this micro-tile are accumulated directly in a register array (float regs[MR][NR];), avoiding shared memory traffic for accumulation.

Small blocks of data are loaded from Shared Memory into temporary registers (avals, bvals), and this register-resident data is reused for multiple multiply-add operations (regs[i][j] += avals[i] * bvals[j];).

Below is the trace of a single thread. Assume TS = 64 (Shared Memory Tile Size) and MR = 2, NR = 2 (Micro-tile size).

Thread goal: Compute a 2×2 micro-tile of C.

Thread's Private Registers (regs[MR][NR]) initialized to 0.0f.
+-------+-------+
| 0.0f  | 0.0f  |  <- regs[0][0], regs[0][1]
+-------+-------+
| 0.0f  | 0.0f  |  <- regs[1][0], regs[1][1]
+-------+-------+

Inside the k-Loop:
The thread loads a column slice from the shared As tile into its private avals registers.
SHARED MEMORY (As[TS][TS+1])                    THREAD'S REGISTERS (avals[MR])
                                                +-------+
... As[row0_idx + 0][K_ITERATION] ...  ------->| avals[0]|
... As[row0_idx + 1][K_ITERATION] ...  ------->| avals[1]|
                                                +-------+
The thread loads a row slice from the shared Bs tile into its private bvals registers.
SHARED MEMORY (Bs[TS][TS+1])                    THREAD'S REGISTERS (bvals[NR])
                                                +-------+-------+
Bs[K_ITERATION][col0_idx + 0] ...  ----------->| bvals[0]|bvals[1]|
Bs[K_ITERATION][col0_idx + 1] ...  -----------> +-------+-------+

Perform Multiply-Adds (Using only Registers)

Now, with avals and bvals in registers, the thread performs the core accumulation using these values and its regs accumulator.

THREAD'S REGISTERS (avals[MR])                 THREAD'S REGISTERS (bvals[NR])
+-------+                                       +-------+-------+
|avals[0]|                                     |bvals[0]|bvals[1]|
+-------+                                       +-------+-------+
|avals[1]|
+-------+
    |                                                 |
    v (Multiplied and Added)                          v
    ---------------------------------------------------
                            |
                            v
THREAD'S PRIVATE REGISTERS (regs[MR][NR] - Accumulator)
(Update for current K_ITERATION)
+-------------------------------------------------------------+
| regs[0][0] += avals[0] * bvals[0]   | regs[0][1] += avals[0] * bvals[1] |
+-------------------------------------+---------------------------------+
| regs[1][0] += avals[1] * bvals[0]   | regs[1][1] += avals[1] * bvals[1] |
+-------------------------------------+---------------------------------+

Once the k-loop finishes, the regs array holds the final, complete MR×NR micro-tile for matrix C.

Thread's Private Registers (regs[MR][NR])
(Final accumulated values for C_micro)
+-------+-------+
| C_0_0 | C_0_1 |
+-------+-------+
| C_1_0 | C_1_1 |
+-------+-------+


================================================================================
GEMM Optimization Method 3: Block-Tiling with Register Blocking & Double Buffering
================================================================================
The matrix C is divided into TS×TS sub-blocks. Each sub-block is computed by a CUDA thread block.

Data for the current tile of A and B needed by the thread block is staged from global memory into shared memory (As_buf and Bs_buf are the two double-buffered arrays for matrix A and B).

Each individual thread computes a small MR×NR micro-tile of the C matrix. This data is stored in registers. The regs[_MR][_NR] array holds the thread's contribution.

Two shared memory buffers are used (As_buf[0] and As_buf[1]). While the threads are computing using data from the current buffer (cur), other threads are simultaneously fetching the next tile of data into the other buffer (next). This hides memory latency.





