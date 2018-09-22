1.1 V9.1.85

1.2
	a) "GeForce GTX 1050 Ti", 6.1
	b) (6) Multiprocessors, (128) CUDA Cores/MP
	c) 4040 MBytes
	d) 65536, 49152 bytes

2.3
  squareArray: Used 5 registers, 328 bytes cmem[0] 
  addArrays: Used 8 registers, 348 bytes cmem[0]

3.3
  without memory operations: GPU - 1.85ms CPU - 49.15ms
  with memory operations: GPU - 3.34ms CPU - 50.72ms

3.4
  (8,4,3) 1.5416 ms
  (8,8,3) 1.5615 ms
  (32,8,3) 1.84985 ms
  (32,4,3) 1.8749 ms
  (8,4,3) seems to be optimal

5.5
  sigma	runtime
  1	16.40ms
  3	123.16ms
  5	323.29ms
  7	622.526ms

6.4
  memory	runtime
  global	123.16ms
  shared	82.12ms
  texture	444.73ms

9.6
  dt - large: the gradients and the image disappear
  N - large: the image gets an average value

9.7
  the result for linear diffusion and gaussian convolution are the same

9.8
  a) the resulting image is still blurred, however less than in gaussian case, edges are preserved
  b) the resulting image is not blurred, edges are preserved

12.2
  reduce0 0.355ms
  cublasSasum 0.00013ms

13.3
  global  0.3766ms
  shared  0.4127ms

  implementation did not improve the performance