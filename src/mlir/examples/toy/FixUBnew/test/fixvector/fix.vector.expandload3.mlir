func.func @func1() {
  %base = memref.alloca() : memref<1x6xf32>
  %i = index.constant 83
  %mask = vector.create_mask %i : vector<7xi1>
  %0 = arith.constant 85.96 : f32
  %value = vector.broadcast %0 : f32 to vector<7xf32>
  %idx0 = index.constant 84
  %idx1 = index.constant 93
  vector.compressstore %base[%idx0, %idx1], %mask, %value : memref<1x6xf32>, vector<7xi1>, vector<7xf32>
  return 
}