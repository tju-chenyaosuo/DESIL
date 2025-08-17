func.func @func1() {
  %base = memref.alloca() : memref<8xf32>
  %i = index.constant 1
  %mask = vector.create_mask %i : vector<8xi1>
  %0 = arith.constant 0.0 : f32
  %value = vector.broadcast %0 : f32 to vector<8xf32>
  vector.compressstore %base[%i], %mask, %value : memref<8xf32>, vector<8xi1>, vector<8xf32>
  return
}