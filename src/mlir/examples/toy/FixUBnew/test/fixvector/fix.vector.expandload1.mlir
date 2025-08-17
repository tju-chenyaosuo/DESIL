func.func @func1() -> vector<8xf32> {
  %base = memref.alloca() : memref<8xf32>
  %i = index.constant 1
  %mask = vector.create_mask %i : vector<8xi1>
  %0 = arith.constant 0.0 : f32
  %pass_thru = vector.broadcast %0 : f32 to vector<8xf32>
  %res = vector.expandload %base[%i], %mask, %pass_thru : memref<8xf32>, vector<8xi1>, vector<8xf32> into vector<8xf32>
  return %res : vector<8xf32>
}