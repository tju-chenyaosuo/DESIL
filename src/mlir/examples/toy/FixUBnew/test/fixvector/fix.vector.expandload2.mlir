func.func @func1() {
  %memref_dim0 = index.constant 19
  %memref_dim1 = index.constant 87
  %base = memref.alloca(%memref_dim0, %memref_dim1) : memref<?x10x?x3xf32>
  %i = index.constant 75
  %mask = vector.create_mask %i : vector<5xi1>
  %0 = arith.constant 90.84 : f32
  %value = vector.broadcast %0 : f32 to vector<5xf32>
  %idx0 = index.constant 77
  %idx1 = index.constant 86
  %idx2 = index.constant 96
  %idx3 = index.constant 19
  vector.compressstore %base[%idx0, %idx1, %idx2, %idx3], %mask, %value : memref<?x10x?x3xf32>, vector<5xi1>, vector<5xf32>
  return 
}