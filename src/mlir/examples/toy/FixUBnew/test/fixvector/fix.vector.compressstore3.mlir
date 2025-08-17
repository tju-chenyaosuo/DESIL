func.func @func1() {
  %memref_dim0 = index.constant 41
  %memref_dim1 = index.constant 72
  %memref_dim2 = index.constant 32
  %base = memref.alloca(%memref_dim0, %memref_dim1, %memref_dim2) : memref<?x3x?x?xf32>
  %i = index.constant 25
  %mask = vector.create_mask %i : vector<10xi1>
  %0 = arith.constant 48.32 : f32
  %value = vector.broadcast %0 : f32 to vector<10xf32>
  %idx0 = index.constant 36
  %idx1 = index.constant 51
  %idx2 = index.constant 54
  %idx3 = index.constant 16
  vector.compressstore %base[%idx0, %idx1, %idx2, %idx3], %mask, %value : memref<?x3x?x?xf32>, vector<10xi1>, vector<10xf32>
  return 
}