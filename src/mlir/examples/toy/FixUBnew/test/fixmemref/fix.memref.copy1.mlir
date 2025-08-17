func.func @func1() {
  %memref_size0 = index.constant 10
  %memref_size1 = index.constant 20
  %memref0 = memref.alloca(%memref_size0) : memref<?xf32>
  %memref1 = memref.alloca(%memref_size1) : memref<?xf32>
  memref.copy %memref0, %memref1 : memref<?xf32> to memref<?xf32>
  return 
}