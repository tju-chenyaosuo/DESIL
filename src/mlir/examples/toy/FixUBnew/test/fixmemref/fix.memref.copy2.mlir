func.func @func1() {
  %memref_size0 = index.constant 59
  %memref_size1 = index.constant 36
  %memref_size2 = index.constant 30

  %memref_size3 = index.constant 14
  %memref_size4 = index.constant 36
  %memref_size5 = index.constant 7

  %memref0 = memref.alloca(%memref_size0, %memref_size1, %memref_size2) : memref<?x7x8x?x?xf32>
  %memref1 = memref.alloca(%memref_size3, %memref_size4, %memref_size5) : memref<2x?x?x2x?xf32>
  memref.copy %memref0, %memref1 : memref<?x7x8x?x?xf32> to memref<2x?x?x2x?xf32>
  return 
}