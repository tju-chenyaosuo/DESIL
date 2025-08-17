func.func @func1() {
  %memref_size0 = index.constant 41
  %memref_size1 = index.constant 63

  %memref_size2 = index.constant 4

  %memref0 = memref.alloca(%memref_size0, %memref_size1) : memref<?x?xf80>
  %memref1 = memref.alloca(%memref_size2) : memref<10x?xf80>
  memref.copy %memref0, %memref1 : memref<?x?xf80> to memref<10x?xf80>
  return 
}