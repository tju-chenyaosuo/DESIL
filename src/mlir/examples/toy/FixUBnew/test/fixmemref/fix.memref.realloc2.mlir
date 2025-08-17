func.func @func1() {
  %idx85 = index.constant 85
  %memref0 = memref.alloc(%idx85) : memref<?xi32>

  // memref.assume_alignment
  memref.assume_alignment %memref0, 32 : memref<?xi32>

  // memref.copy
  %idx40 = index.constant 40
  %memref1 = memref.alloc(%idx40) : memref<?xi32>
  memref.copy %memref0, %memref1 : memref<?xi32> to memref<?xi32>

  // memref.copy
  %idx54 = index.constant 54
  %memref2 = memref.alloc(%idx54) : memref<?xi32>
  memref.copy %memref0, %memref2 : memref<?xi32> to memref<?xi32>

  // memref.cast
  %before2 = memref.cast %memref0 : memref<?xi32> to memref<97xi32>

  %0 = memref.realloc %memref0 : memref<?xi32> to memref<921xi32>

  // memref.copy
  %idx16 = index.constant 16
  %memref3 = memref.alloc(%idx16) : memref<?xi32>
  memref.copy %memref0, %memref3 : memref<?xi32> to memref<?xi32>

  // memref.cast
  %after1 = memref.cast %memref0 : memref<?xi32> to memref<68xi32>

  return
}