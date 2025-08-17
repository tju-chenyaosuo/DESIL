func.func @func1() {
  %memref0 = memref.alloc() : memref<500xindex>
  
  // memref.cast
  %before1 = memref.cast %memref0 : memref<500xindex> to memref<?xindex>

  %0 = memref.realloc %memref0 : memref<500xindex> to memref<647xindex>

  // 'memref.store', 'memref.copy', 'memref.cast', 'memref.assume_alignment'

  // memref.store
  %idx68 = index.constant 68
  %idx46 = index.constant 46
  memref.store %idx68, %memref0[%idx46] : memref<500xindex>

  // memref.copy
  %idx55 = index.constant 55
  %memref3 = memref.alloc(%idx55) : memref<?xindex>
  memref.copy %memref0, %memref3 : memref<500xindex> to memref<?xindex>

  // memref.cast
  %after1 = memref.cast %memref0 : memref<500xindex> to memref<?xindex>

  // memref.assume_alignment
  memref.assume_alignment %memref0, 64 : memref<500xindex>

  return
}