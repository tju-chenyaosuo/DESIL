func.func @func1() {
  %0 = memref.alloca() : memref<8x64xi6>
  memref.assume_alignment %0, 2 : memref<8x64xi6>
  return
}