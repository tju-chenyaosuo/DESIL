

# DESIL

Welcome to the home page of DESIL's repository.
Note that this repository only contains the source code of DESIL.
The artifact of DESIL (including experiment script, experiment data, and docker environment) can be found in zenodo (https://zenodo.org/records/15727517).


# Build DESIL (src folder)

**Note that we integrated DESIL, MLIRSmith, MLIRod, and MLIR compiler infrastructure together into**
```src```
**, you can build all of them together!**

Use the following commands to build DESIL, MLIRSmith, MLIRod, and MLIR compiler infrastructure together:

```
cd src
mkdir build
cd build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build .
```

<!-- ---

You can use additional settings:

```
-DCMAKE_C_COMPILER=afl-cc \
-DCMAKE_CXX_COMPILER=afl-c++ \
```
to support AFL instrumentation for edge coverage collection.

---

or use settings:
```
-DCMAKE_C_FLAGS="-g -O0 -fprofile-arcs -ftest-coverage" \
-DCMAKE_CXX_FLAGS="-g -O0 -fprofile-arcs -ftest-coverage" \
-DCMAKE_EXE_LINKER_FLAGS="-g -fprofile-arcs -ftest-coverage -lgcov" \
```
to enable gcov for line coverage collection. -->

```src``` is developed based on LLVM repository (git version ```1d44ecb9daffbc3b1ed78b7c95662a6fea3f90b9```, not the version under evaluate ```c6d6da4659599507b44c167f335639082f28fae6```), and more detailed information about building MLIR compiler infrastructure can be found in https://mlir.llvm.org/getting_started/

