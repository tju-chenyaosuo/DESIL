module {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @printF32(f32)
  llvm.func @printF16(i16)
  llvm.func @printNewline()
  llvm.func @printI64(i64)
  llvm.func @func1(%arg0: f32, %arg1: i64, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64, %arg8: i64) -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(10 : index) : i64
    %1 = llvm.mlir.constant(64 : index) : i64
    %2 = llvm.mlir.constant(540 : index) : i64
    %3 = llvm.mlir.constant(20 : index) : i64
    %4 = llvm.mlir.constant(840 : index) : i64
    %5 = llvm.mlir.constant(30 : index) : i64
    %6 = llvm.mlir.constant(27 : index) : i64
    %7 = llvm.mlir.constant(31 : index) : i64
    %8 = llvm.mlir.constant(15 : index) : i64
    %9 = llvm.mlir.constant(dense<6.153600e+04> : vector<1xf16>) : vector<1xf16>
    %10 = llvm.mlir.constant(5 : index) : i64
    %11 = llvm.mlir.constant(dense<406327914> : vector<i64>) : vector<1xi64>
    %12 = llvm.mlir.constant(dense<2.270400e+04> : vector<10xf16>) : vector<10xf16>
    %13 = llvm.mlir.constant(dense<1.000000e+00> : vector<10x1x10xf32>) : !llvm.array<10 x array<1 x vector<10xf32>>>
    %14 = llvm.mlir.constant(dense<0> : vector<1xi32>) : vector<1xi32>
    %15 = llvm.mlir.constant(dense<true> : vector<1xi1>) : vector<1xi1>
    %16 = llvm.mlir.constant(dense<false> : vector<10x6xi1>) : !llvm.array<10 x vector<6xi1>>
    %17 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %18 = llvm.mlir.constant(6.153600e+04 : f16) : f16
    %19 = llvm.mlir.constant(2.270400e+04 : f16) : f16
    %20 = llvm.mlir.constant(685031725 : i64) : i64
    %21 = llvm.mlir.constant(6.224000e+03 : f16) : f16
    %22 = llvm.mlir.constant(2088370536 : i64) : i64
    %23 = llvm.mlir.constant(1862809330 : i64) : i64
    %24 = llvm.mlir.constant(6.550400e+04 : f16) : f16
    %25 = llvm.mlir.constant(406327914 : i64) : i64
    %26 = llvm.mlir.constant(1 : index) : i64
    %27 = llvm.mlir.constant(2 : index) : i64
    %28 = llvm.mlir.constant(3 : index) : i64
    %29 = llvm.mlir.constant(7 : index) : i64
    %30 = llvm.mlir.constant(17 : index) : i64
    %31 = llvm.mlir.constant(26 : index) : i64
    %32 = llvm.mlir.constant(28 : index) : i64
    %33 = llvm.mlir.constant(dense<0.000000e+00> : vector<10x1x10xf32>) : !llvm.array<10 x array<1 x vector<10xf32>>>
    %34 = llvm.mlir.constant(1 : i64) : i64
    %35 = llvm.mlir.constant(0 : i64) : i64
    %36 = llvm.mlir.constant(-32610 : i64) : i64
    %37 = llvm.mlir.constant(25943 : i64) : i64
    %38 = llvm.mlir.constant(0 : i32) : i32
    %39 = llvm.mlir.constant(0 : index) : i64
    %40 = llvm.mlir.constant(6 : index) : i64
    %41 = llvm.mlir.constant(dense<0.000000e+00> : vector<1x10xf32>) : !llvm.array<1 x vector<10xf32>>
    %42 = builtin.unrealized_conversion_cast %30 : i64 to index
    %43 = builtin.unrealized_conversion_cast %39 : i64 to index
    %44 = builtin.unrealized_conversion_cast %16 : !llvm.array<10 x vector<6xi1>> to vector<10x6xi1>
    %45 = builtin.unrealized_conversion_cast %13 : !llvm.array<10 x array<1 x vector<10xf32>>> to vector<10x1x10xf32>
    %46 = builtin.unrealized_conversion_cast %11 : vector<1xi64> to vector<i64>
    %47 = builtin.unrealized_conversion_cast %10 : i64 to index
    %48 = llvm.mlir.null : !llvm.ptr
    %49 = llvm.getelementptr %48[3780] : (!llvm.ptr) -> !llvm.ptr, i64
    %50 = llvm.ptrtoint %49 : !llvm.ptr to i64
    %51 = llvm.add %50, %1  : i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr
    %53 = llvm.ptrtoint %52 : !llvm.ptr to i64
    %54 = llvm.sub %1, %26  : i64
    %55 = llvm.add %53, %54  : i64
    %56 = llvm.urem %55, %1  : i64
    %57 = llvm.sub %55, %56  : i64
    %58 = llvm.inttoptr %57 : i64 to !llvm.ptr
    %59 = llvm.mlir.null : !llvm.ptr
    %60 = llvm.getelementptr %59[100] : (!llvm.ptr) -> !llvm.ptr, i64
    %61 = llvm.ptrtoint %60 : !llvm.ptr to i64
    %62 = llvm.add %61, %1  : i64
    %63 = llvm.call @malloc(%62) : (i64) -> !llvm.ptr
    %64 = llvm.ptrtoint %63 : !llvm.ptr to i64
    %65 = llvm.sub %1, %26  : i64
    %66 = llvm.add %64, %65  : i64
    %67 = llvm.urem %66, %1  : i64
    %68 = llvm.sub %66, %67  : i64
    %69 = llvm.inttoptr %68 : i64 to !llvm.ptr
    %70 = llvm.mlir.null : !llvm.ptr
    %71 = llvm.getelementptr %70[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %72 = llvm.ptrtoint %71 : !llvm.ptr to i64
    %73 = llvm.call @malloc(%72) : (i64) -> !llvm.ptr
    %74 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %75 = llvm.insertvalue %73, %74[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %76 = llvm.insertvalue %73, %75[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.insertvalue %39, %76[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %78 = llvm.insertvalue %26, %77[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %79 = llvm.insertvalue %26, %78[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %80 = builtin.unrealized_conversion_cast %79 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi32>
    %81 = llvm.mlir.null : !llvm.ptr
    %82 = llvm.getelementptr %81[186] : (!llvm.ptr) -> !llvm.ptr, i64
    %83 = llvm.ptrtoint %82 : !llvm.ptr to i64
    %84 = llvm.call @malloc(%83) : (i64) -> !llvm.ptr
    %85 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %86 = llvm.insertvalue %84, %85[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %87 = llvm.insertvalue %84, %86[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %88 = llvm.insertvalue %39, %87[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %89 = llvm.insertvalue %7, %88[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %90 = llvm.insertvalue %40, %89[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.insertvalue %40, %90[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.insertvalue %26, %91[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %93 = builtin.unrealized_conversion_cast %92 : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> to memref<31x6xi64>
    %94 = llvm.mlir.null : !llvm.ptr
    %95 = llvm.getelementptr %94[36] : (!llvm.ptr) -> !llvm.ptr, i64
    %96 = llvm.ptrtoint %95 : !llvm.ptr to i64
    %97 = llvm.call @malloc(%96) : (i64) -> !llvm.ptr
    %98 = llvm.mlir.null : !llvm.ptr
    %99 = llvm.getelementptr %98[60] : (!llvm.ptr) -> !llvm.ptr, f16
    %100 = llvm.ptrtoint %99 : !llvm.ptr to i64
    %101 = llvm.call @malloc(%100) : (i64) -> !llvm.ptr
    %102 = llvm.mlir.null : !llvm.ptr
    %103 = llvm.getelementptr %102[75] : (!llvm.ptr) -> !llvm.ptr, i1
    %104 = llvm.ptrtoint %103 : !llvm.ptr to i64
    %105 = llvm.call @malloc(%104) : (i64) -> !llvm.ptr
    %106 = llvm.mlir.null : !llvm.ptr
    %107 = llvm.getelementptr %106[60] : (!llvm.ptr) -> !llvm.ptr, f16
    %108 = llvm.ptrtoint %107 : !llvm.ptr to i64
    %109 = llvm.call @malloc(%108) : (i64) -> !llvm.ptr
    %110 = llvm.mlir.null : !llvm.ptr
    %111 = llvm.getelementptr %110[1] : (!llvm.ptr) -> !llvm.ptr, f16
    %112 = llvm.ptrtoint %111 : !llvm.ptr to i64
    %113 = llvm.call @malloc(%112) : (i64) -> !llvm.ptr
    %114 = llvm.mlir.null : !llvm.ptr
    %115 = llvm.getelementptr %114[22680] : (!llvm.ptr) -> !llvm.ptr, i64
    %116 = llvm.ptrtoint %115 : !llvm.ptr to i64
    %117 = llvm.call @malloc(%116) : (i64) -> !llvm.ptr
    %118 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %119 = llvm.insertvalue %117, %118[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %120 = llvm.insertvalue %117, %119[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %121 = llvm.insertvalue %39, %120[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %122 = llvm.insertvalue %6, %121[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %123 = llvm.insertvalue %32, %122[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %124 = llvm.insertvalue %5, %123[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %125 = llvm.insertvalue %4, %124[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %126 = llvm.insertvalue %5, %125[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %127 = llvm.insertvalue %26, %126[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %128 = builtin.unrealized_conversion_cast %127 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> to memref<27x28x30xi64>
    %129 = llvm.ptrtoint %109 : !llvm.ptr to i64
    %130 = llvm.and %129, %26  : i64
    %131 = llvm.icmp "eq" %130, %39 : i64
    "llvm.intr.assume"(%131) : (i1) -> ()
    %132 = llvm.mlir.null : !llvm.ptr
    %133 = llvm.getelementptr %132[1] : (!llvm.ptr) -> !llvm.ptr, i1
    %134 = llvm.ptrtoint %133 : !llvm.ptr to i64
    %135 = llvm.add %134, %1  : i64
    %136 = llvm.call @malloc(%135) : (i64) -> !llvm.ptr
    %137 = llvm.ptrtoint %136 : !llvm.ptr to i64
    %138 = llvm.sub %1, %26  : i64
    %139 = llvm.add %137, %138  : i64
    %140 = llvm.urem %139, %1  : i64
    %141 = llvm.sub %139, %140  : i64
    %142 = llvm.inttoptr %141 : i64 to !llvm.ptr
    %143 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %144 = llvm.insertvalue %136, %143[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %145 = llvm.insertvalue %142, %144[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %146 = llvm.insertvalue %39, %145[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %147 = llvm.insertvalue %26, %146[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %148 = llvm.insertvalue %26, %147[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %149 = builtin.unrealized_conversion_cast %148 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<1xi1>
    %150 = vector.gather %149[%42] [%14], %15, %15 : memref<1xi1>, vector<1xi32>, vector<1xi1>, vector<1xi1> into vector<1xi1>
    %151 = llvm.intr.matrix.transpose %15 {columns = 1 : i32, rows = 1 : i32} : vector<1xi1> into vector<1xi1>
    %152 = llvm.extractvalue %13[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %153 = llvm.extractvalue %13[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %154 = llvm.extractvalue %13[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %155 = llvm.intr.fmuladd(%152, %153, %154)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %156 = llvm.insertvalue %155, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %157 = llvm.insertvalue %156, %33[0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %158 = llvm.extractvalue %13[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %159 = llvm.extractvalue %13[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %160 = llvm.extractvalue %13[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %161 = llvm.intr.fmuladd(%158, %159, %160)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %162 = llvm.insertvalue %161, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %163 = llvm.insertvalue %162, %157[1] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %164 = llvm.extractvalue %13[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %165 = llvm.extractvalue %13[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %166 = llvm.extractvalue %13[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %167 = llvm.intr.fmuladd(%164, %165, %166)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %168 = llvm.insertvalue %167, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %169 = llvm.insertvalue %168, %163[2] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %170 = llvm.extractvalue %13[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %171 = llvm.extractvalue %13[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %172 = llvm.extractvalue %13[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %173 = llvm.intr.fmuladd(%170, %171, %172)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %174 = llvm.insertvalue %173, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %175 = llvm.insertvalue %174, %169[3] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %176 = llvm.extractvalue %13[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %177 = llvm.extractvalue %13[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %178 = llvm.extractvalue %13[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %179 = llvm.intr.fmuladd(%176, %177, %178)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %180 = llvm.insertvalue %179, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %181 = llvm.insertvalue %180, %175[4] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %182 = llvm.extractvalue %13[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %183 = llvm.extractvalue %13[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %184 = llvm.extractvalue %13[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %185 = llvm.intr.fmuladd(%182, %183, %184)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %186 = llvm.insertvalue %185, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %187 = llvm.insertvalue %186, %181[5] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %188 = llvm.extractvalue %13[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %189 = llvm.extractvalue %13[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %190 = llvm.extractvalue %13[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %191 = llvm.intr.fmuladd(%188, %189, %190)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %192 = llvm.insertvalue %191, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %193 = llvm.insertvalue %192, %187[6] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %194 = llvm.extractvalue %13[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %195 = llvm.extractvalue %13[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %196 = llvm.extractvalue %13[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %197 = llvm.intr.fmuladd(%194, %195, %196)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %198 = llvm.insertvalue %197, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %199 = llvm.insertvalue %198, %193[7] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %200 = llvm.extractvalue %13[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %201 = llvm.extractvalue %13[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %202 = llvm.extractvalue %13[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %203 = llvm.intr.fmuladd(%200, %201, %202)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %204 = llvm.insertvalue %203, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %205 = llvm.insertvalue %204, %199[8] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %206 = llvm.extractvalue %13[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %207 = llvm.extractvalue %13[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %208 = llvm.extractvalue %13[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %209 = llvm.intr.fmuladd(%206, %207, %208)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %210 = llvm.insertvalue %209, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %211 = llvm.insertvalue %210, %205[9] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %212 = builtin.unrealized_conversion_cast %211 : !llvm.array<10 x array<1 x vector<10xf32>>> to vector<10x1x10xf32>
    %213 = llvm.mul %29, %40  : i64
    %214 = llvm.add %213, %28  : i64
    %215 = llvm.getelementptr %101[%214] : (!llvm.ptr, i64) -> !llvm.ptr, f16
    llvm.store %12, %215 {alignment = 2 : i64} : vector<10xf16>, !llvm.ptr
    %216 = llvm.intr.matrix.transpose %151 {columns = 1 : i32, rows = 1 : i32} : vector<1xi1> into vector<1xi1>
    %217 = llvm.extractelement %11[%39 : i64] : vector<1xi64>
    %218 = llvm.mul %32, %40  : i64
    %219 = llvm.add %218, %31  : i64
    %220 = llvm.getelementptr %97[%219] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    llvm.store %217, %220 : i64, !llvm.ptr
    %221 = llvm.mul %39, %2  : i64
    %222 = llvm.mul %39, %3  : i64
    %223 = llvm.add %221, %222  : i64
    %224 = llvm.add %223, %39  : i64
    %225 = llvm.getelementptr %58[%224] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %226 = llvm.load %225 : !llvm.ptr -> i64
    %227 = vector.load %80[%43] : memref<1xi32>, vector<10x1x10xi32>
    %228 = llvm.mul %26, %0  : i64
    %229 = llvm.mul %39, %0  : i64
    %230 = llvm.add %228, %229  : i64
    %231 = llvm.add %230, %39  : i64
    %232 = llvm.getelementptr %69[%231] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %233 = llvm.load %232 : !llvm.ptr -> i64
    %234 = llvm.mul %39, %40  : i64
    %235 = llvm.add %234, %27  : i64
    %236 = llvm.getelementptr %84[%235] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %237 = llvm.atomicrmw max %236, %226 acq_rel : !llvm.ptr, i64
    %238 = llvm.extractvalue %13[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %239 = llvm.extractvalue %13[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %240 = llvm.extractvalue %157[0, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %241 = llvm.intr.fmuladd(%238, %239, %240)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %242 = llvm.insertvalue %241, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %243 = llvm.insertvalue %242, %33[0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %244 = llvm.extractvalue %13[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %245 = llvm.extractvalue %13[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %246 = llvm.extractvalue %163[1, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %247 = llvm.intr.fmuladd(%244, %245, %246)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %248 = llvm.insertvalue %247, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %249 = llvm.insertvalue %248, %243[1] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %250 = llvm.extractvalue %13[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %251 = llvm.extractvalue %13[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %252 = llvm.extractvalue %169[2, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %253 = llvm.intr.fmuladd(%250, %251, %252)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %254 = llvm.insertvalue %253, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %255 = llvm.insertvalue %254, %249[2] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %256 = llvm.extractvalue %13[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %257 = llvm.extractvalue %13[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %258 = llvm.extractvalue %175[3, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %259 = llvm.intr.fmuladd(%256, %257, %258)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %260 = llvm.insertvalue %259, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %261 = llvm.insertvalue %260, %255[3] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %262 = llvm.extractvalue %13[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %263 = llvm.extractvalue %13[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %264 = llvm.extractvalue %181[4, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %265 = llvm.intr.fmuladd(%262, %263, %264)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %266 = llvm.insertvalue %265, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %267 = llvm.insertvalue %266, %261[4] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %268 = llvm.extractvalue %13[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %269 = llvm.extractvalue %13[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %270 = llvm.extractvalue %187[5, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %271 = llvm.intr.fmuladd(%268, %269, %270)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %272 = llvm.insertvalue %271, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %273 = llvm.insertvalue %272, %267[5] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %274 = llvm.extractvalue %13[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %275 = llvm.extractvalue %13[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %276 = llvm.extractvalue %193[6, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %277 = llvm.intr.fmuladd(%274, %275, %276)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %278 = llvm.insertvalue %277, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %279 = llvm.insertvalue %278, %273[6] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %280 = llvm.extractvalue %13[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %281 = llvm.extractvalue %13[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %282 = llvm.extractvalue %199[7, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %283 = llvm.intr.fmuladd(%280, %281, %282)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %284 = llvm.insertvalue %283, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %285 = llvm.insertvalue %284, %279[7] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %286 = llvm.extractvalue %13[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %287 = llvm.extractvalue %13[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %288 = llvm.extractvalue %205[8, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %289 = llvm.intr.fmuladd(%286, %287, %288)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %290 = llvm.insertvalue %289, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %291 = llvm.insertvalue %290, %285[8] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %292 = llvm.extractvalue %13[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %293 = llvm.extractvalue %13[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %294 = llvm.extractvalue %211[9, 0] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %295 = llvm.intr.fmuladd(%292, %293, %294)  : (vector<10xf32>, vector<10xf32>, vector<10xf32>) -> vector<10xf32>
    %296 = llvm.insertvalue %295, %41[0] : !llvm.array<1 x vector<10xf32>> 
    %297 = llvm.insertvalue %296, %291[9] : !llvm.array<10 x array<1 x vector<10xf32>>> 
    %298 = builtin.unrealized_conversion_cast %297 : !llvm.array<10 x array<1 x vector<10xf32>>> to vector<10x1x10xf32>
    %299 = llvm.intr.matrix.multiply %216, %151 {lhs_columns = 1 : i32, lhs_rows = 1 : i32, rhs_columns = 1 : i32} : (vector<1xi1>, vector<1xi1>) -> vector<1xi1>
    %300 = vector.load %93[%43, %47] : memref<31x6xi64>, vector<10x1x10xi64>
    %301 = llvm.mul %39, %8  : i64
    %302 = llvm.mul %39, %8  : i64
    %303 = llvm.add %301, %302  : i64
    %304 = llvm.add %303, %39  : i64
    %305 = llvm.getelementptr %105[%304] : (!llvm.ptr, i64) -> !llvm.ptr, i1
    %306 = llvm.load %305 : !llvm.ptr -> i1
    %307 = llvm.mlir.undef : vector<1xi1>
    %308 = llvm.insertelement %306, %307[%38 : i32] : vector<1xi1>
    %309 = llvm.shufflevector %308, %307 [0] : vector<1xi1> 
    %310 = vector.load %128[%43, %43, %43] : memref<27x28x30xi64>, vector<10x1x10xi64>
    %311 = llvm.getelementptr %113[15] : (!llvm.ptr) -> !llvm.ptr, f16
    %312 = llvm.load %311 {alignment = 2 : i64} : !llvm.ptr -> vector<10xf16>
    vector.print %15 : vector<1xi1>
    vector.print %14 : vector<1xi32>
    vector.print %150 : vector<1xi1>
    vector.print %151 : vector<1xi1>
    vector.print %45 : vector<10x1x10xf32>
    vector.print %212 : vector<10x1x10xf32>
    vector.print %216 : vector<1xi1>
    vector.print %46 : vector<i64>
    vector.print %227 : vector<10x1x10xi32>
    vector.print %45 : vector<10x1x10xf32>
    vector.print %298 : vector<10x1x10xf32>
    vector.print %299 : vector<1xi1>
    vector.print %9 : vector<1xf16>
    vector.print %300 : vector<10x1x10xi64>
    vector.print %309 : vector<1xi1>
    vector.print %310 : vector<10x1x10xi64>
    vector.print %44 : vector<10x6xi1>
    vector.print %14 : vector<1xi32>
    vector.print %312 : vector<10xf16>
    llvm.call @printI64(%34) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%34) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %313 = llvm.bitcast %18 : f16 to i16
    llvm.call @printF16(%313) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%35) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %314 = llvm.bitcast %19 : f16 to i16
    llvm.call @printF16(%314) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%20) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%35) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %315 = llvm.bitcast %21 : f16 to i16
    llvm.call @printF16(%315) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%22) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%23) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%36) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    %316 = llvm.bitcast %24 : f16 to i16
    llvm.call @printF16(%316) : (i16) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%25) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%37) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%34) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%35) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%35) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%34) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printF32(%17) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%226) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printI64(%233) : (i64) -> ()
    llvm.call @printNewline() : () -> ()
    llvm.call @printF32(%17) : (f32) -> ()
    llvm.call @printNewline() : () -> ()
    %317 = llvm.mlir.null : !llvm.ptr
    %318 = llvm.getelementptr %317[150] : (!llvm.ptr) -> !llvm.ptr, f16
    %319 = llvm.ptrtoint %318 : !llvm.ptr to i64
    %320 = llvm.add %319, %1  : i64
    %321 = llvm.call @malloc(%320) : (i64) -> !llvm.ptr
    %322 = llvm.ptrtoint %321 : !llvm.ptr to i64
    %323 = llvm.sub %1, %26  : i64
    %324 = llvm.add %322, %323  : i64
    %325 = llvm.urem %324, %1  : i64
    %326 = llvm.sub %324, %325  : i64
    %327 = llvm.inttoptr %326 : i64 to !llvm.ptr
    %328 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
    %329 = llvm.insertvalue %321, %328[0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %330 = llvm.insertvalue %327, %329[1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %331 = llvm.insertvalue %39, %330[2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %332 = llvm.insertvalue %8, %331[3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %333 = llvm.insertvalue %27, %332[3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %334 = llvm.insertvalue %10, %333[3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %335 = llvm.insertvalue %0, %334[4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %336 = llvm.insertvalue %10, %335[4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    %337 = llvm.insertvalue %26, %336[4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> 
    llvm.return %337 : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
  }
}

