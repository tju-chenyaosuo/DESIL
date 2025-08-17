#include <random>


#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
//#include "mlir/Dialect/Index/IR/IndexEnums.h.inc"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <map>


enum value_type
{
    SCALAR,
    TENSOR,
    VECTOR,
    MEMREF
};
struct CheckSum {

    std::map<std::string, std::map<value_type, std::vector<mlir::Value>>> checksumValuePool; // example: checksumValuePool["i16"][SCALAR] = %1 (value)

    std::map<value_type, std::map<std::string, mlir::func::FuncOp>> funcOpPool; // example: funcOpPool[TENSOR]["i16"] = tensor_i16(funcOp)

    static mlir::Operation *mainfunc, *idx0op, *idx1op;

    void addValue(mlir::Value val);

    mlir::Value calCheckSumAll(mlir::Location loc, mlir::OpBuilder &builder);

    mlir::Value calCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder, mlir::Type type);

    mlir::Value calTensorCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::Type type, std::string typestr,mlir::Value lstval);

    mlir::Value calMemrefCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::Type type, std::string typestr,mlir::Value lstval);

    mlir::Value calVectorCheckSumByType(mlir::Location loc, mlir::OpBuilder &builder,
                                       mlir::Type type, std::string typestr,mlir::Value lstval);

    mlir::func::FuncOp calCheckSumTensor(mlir::Type type);

    mlir::func::FuncOp calCheckSumMemref(mlir::Type type);

    inline static std::string getValueTypeStr(mlir::Value v);
    
    inline static std::string getTypeStr(mlir::Type type);

};

