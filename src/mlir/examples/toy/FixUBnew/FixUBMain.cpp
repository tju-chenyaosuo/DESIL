#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

#include "fixub/ConstantPool.h"
#include "fixub/DelOp.h"
#include "fixub/FixAffine.h"
#include "fixub/FixArith.h"
#include "fixub/FixIndex.h"
#include "fixub/FixLinalg.h"
#include "fixub/FixMemRef.h"
#include "fixub/FixTensor.h"
#include "fixub/FixTosa.h"
#include "fixub/FixUBUtils.h"
#include "fixub/FixVector.h"
#include "fixub/FixScf.h"

static llvm::cl::opt<std::string>
    inputFilename(llvm::cl::Positional, llvm::cl::desc("<input MLIR file>"),
                  llvm::cl::init("-"), llvm::cl::value_desc("filename"));

struct FixUB {
  static inline std::vector<std::string> UBProneOperations = {
      "affine.load",
      "affine.store",
      "affine.vector_store",
      "affine.vector_load",
      "affine.yield",
      "arith.shli",
      "arith.shrsi",
      "arith.shrui",
      "arith.ceildivsi",
      "arith.divsi",
      "arith.floordivsi",
      "arith.ceildivui",
      "arith.divui",
      "arith.remui",
      "arith.remsi",
      "index.shl",
      "index.shrs",
      "index.shru",
      "index.ceildivs",
      "index.divs",
      "index.floordivs",
      "index.ceildivu",
      "index.divu",
      "index.remu",
      "index.rems",
      "scf.condition",
      "linalg.copy",
      "linalg.broadcast",
      "linalg.generic",
      "linalg.map",
      "linalg.matmul",
      "linalg.transpose",
      "memref.alloc",
      "memref.alloca",
      "memref.assume_alignment",
      "memref.copy",
      "memref.cast",
      "memref.realloc",
      "memref.load",
      "memref.store",
      "memref.dim",
      "tensor.empty",
      "tensor.cast",
      "tensor.dim",
      "tensor.scatter",
      "tensor.unpack",
      "tensor.pack",
      "tensor.extract",
      "tensor.insert",
      "vector.maskedload",
      "vector.maskedstore",
      "vector.compressstore",
      "vector.expandload",
      "vector.gather",
      "vector.scatter",
      "vector.reshape",
      "vector.insertelement",
      "vector.load",
      "tosa.transpose_conv2d",
  };
  static inline std::vector<mlir::Operation *> UBProneOpList =
      std::vector<mlir::Operation *>();
  static void fixUB(mlir::Operation *op);
  static void registOperation(mlir::Operation *op);
  static void traverseAndRegistAll(mlir::Operation *op);
};

void FixUB::traverseAndRegistAll(mlir::Operation *op) {
  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      for (mlir::Operation &op : block.getOperations()) {
        traverseAndRegistAll(&op);
      }
    }
  }
  registOperation(op);
}

void FixUB::registOperation(mlir::Operation *op) {
  // llvm::outs() << "[FixUB::registOperation] op info: " <<
  // FixUBUtils::getOperationNameParamType(op) << "\n";
  std::string opname = op->getName().getStringRef().str();
  auto pos =
      std::find(UBProneOperations.begin(), UBProneOperations.end(), opname);
  if (pos != UBProneOperations.end()) {
    UBProneOpList.push_back(op);
  }
}

void FixUB::fixUB(mlir::Operation *op) {
  traverseAndRegistAll(op);
  for (mlir::Operation *op : UBProneOpList) {
    llvm::outs() << "[FixUB::fixUB] op info: "
                 << FixUBUtils::getOperationNameParamType(op) << " "
                 << FixUBUtils::getLocationStr(op) << "\n";
    FixAffine::fixaffine(op);
    FixArith::fixarith(op);
    FixIndex::fixindex(op);
    FixLinalg::fixlinalg(op);
    FixTensor::fixtensor(op);
    FixVector::fixvector(op);
    FixMemref::fixmemref(op);
    FixTosa::fixtosa(op);
    FixScf::fixscf(op);
  }
  return;
}

int dumpMLIR() {
  mlir::MLIRContext context;
  mlir::registerAllDialects(context);
  // context.getOrLoadDialect<mlir::toy::ToyDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::affine::AffineDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::index::IndexDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::spirv::SPIRVDialect>();
  context.getOrLoadDialect<mlir::tosa::TosaDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
    return 3;
  }
  mlir::Operation *op = module.get();

  for (mlir::Region &region : op->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      for (mlir::Operation &op : block.getOperations()) {
        std::string opname = op.getName().getStringRef().str();
        if (opname == "func.func") {
          ConstantPool::funcOp = &op;
        }
      }
      if (ConstantPool::funcOp) {
        break;
      }
    }
    if (ConstantPool::funcOp) {
      break;
    }
  }

  // The entry of FIXUB
  FixUB::fixUB(op);
  for (mlir::Operation *op : delOps) {
    op->erase();
  }

  module->dump();

  return 0;
}

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "FixUB tool\n");
  llvm::InitLLVM(argc, argv);

  dumpMLIR();

  return 0;
}
