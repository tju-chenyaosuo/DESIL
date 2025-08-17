#include "mlir/IR/MLIRContext.h"
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

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Debug.h"
#include "MutationUtil.h"
#include "Traverse.h"

using namespace mlir;

namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input mlir file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));
static cl::opt<int> inconsistentIndex(cl::Positional,
                                      cl::desc("<op count to be remove>"),
                                      cl::init(1), cl::value_desc("integer"));
static cl::opt<std::string> outputFilename(cl::Positional,
                                           cl::desc("<output mlir file>"),
                                           cl::init("-"),
                                           cl::value_desc("filename"));

unsigned lineNumber() {
  std::ifstream file(inputFilename);
  if (!file.is_open()) {
    std::cerr << "Unable to open file: " << inputFilename << std::endl;
    return 1;
  }
  std::string line;
  unsigned lineNumber = 0;
  while (std::getline(file, line)) {
    lineNumber++;
    // Process the line as needed
    std::cout << "Line " << lineNumber << ": " << line << std::endl;
  }
  file.close();
  return lineNumber;
}

// Usage: /data/src/mlirsmith-dev/build/bin/mutator --operation-prob=0.1
// --operand-prob=0.5 test.mlir 1.mlir
int main(int argc, char **argv) {
  // regist the options
  cl::ParseCommandLineOptions(argc, argv, "mlir mutator\n");
  llvm::InitLLVM(argc, argv);

  // Load the mlir file
  MLIRContext context;
  registerAllDialects(context);
  context.allowUnregisteredDialects();
  OwningOpRef<ModuleOp> module;
  llvm::SourceMgr sourceMgr;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
    return 3;
  }
#ifdef DEBUG
  llvm::errs() << "[main] Load file " << inputFilename << " success!\n";
#endif

  // auto firstOp = module.getOps();
  // llvm::errs() << "[main] " << firstOp.getName() << "\n";

  // unsigned fileLineNumber = lineNumber();
  // unsigned mutPos = rollIdx(fileLineNumber);

  srand(time(0));
  Operation *op = module.get();
  // MutationParser mp = MutationParser(mutPos);
  // MutationBlock *mb = new MutationBlock();
  // mp.printOperation(op, mb);

  // traverse(op);

  op->walk([](func::FuncOp funcOp) {
    unsigned numArgs = funcOp.getNumArguments();
    if (numArgs == 0) {
      llvm::outs() << "[main] Collect all top level operations\n";
      llvm::SmallVector<Operation *> ops = llvm::SmallVector<Operation *>();
      for (Region &region : funcOp.getOperation()->getRegions()) {
        for (Block &block : region.getBlocks()) {
          for (Operation &op : block.getOperations()) {
            ops.push_back(&op);
          }
        }
      }

      // ops[ops.size()-2] must be the last non-return operation

      // get target op
      // mlir::Operation *targetPrintOp = ops[ops.size() - 1 -
      // inconsistentIndex];
      mlir::Operation *targetPrintOp = NULL;
      int print_cnt = -1;
      for (int i = 0; i < ops.size(); ++i) {
        if (getOperationName(ops[i]) == "vector.print") {
          ++print_cnt;
        }
        if (print_cnt == inconsistentIndex) {
          targetPrintOp = ops[i];
          break;
        }
      }
      llvm::outs() << "[main] target op is: " << getOperationName(targetPrintOp)
                   << "\n";

      // calculate ops to remain/remove
      llvm::outs() << "[main] Calculate operations to be removed\n";
      std::vector<Operation *> ops2Remain = std::vector<Operation *>();
      ops2Remain.push_back(targetPrintOp);
      unsigned idx = 0;
      while (idx < ops2Remain.size()) {
        for (unsigned i = 0; i < ops2Remain[idx]->getNumOperands(); ++i) {
          mlir::Operation *o = ops2Remain[idx]->getOperand(i).getDefiningOp();
          if (std::find(ops2Remain.begin(), ops2Remain.end(), o) ==
                  ops2Remain.end() &&
              std::find(ops.begin(), ops.end(), o) != ops.end())
            ops2Remain.push_back(o);
        }
        for (int i = 0; i < ops2Remain[idx]->getNumResults(); ++i) {
          for (auto &use : ops2Remain[idx]->getResult(i).getUses()) {
            mlir::Operation *userOp = use.getOwner();
            if (std::find(ops2Remain.begin(), ops2Remain.end(), userOp) ==
                    ops2Remain.end() &&
                std::find(ops.begin(), ops.end(), userOp) != ops.end())
              ops2Remain.push_back(userOp);
          }
        }
        ++idx;
      }
      // for (int i = 0; i < ops2Remain.size(); ++i) {
      //   llvm::outs() << "[main] ops2Remain[" << i
      //                << "]=" << getOperationName(ops2Remain[i]) << "-"
      //                << getLocationStr(ops2Remain[i]) << "\n";
      // }
      // ops2Remain.erase(ops2Remain.begin()); // remove target vector.print

      llvm::SmallVector<Operation *> ops2Remove =
          llvm::SmallVector<Operation *>();
      for (int i = ops.size() - 2; i >= 0; --i) {
        bool isRemain = false;
        for (int idxRemain = 0; idxRemain < ops2Remain.size() && !isRemain;
             ++idxRemain) {
          isRemain = (ops2Remain[idxRemain] == ops[i]);
        }
        if (!isRemain) {
          ops2Remove.push_back(ops[i]);
        }
      }

      // Remove the unrelated ops
      llvm::outs() << "[main] Remove operations\n";
      for (unsigned i = 0; i < ops2Remove.size(); ++i) {
        if (ops2Remove[i]->getUses().empty() &&
            ops2Remove[i]->getUsers().empty()) {
          llvm::outs() << "remove " << getOperationName(ops2Remove[i]) << "-"
                       << getLocationStr(ops2Remove[i]) << "\n";
          ops2Remove[i]->erase();
        }
      }
    }
  });

  std::error_code error;
  llvm::raw_fd_ostream output(outputFilename, error);
  if (error) {
    llvm::errs() << "Error opening file for writing: " << error.message()
                 << "\n";
    return 1;
  }
  module->print(output);

  return 0;
}

// int main1(int argc, char **argv) {
//   // regist the options
//   cl::ParseCommandLineOptions(argc, argv, "mlir mutator\n");
//   llvm::InitLLVM(argc, argv);

//   // Load the mlir file
//   MLIRContext context;
//   registerAllDialects(context);
//   context.allowUnregisteredDialects();
//   OwningOpRef<ModuleOp> module;
//   llvm::SourceMgr sourceMgr;

//   llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
//       llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
//   sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
//   module = parseSourceFile<ModuleOp>(sourceMgr, &context);
//   if (!module) {
//     llvm::errs() << "[main] Error can't load file " << inputFilename << "\n";
//     return 3;
//   }
// #ifdef DEBUG
//   llvm::errs() << "[main] Load file " << inputFilename << " success!\n";
// #endif

//   // auto firstOp = module.getOps();
//   // llvm::errs() << "[main] " << firstOp.getName() << "\n";

//   // unsigned fileLineNumber = lineNumber();
//   // unsigned mutPos = rollIdx(fileLineNumber);

//   srand(time(0));
//   Operation *op = module.get();
//   // MutationParser mp = MutationParser(mutPos);
//   // MutationBlock *mb = new MutationBlock();
//   // mp.printOperation(op, mb);

//   // traverse(op);

//   op->walk([](func::FuncOp funcOp) {
//     unsigned numArgs = funcOp.getNumArguments();
//     if (numArgs == 0) {
//       llvm::outs() << "[main] Collect all top level operations\n";
//       llvm::SmallVector<Operation *> ops = llvm::SmallVector<Operation *>();
//       for (Region &region : funcOp.getOperation()->getRegions()) {
//         for (Block &block : region.getBlocks()) {
//           for (Operation &op : block.getOperations()) {
//             ops.push_back(&op);
//           }
//         }
//       }

//       // ops[ops.size()-2] must be the last non-return operation

//       // get target op
//       // mlir::Operation *targetPrintOp = ops[ops.size() - 1 -
//       // inconsistentIndex];
//       mlir::Operation *targetPrintOp = NULL;
//       int print_cnt = -1;
//       for (int i = 0; i < ops.size(); ++i) {
//         if (getOperationName(ops[i]) == "vector.print") {
//           ++print_cnt;
//         }
//         if (print_cnt == inconsistentIndex) {
//           targetPrintOp = ops[i];
//           break;
//         }
//       }

//       // initilaize OpBuilder
//       mlir::Operation *returnOp = ops[ops.size() - 1];
//       mlir::OpBuilder builder(returnOp->getBlock(),
//                               returnOp->getBlock()->begin());
//       builder.setInsertionPoint(returnOp);
//       mlir::Location loc = returnOp->getLoc();

//       // get target vector.print, and print the subvalues of this print op
//       llvm::outs() << "[main] Print target value\n";
//       mlir::Value vectorParam = targetPrintOp->getOperand(0);
//       mlir::Operation *sourceOp = vectorParam.getDefiningOp();
//       std::vector<mlir::Value> toPrint = std::vector<mlir::Value>();
//       for (unsigned i = 0; i < sourceOp->getNumOperands(); ++i) {
//         mlir::Value v = sourceOp->getOperand(i);
//         mlir::Type t = v.getType();
//         if (t.isIntOrIndex())
//           toPrint.push_back(sourceOp->getOperand(i));
//       }

//       // builder.create<vector::PrintOp>(loc, sourceOp->getOperand(i));
//       while (!toPrint.empty()) {
//         int minLocIdx = -1;
//         int idx = 0;
//         int minLoc = 10000000;
//         while (idx < toPrint.size()) {
//           int opLoc = getLineNumber(toPrint[idx].getDefiningOp());
//           if (opLoc < minLoc) {
//             minLoc = opLoc;
//             minLocIdx = idx;
//           }
//           ++idx;
//         }
//         builder.create<vector::PrintOp>(loc, toPrint[minLocIdx]);
//         toPrint.erase(toPrint.begin() + minLocIdx);
//       }

//       // remove all old op
//       for (int i = 0; i < ops.size(); ++i) {
//         if (getOperationName(ops[i]) == "vector.print")
//           ops[i]->erase();
//       }
//     }
//   });

//   std::error_code error;
//   llvm::raw_fd_ostream output(outputFilename, error);
//   if (error) {
//     llvm::errs() << "Error opening file for writing: " << error.message()
//                  << "\n";
//     return 1;
//   }
//   module->print(output);

//   return 0;
// }
