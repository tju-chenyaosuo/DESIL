//
// Created by Stan Wang on 2022/9/13.
//
#ifndef MLIRSMITH_H
#define MLIRSMITH_H

#include "smith/RegionGeneration.h"
#include "smith/TypeGeneration.h"
#include "smith/generators/OpGeneration.h"
#include "toy/Dialect.h"
#include "smith/DiversityCriteria.h"

// using namespace mlir;

int printConfig();
std::unique_ptr<mlir::Pass> createMLIRSmithPass();

#endif