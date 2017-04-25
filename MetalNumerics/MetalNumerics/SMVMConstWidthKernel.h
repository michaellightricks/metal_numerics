// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import <Foundation/Foundation.h>

#import "MNKernel.h"

NS_ASSUME_NONNULL_BEGIN

@class MNConstantWidthSparseMatrix;

@interface SMVMConstWidthKernel : MNKernel

- (instancetype)initWithContext:(MNContext *)context;

@property (strong, nonatomic) MNConstantWidthSparseMatrix *matrix;
@property (strong, nonatomic) id<MTLBuffer> vector;
@property (strong, nonatomic) id<MTLBuffer> result;

@end

NS_ASSUME_NONNULL_END
