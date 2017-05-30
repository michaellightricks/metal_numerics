// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNKernel.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNAverageKernel : MNKernel

@property (strong, nonatomic) id<MTLBuffer> buffer;
@property (strong, nonatomic) id<MTLBuffer> result;

@end

NS_ASSUME_NONNULL_END
