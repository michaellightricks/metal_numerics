// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNKernel.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNCovarianceKernel : MNKernel

+ (void)testWithContext:(MNContext *)context;

@property (strong, nonatomic) id<MTLTexture> inputTexture;

@property (strong, nonatomic) id<MTLBuffer> outputBuffer;

@end

NS_ASSUME_NONNULL_END
