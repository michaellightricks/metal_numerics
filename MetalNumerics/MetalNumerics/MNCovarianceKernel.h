// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNKernel.h"
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN

@interface MNCovarianceKernel : MNKernel

+ (void)testWithContext:(MNContext *)context;

- (void)setInputBuffer:(id<MTLBuffer>)buffer size:(CGSize)size;

@property (strong, nonatomic) id<MTLBuffer> outputBuffer;

@end

NS_ASSUME_NONNULL_END
