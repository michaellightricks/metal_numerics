// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNKernel.h"

@class MNContext, MTLBuffer;

NS_ASSUME_NONNULL_BEGIN

@interface MNSigmoidKernel : MNKernel

- (void)setInputBuffer:(id<MTLBuffer>)inputBuffer withElementsCount:(uint)count;

@property (strong, nonatomic) id<MTLBuffer> result;

@property (readonly, nonatomic) uint elementsNumber;

+ (void)testWithContext:(MNContext *)context;

@end

NS_ASSUME_NONNULL_END
