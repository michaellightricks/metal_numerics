// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNKernel.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNAverageKernel : MNKernel

//@property (strong, nonatomic) id<MTLBuffer> buffer;
- (void)setInputBuffer:(id<MTLBuffer>)inputBuffer withElementsCount:(uint)count;

@property (strong, nonatomic) id<MTLBuffer> result;

+ (void)testWithContext:(MNContext *)context;

@end

NS_ASSUME_NONNULL_END
