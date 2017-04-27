// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import <simd/simd.h>

NS_ASSUME_NONNULL_BEGIN

@class MNContext;

@interface MNConstantWidthSparseMatrix : NSObject

- (instancetype)initWithContext:(MNContext *)context
                   numberOfRows:(ushort)rowsNumber width:(ushort)width;

@property (readonly, nonatomic) id<MTLBuffer> buffer;
@property (readonly, nonatomic) id<MTLBuffer> columnIndices;
@property (readonly, nonatomic) ushort rowsNumber;

- (void)insertElement:(float16_t)element row:(ushort)row index:(ushort)index column:(ushort)column;

@end

NS_ASSUME_NONNULL_END
