// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNConstantWidthSparseMatrix.h"

#import "MNContext.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNConstantWidthSparseMatrix ()

@property (readonly, nonatomic) ushort groupsNumber;

@end

@implementation MNConstantWidthSparseMatrix

- (instancetype)initWithContext:(MNContext *)context numberOfRows:(ushort)rowsNumber
                          width:(ushort)width {
  if (self = [super init]) {
    ushort reminder = width % 4;
    _rowsNumber = rowsNumber;
    _groupsNumber = width / 4 + (reminder ? 1 : 0);
    _buffer = [context.device
               newBufferWithLength:rowsNumber * self.groupsNumber * sizeof(simd::float4)
               options:MTLResourceStorageModeShared];
    _columnIndices = [context.device
                      newBufferWithLength:rowsNumber * self.groupsNumber * sizeof(simd::ushort4)
                      options:MTLResourceStorageModeShared];
  }

  return self;
}

- (int)widthInBytes:(ushort)width {
  ushort float4num = width / 4;
  if (width % 4 != 0) {
    float4num += 1;
  }

  return float4num * sizeof(simd::float4);
}

- (void)insertElement:(float)element row:(ushort)row index:(ushort)index column:(ushort)column {
  float *bufferPtr = (float *)self.buffer.contents;
  float *elementPtr = bufferPtr + row * self.groupsNumber * 4 + index;
  *elementPtr = element;
  ushort *indeciesPtr = (ushort *)self.columnIndices.contents;
  ushort *indexPtr = indeciesPtr + row * self.groupsNumber * 4 + index;
  *indexPtr = column;
}

@end

NS_ASSUME_NONNULL_END
