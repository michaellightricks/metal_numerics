// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import "MNConstantWidthSparseMatrix.h"

#import "MNContext.h"

NS_ASSUME_NONNULL_BEGIN

@interface MNConstantWidthSparseMatrix ()

@property (readonly, nonatomic) ushort width;

@end

@implementation MNConstantWidthSparseMatrix

- (instancetype)initWithContext:(MNContext *)context numberOfRows:(ushort)rowsNumber
                          width:(ushort)width {
  if (self = [super init]) {
    _rowsNumber = rowsNumber;
    _width = width;
    _buffer = [context.device
               newBufferWithLength:rowsNumber * width * sizeof(float16_t)
               options:MTLResourceStorageModeShared];
    _columnIndices = [context.device
                      newBufferWithLength:rowsNumber * width * sizeof(ushort)
                      options:MTLResourceStorageModeShared];
  }

  return self;
}

- (void)insertElement:(float16_t)element row:(ushort)row index:(ushort)index column:(ushort)column {
  float16_t *bufferPtr = (float16_t *)self.buffer.contents;
  float16_t *elementPtr = bufferPtr + row * self.width + index;
  *elementPtr = element;
  ushort *indeciesPtr = (ushort *)self.columnIndices.contents;
  ushort *indexPtr = indeciesPtr + row * self.width + index;
  *indexPtr = column;
}

@end

NS_ASSUME_NONNULL_END
