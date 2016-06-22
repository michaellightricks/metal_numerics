//
//  MNSparseMatrix.h
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import <simd/simd.h>

@interface MNCRSSparseMatrix : NSObject

@property (readonly, nonatomic) MTLSize realSize;

@property (readonly, nonatomic) id<MTLBuffer> buffer;
@property (readonly, nonatomic) id<MTLBuffer> columnIndices;
@property (readonly, nonatomic) id<MTLBuffer> nonZeroElementsNumberInRow;

@end
