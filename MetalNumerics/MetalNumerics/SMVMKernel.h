//
//  SMVMKernel.h
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import <Foundation/Foundation.h>

#import "MNKernel.h"

@class MNContext;
@class MNCRSSparseMatrix;

@interface SMVMKernel : MNKernel

- (instancetype)initWithContext:(MNContext *)context;

@property (strong, nonatomic) MNCRSSparseMatrix *matrix;
@property (strong, nonatomic) id<MTLBuffer> vector;
@property (strong, nonatomic) id<MTLBuffer> result;

@end
