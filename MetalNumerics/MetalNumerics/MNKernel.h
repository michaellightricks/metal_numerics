//
//  MNKernel.h
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

@class MNContext;

@interface MNKernel : NSObject

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithContext:(MNContext *)context functionName:(NSString *)name
    NS_DESIGNATED_INITIALIZER;

- (void)run;

- (id<MTLCommandBuffer>)enque;

- (void)enqueTo:(id<MTLCommandBuffer>)commandBuffer;

/// protected abstract
- (void)configureCommandEncoder:(id<MTLComputeCommandEncoder>)encoder;

- (MTLSize)threadGroupSize;

- (MTLSize)threadGroupsCount:(MTLSize)threadGroupSize;

@property (readonly, nonatomic) MNContext *context;

@property (nonatomic, strong) id<MTLComputePipelineState> pipeline;

@property (nonatomic, strong) id<MTLFunction> kernelFunction;

@end
