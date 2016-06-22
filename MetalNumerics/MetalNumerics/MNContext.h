//
//  MNContext.h
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import <Metal/Metal.h>

#import <Foundation/Foundation.h>

@interface MNContext : NSObject

- (instancetype)initWithDevice:(id<MTLDevice>)device;

@property (readonly, nonatomic) id<MTLDevice> device;
@property (readonly, nonatomic) id<MTLCommandQueue> queue;
@property (readonly, nonatomic) id<MTLLibrary> library;

@end
