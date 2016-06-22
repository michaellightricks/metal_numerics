//
//  MNContext.m
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright © 2016 Michael Kupchick. All rights reserved.
//

#import "MNContext.h"

@implementation MNContext

- (instancetype)initWithDevice:(id<MTLDevice>)device {
  if (self = [super init]) {
    _device = device;
    _queue = [device newCommandQueue];
    _library = [device newDefaultLibrary];
  }

  return self;
}

@end
