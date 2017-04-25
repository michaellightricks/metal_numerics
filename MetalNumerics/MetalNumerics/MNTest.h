// Copyright (c) 2017 Lightricks. All rights reserved.
// Created by Michael Kupchick.

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class MNContext;

@interface MNTest : NSObject

- (instancetype)initWithContext:(MNContext *)context;

- (void)run;

@end

NS_ASSUME_NONNULL_END
