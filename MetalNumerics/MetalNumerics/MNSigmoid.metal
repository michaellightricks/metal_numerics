//
//  MNSigmoid.metal
//  MetalNumerics
//
//  Created by Michael Kupchick on 9/7/17.
//  Copyright © 2017 Michael Kupchick. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid(device float *buffer [[buffer(0)]],
                    device float *result [[buffer(1)]],
                    constant uint *bufferSize [[buffer(2)]],
                    uint threadInGroupIdx [[ thread_index_in_threadgroup ]],
                    uint groupIdx [[threadgroup_position_in_grid]],
                    uint gridDim [[threadgroups_per_grid]],
                    uint groupDim [[threads_per_threadgroup]]) {
  uint N = bufferSize[0];

  float sum = 0;
  for (uint i = groupIdx * groupDim + threadInGroupIdx; i < N; i += groupDim * gridDim) {
    sum += (1 + exp(buffer[i]));
    result[i] = sum;
  }


}
