//
//  SharedStructures.h
//  MetalNumerics
//
//  Created by Michael Kupchick on 22/06/2016.
//  Copyright (c) 2016 Michael Kupchick. All rights reserved.
//

#ifndef SharedStructures_h
#define SharedStructures_h

#include <simd/simd.h>

typedef struct
{
    matrix_float4x4 modelview_projection_matrix;
    matrix_float4x4 normal_matrix;
} uniforms_t;

#endif /* SharedStructures_h */

