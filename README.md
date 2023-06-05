# ZEIN
Zig-based implementation of tensors

# Using the Zein library

The main ZEIN/Zein.zig file provides an interface for the library implementation.

The implementations are in the Core folder. They will be labled as "VX" where X is the verion number.

This enables swapping out the implentation for providing variable behavior based on the Core version that is being used.
This helps with backwards compatibility as this library may be aggressively changed. Likewise, some decisions may not be supported on all 
architectures (AVX or CUDA, for instance).

Currently, V1 requires AVX support. Accomodations will be provided as the library develops.

# Using Tensor Objects

This library currently supports tensors within rank [1, 64). 

Tensors can be created in the following way:


```zig
const Tensor = @import("ZEIN/Zein.zig").Tensor;
const Rowwise = @import("ZEIN/Zein.zig").Rowwise;

// initialize underlying tensor memory:

var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

// create a rank 2, 3x3, Rowwise tensor of i32 from data:

var X = Tensor(i32, 2, Rowwise).init(
        &data, .{ 3, 3 }
    );    

const x = X.getValue(.{0, 2}); // access value 3...

try X.permutate(.{1, 0}); // transpose tensor...
```

# Allocating Tensor Data
Using the TensorAllocator is easy and intuitive and designed to work with Tensor objects:

```zig
var GPA = std.heap.GeneralPurposeAllocator(.{ }){ };

// Accepts standard conforming allocator.
var allocator = TensorAllocator(f32).init(GPA.allocator());

// Option 1: assign memory into existing tensor.
var X = Tensor(f32, 2, Rowwise).init(null, .{ 10, 10 });

try allocator.allocToTensor(&X); // alloc 100 elements...

// Option 2: assign a new tensor from allocator.
var Y = try allocator.allocTensor(2, Rowwise, .{10, 10});

// Deallocate tensor values...
allocator.freeFromTensor(&X); // free and reset X...
allocator.freeFromTensor(&Y); // free and reset Y...
```

# Memory owernship and viewership
Currently, tensor permutations only change the indexing of a tensor - they do not
invalidate underyling memory. There will be support for data transformations as well,
but they will be in the form of free functions with descriptive names. Until then,
tensor member functions do not mutate underlying memory, so different tensors can
view the same data in a variety of ways safely.

# Additonal functionality coming soon.
This library is still in the beginning phases. If you want to contribute, please
contact me! This is a big job and I'll take the help!
