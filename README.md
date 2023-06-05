# ZEIN
Zig-based implementation of tensors

# Project Structure
The main ZEIN file provides an interface for the library implementation.

The implementations are in the Core folder. They will be labled as "VX" where X is the verion number.

This enables swapping out the implentation for both testing purposes and for providing variable behavior based on the Core version that is being used.
Additionally, this helps with backwards compatibility as this library may be aggressively changed. Likewise, some decisions may not be supported on all 
architectures (AVX or CUDA, for instance).

# Using Tensor Libray Functions
Currently, V1 requires AVX support. Accomodations will be provided as the library develops.

Tensors can be created in the following way:

This library currently supports tensors within rank [1, 64). 

```
const Tensor = @import("ZEIN/Zein.Zig").Tensor;
const Rowwise = @import("ZEIN/Zein.Zig").Rowwise;

// initialize underlying tensor memory:

var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

// create a rank 2, 3x3, Rowwise tensor of i32 from data:

var X = Tensor(i32, 2, Rowwise).init(
        &data, .{ 3, 3 }
    );    

// access the third element at [0, 2]:

const x = X.getValue(.{0, 2});

// transpose the tensor's view:

try x.permutate(.{1, 0}); // initially was {0, 1}...

```

Tensor Factories with Allocator support will be coming very soon! Use your desired
allocator to quickly create tensors, or initialize them from existing memory...
your choice!

Currently, tensor permutations only change the indexing of a tensor - they do not
invalidate underyling memory. There will be support for data transformations as well,
but they will be in the form of free functions with descriptive names. Until then,
tensor member functions do not mutate underlying memory, so different tensors can
view the same data in a variety of ways safely.

# Additonal functionality coming soon.
This library is still in the beginning phases. If you want to contribute, please
contact me! This is a big job and I'll take the help!
