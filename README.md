# ZEIN
Zig-based implementation of general-rank tensors! [1, 64)

# Using Tensor Objects
Tensors can be created in the following way:

```zig
// initialize underlying tensor memory:
var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

// create a rank 2, 3x3, Rowwise tensor of i32 from data:
var X = zein.Tensor(i32, 2, Rowwise).init(&data, .{ 3, 3 });    

const x = X.getValue(.{0, 2}); // access value 3...
```

# Allocating Tensor Data
The TensorFactory offers the ability to track and free allocations:

```zig
// null will automatically use the GPA allocator:
var factory = zein.TensorFactory(f32).init(null);

// Begin tracking tensor allocations (default is no-tracking):
factory.tracking(.start);

// Stop tracking tensor allocations (does not free tensors):
factory.tracking(.stop);

// Free tracked tensor allocations (no-op if no tensors are tracked):
factory.tracking(.free);

// Deinit will free the allocator and currently tracked tensors:
factory.deinit();
````
```zig
// Assign a new tensor from allocator:
var Y = try factory.allocTensor(2, Rowwise, .{10, 10});
```
```zig
// Assign memory into existing tensor:
var X = Tensor(f32, 2, Rowwise).init(null, .{ 10, 10 });

try factory.allocToTensor(&X); // alloc 100 elements...
````

# Tensor Operations
Tensor operations are are in the form of either _Free Functions_ or _Factory Functions_:

        - Free Functions require operands and the destination tensor.

        - Factory Functions use operands to create the destination tensor.

The operations use compile time strings as einsum notation:

```zig
// Transpose/permutate tensor views (does not modify underlying data).
var y = try x.permutate("ijk->kji");
```
```zig
// Collapse tensor values using contraction:
try zein.contraction("ijk->ji", &x, &y);
```
```zig
// Elementary vectorized reduction functions (sum, product, min, max):
const s = try zein.sum(&x);
```

# Using the Zein library
The main ZEIN/Zein.zig file provides an interface for the library implementation.

The implementations are in the Core folder. They will be labled as "VX" where X is the verion number.

Most function in the Zein library will have default and "unchecked" versions.

Default functions ensure operational correctness while unchecked versions do not.

Unchecked versions are provided for api-level optimizations where performance matters.

# Memory Owernship and Viewership
Currently, tensor permutations only change the indexing of a tensor - they do not
invalidate underyling memory. If the user chooses to use the TensorFactory,
it will track allocations and delete them automatically when calling deinit.
V1 is only tested on single thread environments - thread safety with allocators
will be coming in a later version!

# Additonal functionality coming soon.
This library is still in the beginning phases. If you want to contribute, please
contact me! This is a big job and I'll take the help!
