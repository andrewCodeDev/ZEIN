
// DESIGN PHILOSOPHY June 6th, 2023 //

// The TensorOps class is a general interface for common tensor operations.
// It has an "OpsPolicy" which will allow for users to quickly switch between
// modes. For instance, the policy (with a single field) can turn on validation
// checks and be turned off after for higher performance. Since these are comptime
// parameters, the checks will get compiled out during runtime.

// The goal for V1 is simple. Provide reliable (albeit naive) functionality
// that focuses on correctness first. Once that is established, V2 can use
// V1 as a reference for future versions, therefore creating a baseline
// for correctness. As such, the current goal is to provide a complete set
// of functionalities and replace them with more optimal solutions over time.


const Tensor = @import("Tensor.zig").Tensor;
const TensorError = @import("Tensor.zig").TensorError;
const TensorAllocator = @import("TensorAllocator.zig").TensorAllocator;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;
const SizeAndStrideType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;
const defaultPermuation = @import("SizesAndStrides.zig").defaultPermutation;

const ReduceOp = @import("std").builtin.ReduceOp;
const mem = @import("std").mem;

// The OpsPolicy controls the behavior of the math
// class. This includes behavior such as whether or
// not the class can allocate new tensors if output
// tensors are are not provided

pub const OpsPolicy = struct {
    // Flag to allocate more scratch memory.
    // Some ops work best with scratch memory.
    alloc_scratch: bool = true,

    // Flag to check arguments for validity.
    validate_args: bool = true,
};

pub fn TensorOps(comptime alloc_type: type, comptime policy: OpsPolicy) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        const SizesType = SizeAndStrideType;

        const AllocType = alloc_type;

        const Policy = policy;

        // The allocator data member is here incase
        // a user does not provide enough memory
        allocator: TensorAllocator(AllocType),

        // Scratch memory for operations
        scratch: []AllocType = &[_]AllocType{ },

        pub fn init(allocator: TensorAllocator(AllocType)) Self {
            return Self { .allocator = allocator };
        }

        pub fn scratchSize(self: ConstSelfPtr) usize {
            return self.*.scratch.len;
        }

        pub fn releaseScratch(self: SelfPtr) []AllocType {
            var tmp = self.*.scratch;
            self.*.scratch = &[_]AllocType{};
            return tmp;
        }        

        pub fn resizeScratch(self: SelfPtr, size: usize) !void {
            if(self.*.scratch.len == size) {
                return;
            }
            if(self.*.scratchSize() != 0) {
                self.*.allocator.freeValues(self.*.releaseScratch());
            }
            self.*.scratch = try self.*.allocator.allocValues(size);
        }

        pub fn deinit(self: SelfPtr) void {
            if(0 < self.*.scratchSize()) {
                self.*.allocator.freeValues(self.*.releaseScratch());
            }
        }

        //pub fn add(self: SelfPtr, X: anytype, Y: anytype, Z: anytype) !void {
        //    _ = self;
        //    const XT = @TypeOf(X.*);
        //    const YT = @TypeOf(Y.*);

        //    if(XT != YT) {
        //        @compileError("Cannot add tensors of different types.");
        //    }
        //    if(XT.ValueType != AllocType) {
        //        @compileError("Cannot add tensors of different value types.");
        //    }
        //    if(Policy.validate_args) {
        //        try expect(X.*.isValid() and Y.*.isValid() and Z.*.isValid());
        //        try expect(X.*.valueSize() == Y.*.valueSize());
        //        try expect(X.*.valueSize() == Z.*.valueSize());
        //    }            
        //    @compileError("Needs Implementation");
        //}

        //pub fn multiply(self: SelfPtr, X: anytype, Y: anytype, Z: anytype) !void {
        //    _ = self;
        //    const XT = @TypeOf(X.*);
        //    const YT = @TypeOf(Y.*);

        //    if(XT != YT) {
        //        @compileError("Cannot multiply tensors of different types.");
        //    }
        //    if(XT.ValueType != AllocType) {
        //        @compileError("Cannot multiply tensors of different value types.");
        //    }
        //    if(Policy.validate_args) {
        //        try expect(X.*.isValid() and Y.*.isValid() and Z.*.isValid());
        //        try expect(X.*.valueSize() == Y.*.valueSize());
        //        try expect(X.*.valueSize() == Z.*.valueSize());
        //    }
        //    @compileError("Needs Implementation");
        //}

        // True to the name, this function will permutate the values of your tensor
        // but not the tensor sizes and strides. Because of this, any tensor that
        // also reference the underlying memory will be effected as well.

        pub fn permutateValues(self: SelfPtr, x: anytype, permutation: [@TypeOf(x.*).Rank]SizesType) !void {
            const XT = @TypeOf(x.*);

            // If we do not leave the input tensor's sizes and strides alone, then it will
            // return the same value for a given index. This is because it's new permutated
            // layout will cancel out the effect of permutating the values.

            //var tmp = x.*.sizes_and_strides;

            if(Policy.validate_args) {
                if (!x.*.isValid()) { 
                    return TensorError.InvalidTensorLayout; 
                }
                try x.*.permutate(permutation);
            }
            else {
                x.*.permutateUnchecked(permutation);
            }
            
            if(Policy.alloc_scratch) {
                // check if we have enough scratch memory
                if(self.*.scratchSize() < x.*.valueSize()){
                    try self.*.resizeScratch(x.*.valueSize());
                }

                // for the V1 naive implementation, this will be
                // the array that caries forward the indicies when
                // we inline the recursive loops.
                var indices: [XT.Rank]SizesType = undefined;

                // counter for iterating through the scratch memory
                var counter: XT.SizesType = 0;

                @call(.always_inline, recursivePermutateValues, .{
                     XT.ValueType, SizesType, XT.Rank, 0, x, self.*.scratch, &indices, &counter
                });

                @memcpy(x.*.values, self.*.scratch[0..x.*.valueSize()]);
            }
            else {
                @compileError("Non-scratch memory version of permutateValues is not implemented.");
            } 
        }
    };
}

/////////////////////////////////////////////////////////////
// This is the naive version of a general tensor permutation.
// In the future, I plan on making more optimal versions of
// this, but it's reliable baseline for future work.
//
// If all goes well, it will unroll to something like this:
//
//    for i..I
//        indices[0] = i
//        for j..J
//            indices[1] = j
//                ...
//                for n..N
//                    scratch[count] = x.getValue(indices);
//                    count += 1
//

inline fn recursivePermutateValues(
    comptime VT: type, // value type
    comptime IT: type, // int type
    comptime R: usize, // tensor rank
    comptime I: usize, // starting index
    x: anytype, // source tensor
    y: []VT, // destination memory
    c: *[R]IT, // index container
    n: *IT // scratch counter
) void {

    if(I == (R - 1)) {     
        // we only need to make this once really...
        const x_ss : @Vector(R, IT) = x.*.sizes_and_strides.strides;

        var i: IT = 0;
        var n_i = n.*;
        while(i < x.*.getSize(I)) : ({ i += 1; n_i += 1; }) {

            c[I] = i;
            const x_c : @Vector(R, IT) = c.*;
            const x_i = @reduce(ReduceOp.Add, x_c * x_ss);

            y[n_i] = x.*.values[x_i];
        }
        n.* += i;
    }

    else {
        var i: IT = 0;
        while(i < x.*.getSize(I)) : (i += 1) {
            
            c[I] = i; 
            
            @call(.always_inline, recursivePermutateValues, .{
                 VT, IT, R, (I + 1), x, y, c, n 
            });
        }
    }
}

fn vectorizedReduce(
    comptime N: usize, 
    comptime ReduceType: anytype, 
    comptime ScalarFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
    ) @TypeOf(x.*).ValueType {

    const T = @TypeOf(x.*).ValueType;

    var i: usize = 0;
    var rdx = init;

    // reduce in size N chunks...
    while((i + N) < x.*.valueSize()) : (i += N) {
        const slice = x.*.values[i..N + i];
        const vec: @Vector(N, T) =  slice[0..N].*; // needs compile time length
        rdx = @call(.always_inline, ScalarFunc, .{ rdx, @reduce(ReduceType, vec) });
    }

    // reduce remainder...
    while(i < x.*.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, ScalarFunc, .{ rdx, x.*.values[i] });
    }
    return rdx;
}

// add for testing reduce
inline fn add(x: anytype, y: anytype) @TypeOf(x) {
    return x + y;
}

// mul for testing reduce
inline fn mul(x: anytype, y: anytype) @TypeOf(x) {
    return x * y;
}

// div for testing reduce
inline fn div(x: anytype, y: anytype) @TypeOf(x) {
    return x / y;
}

// mul for testing reduce
inline fn max(x: anytype, y: anytype) @TypeOf(x) {
    return @max(x, y);
}

// mul for testing reduce
inline fn min(x: anytype, y: anytype) @TypeOf(x) {
    return @min(x, y);
}

test "vectorized reduce" {
    const std = @import("std");
    
    var GPA = std.heap.GeneralPurposeAllocator(.{ }){ };
    
    var factory = TensorAllocator(i32).init(GPA.allocator());
    
    var x = try factory.allocTensor(2, Rowwise, .{ 100, 100 });
    
    @memset(x.values, 1);

    { // reduce sum of 10'000 elements
        const y = vectorizedReduce(512, ReduceOp.Add, add, &x, 0);
        try std.testing.expectEqual(y, 10000);
    }
    { // reduce product of 10'000 elements
        const y = vectorizedReduce(512, ReduceOp.Mul, mul, &x, 1);
        try std.testing.expectEqual(y, 1);
    }
    { // reduce max of 10'000 elements
        x.setValue(999, .{24, 62});
        const y = vectorizedReduce(512, ReduceOp.Max, max, &x, std.math.minInt(i32));
        try std.testing.expectEqual(y, 999);
    }
    { // reduce max of 10'000 elements
        x.setValue(-999, .{92, 10});
        const y = vectorizedReduce(512, ReduceOp.Min, min, &x, std.math.maxInt(i32));
        try std.testing.expectEqual(y, -999);
    }
    factory.freeFromTensor(&x);

    if (GPA.deinit() == .leak) { @panic("LEAK DETECTED"); }
}

test "Permutate Values" {
    const std = @import("std");
    
    var GPA = std.heap.GeneralPurposeAllocator(.{ }){ };
    defer if (GPA.deinit() == .leak) @panic("LEAK DETECTED");
    
    var factory = TensorAllocator(i32).init(GPA.allocator());

    var ops = TensorOps(i32, .{}).init(factory);
    defer ops.deinit();

    // need a more convincing test
    var x = try factory.allocTensor(2, Rowwise, .{ 3, 3 });
    defer factory.freeFromTensor(&x);

    var i: i32 = 1;
    for(x.values) |*v| { v.* = i; i += 1; }

    try ops.permutateValues(&x, .{1, 0});

    // basic MxN -> NxM transpose
    try std.testing.expectEqual(x.values[0], 1);
    try std.testing.expectEqual(x.values[1], 4);
    try std.testing.expectEqual(x.values[2], 7);
    try std.testing.expectEqual(x.values[3], 2);
    try std.testing.expectEqual(x.values[4], 5);
    try std.testing.expectEqual(x.values[5], 8);
    try std.testing.expectEqual(x.values[6], 3);
    try std.testing.expectEqual(x.values[7], 6);
    try std.testing.expectEqual(x.values[8], 9);

}
