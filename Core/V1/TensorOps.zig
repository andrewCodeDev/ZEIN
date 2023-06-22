
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
const TensorFactory = @import("TensorFactory.zig").TensorFactory;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;
const SizeAndStrideType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;
const defaultPermuation = @import("SizesAndStrides.zig").defaultPermutation;
var GPA = @import("TensorFactory.zig");
const ReduceOp = @import("std").builtin.ReduceOp;
const math = @import("std").math;

// The OpsPolicy controls the behavior of the math
// class. This includes behavior such as whether or
// not the class can allocate new tensors if output
// tensors are are not provided

const OpsError = error {
    UnequalSize,
    InvalidDimensions,
    InvalidSizes,
    SizeZeroTensor
};

pub const OpsPolicy = struct {
    // Flag to allocate more scratch memory.
    // Some ops work best with scratch memory.
    alloc_scratch: bool = true,

    // Flag to check arguments for validity.
    validate_args: bool = true,
};

// This function detects which kind of number we're using and then
// and then returns the proper initialization value for it.

inline fn initValue(comptime op: ReduceOp, comptime T: type) T {

    const c = @typeName(T)[0];

    if(op == ReduceOp.Add) {
        return 0; // implicit cast
    }
    else if(op == ReduceOp.Mul) {
        return 1; // implicit cast
    }
    else if(op == ReduceOp.Min) {
        if(c == 102) { // "f"
            return math.floatMax(T);
        } else {
            return math.maxInt(T);
        }
    }
    else if(op == ReduceOp.Max) {
        if(c == 102) { // "f"
            return math.floatMin(T);
        } else {
            return math.minInt(T);
        }
    }
    else {
        @compileError("Unknown Operation type for initial value.");
    }
}

///////////////////////////////////////////////////////////
// So... these functions are interesting for a few reasons:
//
// Sum will return zero if the tensor length is zero... so that makes some sense...
// Product, however, really shouldn't return one (the init value), it should also return
// zero. Min and max however are a bit worse though. What's a good value to return if
// they fail? It seems odd to return the init value (lowest or highest possible value)
// and only some types support infinity.

// It's also annoying to put a try statement in front of each of these but that ends
// up being the case anyway. If you get back a bogus value, you have to check it
// already. That's what happens in the C++ standard; they try to get around this
// by returning the element's position but because of that, you always have to check
// if you got a pointer to the end of the container.

// In short, there isn't a good answer. All of them kinda suck. Because of that,
// I'm going to just return an error because at least then, you don't have to
// follow all of these with some if statement to make sure they didn't return
// garbage values. Also, like with the case of sum, you can make an argument
// that it can't fail but then you have to know which ones return errors 
// and which ones don't... I'm choosing consistency.

pub fn sum(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
        return reduceDispatch(ReduceOp.Add, addScalar, x, initValue(ReduceOp.Add, @TypeOf(x.*).ValueType));
    } else {
         return OpsError.SizeZeroTensor;
    }
}
pub fn product(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
        return reduceDispatch(ReduceOp.Mul, mulScalar, x, initValue(ReduceOp.Mul, @TypeOf(x.*).ValueType));
    } else {
        return OpsError.SizeZeroTensor;
    }
}
pub fn min(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
        return reduceDispatch(ReduceOp.Min, minScalar, x, initValue(ReduceOp.Min, @TypeOf(x.*).ValueType));
    } else {
        return OpsError.SizeZeroTensor;
    }
}
pub fn max(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
       return reduceDispatch(ReduceOp.Max, maxScalar, x, initValue(ReduceOp.Max, @TypeOf(x.*).ValueType));
    } else {
        return OpsError.SizeZeroTensor;
    }
}

// To complete the set, for those who like to live dangerously... the unchecked versions.

pub fn sumUnchecked(x: anytype) !@TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Add, addScalar, x, initValue(ReduceOp.Add, @TypeOf(x.*).ValueType));
}
pub fn productUnchecked(x: anytype) !@TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Mul, mulScalar, x, initValue(ReduceOp.Mul, @TypeOf(x.*).ValueType));
}
pub fn minUnchecked(x: anytype) !@TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Min, minScalar, x, initValue(ReduceOp.Min, @TypeOf(x.*).ValueType));
}
pub fn maxUnchecked(x: anytype) !@TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Max, maxScalar, x, initValue(ReduceOp.Max, @TypeOf(x.*).ValueType));
}

pub fn fill(
    x: anytype, 
    init: @TypeOf(x.*).ValueType,
    step: @TypeOf(x.*).ValueType
    ) void {
    var incr = init;
    for(x.values) |*value| {
        value.* = incr;
        incr += step;
    }
}

// The tensor Ops class is a heavier weight object. Some operatoins
// fundamentally do best with scratch memory, especially as we move
// towards gpu implementations. This class will handle any operations
// that require scratch memory so the user doesn't have to babysit
// the allocator. The OpsPolicy will determine what kinds of checks
// need to be done.

pub fn TensorOps(comptime value_type: type, comptime policy: OpsPolicy) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        const SizesType = SizeAndStrideType;

        const ValueType = value_type;

        const Policy = policy;

        // The allocator data member is here incase
        // a user does not provide enough memory
        allocator: *TensorFactory(ValueType),

        // Scratch memory for operations
        scratch: []ValueType = &[_]ValueType{ },

        alloc_index: ?usize,

        pub fn init(allocator: *TensorFactory(ValueType)) Self {
            return Self { .allocator = allocator, .alloc_index = null };
        }

        pub fn scratchSize(self: ConstSelfPtr) usize {
            return self.scratch.len;
        }

        pub fn releaseScratch(self: SelfPtr) []ValueType {
            var tmp = self.scratch;
            self.scratch = &[_]ValueType{};
            return tmp;
        }        

        pub fn resizeScratch(self: SelfPtr, size: usize) !void {
            if(self.scratch.len == size) {
                return;
            }
            if(self.alloc_index) |i| {
                try self.allocator.freeValues(self.scratch, i);
            }
            var indexed_alloc = try self.allocator.allocValues(
                size, self.alloc_index
            );
            self.alloc_index = indexed_alloc.index;
            self.scratch = indexed_alloc.alloc;
        }

        //fn add(self: SelfPtr, X: anytype, Y: anytype, Z: anytype) !void {
        //    const XT = @TypeOf(X.*);
        //    const YT = @TypeOf(Y.*);

        //    if(XT != YT) {
        //        @compileError("Cannot add tensors of different types.");
        //    }
        //    if(XT.ValueType != ValueType) {
        //        @compileError("Cannot add tensors of different value types.");
        //    }
        //    if(Policy.validate_args) {
        //        if(!(X.isValid() and Y.isValid() and Z.isValid())){ 
        //            return TensorError.InvalidTensorLayout; 
        //        }
        //        if(X.valueSize() != Y.valueSize() or X.valueSize() != Z.valueSize()) {
        //             return OpsError.UnequalSize; 
        //        }
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
        //    if(XT.ValueType != ValueType) {
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
//
//        pub fn permutateValues(self: SelfPtr, x: anytype, permutation: [@TypeOf(x.*).Rank]SizesType) !void {
//            const XT = @TypeOf(x.*);
//
//            // If we do not leave the input tensor's sizes and strides alone, then it will
//            // return the same value for a given index. This is because it's new permutated
//            // layout will cancel out the effect of permutating the values.
//
//            var tmp = x.*;
//
//            if(Policy.validate_args) {
//                if (!tmp.isValid()) { 
//                    return TensorError.InvalidTensorLayout; 
//                }
//                try tmp.permutate(permutation);
//            }
//            else {
//                tmp.permutateUnchecked(permutation);
//            }
//            
//            if(Policy.alloc_scratch) {
//                // check if we have enough scratch memory
//                if(self.scratchSize() < tmp.valueSize()){
//                    try self.resizeScratch(tmp.valueSize());
//                }
//
//                // for the V1 naive implementation, this will be
//                // the array that caries forward the indicies when
//                // we inline the recursive loops.
//                var indices: [XT.Rank]SizesType = undefined;
//
//                // counter for iterating through the scratch memory
//                var counter: XT.SizesType = 0;
//
//                @call(.always_inline, recursivePermutateValues, .{
//                     XT.ValueType, SizesType, XT.Rank, 0, &tmp, self.scratch, &indices, &counter
//                });
//
//                @memcpy(tmp.values, self.scratch[0..tmp.valueSize()]);
//            }
//            else {
//                @compileError("Non-scratch memory version of permutateValues is not implemented.");
//            } 
//        }
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

inline fn recursivePermutate(
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
            
            @call(.always_inline, recursivePermutate, .{
                 VT, IT, R, (I + 1), x, y, c, n 
            });
        }
    }
}

/////////////////////////////////////////////////////////////
// This is the naive version of a general tensor contraction.
// In the future, I plan on making more optimal versions of
// this, but it's reliable baseline for future work.
//
// If all goes well, it will unroll to something like this:
//
//    for i..I
//        x_indices[0] = i
//        y_indices[0] = i
//        for j..J
//            x_indices[1] = j
//            y_indices[1] = j
//                ...
//                for n..N
//                    x_indices[I] = n;
//                    y[y_indices] += x.getValue(x_indices);

///////////////////////////////////////////////////////
// THIS FUNCTION IS STILL EXPERIMENTAL (testing soon).

const contractionParse = @import("ExpressionParsing.zig").contractionParse;

pub fn contraction(comptime expression: [] const u8, x: anytype, y: anytype) !void {

    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ip = contractionParse(XT.Rank, YT.Rank, expression);

    const xs = x.getSizes();
    const ys = y.getSizes();

    var i: usize = 1; 
    while(i < YT.Rank) : (i += 1) {
        if(xs[ip.lhs[i]] != ys[ip.rhs[i]]) {
            return OpsError.InvalidSizes;
        }
    }
    var xc: [XT.Rank]XT.SizesType = undefined;
    var yc: [YT.Rank]YT.SizesType = undefined;
    
    @memset(y.values, 0);
    
    @call(.always_inline, recursiveContraction, .{
        XT.ValueType, XT.SizesType, XT.Rank, YT.Rank, 0, x, y, &xc, &yc, &ip.lhs, &ip.rhs
    });
}

pub fn contractionUnchecked(comptime expression: [] const u8, x: anytype, y: anytype) !void {

    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ip = contractionParse(XT.Rank, YT.Rank, expression);

    var xc: [XT.Rank]XT.SizesType = undefined;
    var yc: [YT.Rank]YT.SizesType = undefined;
    
    @memset(y.values, 0);
    
    @call(.always_inline, recursiveContraction, .{
        XT.ValueType, XT.SizesType, XT.Rank, YT.Rank, 0, x, y, &xc, &yc, &ip.lhs, &ip.rhs
    });
}

inline fn recursiveContraction(
    comptime VT: type, // value type
    comptime IT: type, // int type
    comptime XR: usize, // tensor x rank
    comptime YR: usize, // tensor y rank
    comptime I: usize, // starting index
    x: anytype, // source tensor
    y: anytype, // destination memory
    xc: *[XR]IT, // index container
    yc: *[YR]IT, // index container
    xp: *const [XR]IT, // contraction indices
    yp: *const [YR]IT // contraction indices
) void {

    if(XR <= YR) {
        @compileError("Contraction must go from a larger tensor to a smaller one.");
    }

    if(I < YR) {

        const x_perm_index = xp[I];
        const y_perm_index = yp[I];

        // this first branch loads up the x and y indices
        // and passes them to the next loop. In this case,
        // I is still in bounds of both x and y ranks.
        
        var i: IT = 0;
        while(i < x.getSize(x_perm_index)) : (i += 1) {
            
            xc[x_perm_index] = i; 
            yc[y_perm_index] = i; 
            
            @call(.always_inline, recursiveContraction, .{
                 VT, IT, XR, YR, (I + 1), x, y, xc, yc, xp, yp
            });
        }
    }

    else if ((YR <= I) and (I < (XR - 1))) {

        // the second branch deals with values of I that are
        // out-of-bounds for y rank, but still in-bounds for
        // the x rank.

        const x_perm_index = xp[I];
        
        var i: IT = 0;
        while(i < x.getSize(x_perm_index)) : (i += 1) {
            
            xc[x_perm_index] = i; 
            
            @call(.always_inline, recursiveContraction, .{
                 VT, IT, XR, YR, (I + 1), x, y, xc, yc, xp, yp
            });
        }
    }

    else {

        // the third branch deals with summing up the contracted
        // indices and writing them to the related y index
        
        const x_ss : @Vector(XR, IT) = x.*.sizes_and_strides.strides;

        const x_perm_index = xp[I];

        var i: IT = 0;
        var t: VT = 0;
        while(i < x.getSize(x_perm_index)) : (i += 1) {
            xc[x_perm_index] = i;
            const x_c : @Vector(XR, IT) = xc.*;
            const x_i = @reduce(ReduceOp.Add, x_c * x_ss);
            t += x.values[x_i]; // accumulate summations
        }
        const y_ss : @Vector(YR, IT) = y.sizes_and_strides.strides;
        const y_c : @Vector(YR, IT) = yc.*;
        const y_i = @reduce(ReduceOp.Add, y_c * y_ss);
        y.*.values[y_i] += t;
    }
}

fn loopReduce(
    comptime ScalarFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
    ) @TypeOf(x.*).ValueType {
    var i: usize = 0;
    var rdx = init;
    while(i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, ScalarFunc, .{ rdx, x.values[i] });
    }
    return rdx;
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
    while((i + N) < x.valueSize()) : (i += N) {
        const slice = x.values[i..N + i];
        const vec: @Vector(N, T) =  slice[0..N].*; // needs compile time length
        rdx = @call(.always_inline, ScalarFunc, .{ rdx, @reduce(ReduceType, vec) });
    }
    // reduce remainder...
    while(i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, ScalarFunc, .{ rdx, x.values[i] });
    }
    return rdx;
}

fn reduceDispatch(    
    comptime ReduceType: anytype, 
    comptime ScalarFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
) @TypeOf(x.*).ValueType {

    const size = x.*.valueSize();

    if(size < 128) {
        return loopReduce(ScalarFunc, x, init);
    }
    else if(size < 256) {
        return vectorizedReduce(128, ReduceType, ScalarFunc, x, init);
    }
    else if(size < 512) {
        return vectorizedReduce(256, ReduceType, ScalarFunc, x, init);
    }
    else {
        return vectorizedReduce(512, ReduceType, ScalarFunc, x, init);
    }
}

inline fn addScalar(x: anytype, y: anytype) @TypeOf(x) {
    return x + y;
}
inline fn mulScalar(x: anytype, y: anytype) @TypeOf(x) {
    return x * y;
}
inline fn divScalar(x: anytype, y: anytype) @TypeOf(x) {
    return x / y;
}
inline fn maxScalar(x: anytype, y: anytype) @TypeOf(x) {
    return @max(x, y);
}
inline fn minScalar(x: anytype, y: anytype) @TypeOf(x) {
    return @min(x, y);
}

test "vectorized reduce" {
    const std = @import("std");

    var factory = TensorFactory(i32).init(null);
    
    var x = try factory.allocTensor(2, Rowwise, .{ 100, 100 });
    
    @memset(x.values, 1);

    { // reduce sum of 10'000 elements
        const y = try sum(&x);
        try std.testing.expectEqual(y, 10000);
    }
    { // reduce product of 10'000 elements
        const y = try product(&x);
        try std.testing.expectEqual(y, 1);
    }
    { // reduce max of 10'000 elements
        x.setValue(999, .{24, 62});
        const y = try max(&x);
        try std.testing.expectEqual(y, 999);
    }
    { // reduce max of 10'000 elements
        x.setValue(-999, .{92, 10});
        const y = try min(&x);
        try std.testing.expectEqual(y, -999);
    }
    factory.deinit();
}

//test "Permutate Values" {
//    const std = @import("std");
//
//    var factory = TensorFactory(i32).init(null);
//
//    var ops = TensorOps(i32, .{}).init(&factory);
//
//    // need a more convincing test
//    var x = try factory.allocTensor(2, Rowwise, .{ 3, 3 });
//
//    var i: i32 = 1;
//    for(x.values) |*v| { v.* = i; i += 1; }
//
//    try ops.permutateValues(&x, .{1, 0});
//
//    // basic MxN -> NxM transpose
//    try std.testing.expectEqual(x.values[0], 1);
//    try std.testing.expectEqual(x.values[1], 4);
//    try std.testing.expectEqual(x.values[2], 7);
//    try std.testing.expectEqual(x.values[3], 2);
//    try std.testing.expectEqual(x.values[4], 5);
//    try std.testing.expectEqual(x.values[5], 8);
//    try std.testing.expectEqual(x.values[6], 3);
//    try std.testing.expectEqual(x.values[7], 6);
//    try std.testing.expectEqual(x.values[8], 9);
//
//    factory.deinit();
//}

test "contraction" {
    const std = @import("std");

    var factory = TensorFactory(i32).init(null);
    // need a more convincing test

    var x = try factory.allocTensor(3, Rowwise, .{ 3, 4, 3 });

    @memset(x.values, 1);

    var y = try factory.allocTensor(1, Rowwise, .{ 3 });

    try contraction("ijk->i", &x, &y);

    try std.testing.expectEqual(y.values[0], 12);
    try std.testing.expectEqual(y.values[1], 12);
    try std.testing.expectEqual(y.values[2], 12);

    var z = try factory.allocTensor(1, Rowwise, .{ 4 });

    try contraction("ijk->j", &x, &z);

    try std.testing.expectEqual(z.values[0], 9);
    try std.testing.expectEqual(z.values[1], 9);
    try std.testing.expectEqual(z.values[2], 9);
    try std.testing.expectEqual(z.values[3], 9);

    factory.deinit();
}


test "contraction 2" {
    const std = @import("std");

    var factory = TensorFactory(i32).init(null);
    var x = try factory.allocTensor(3, Rowwise, .{ 3, 4, 3 });
    var y = try factory.allocTensor(2, Rowwise, .{ 3, 4 });
    var z = try factory.allocTensor(2, Rowwise, .{ 4, 3 });

    fill(&x, 1, 1);

    try contraction("ijk->ij", &x, &y);

    try std.testing.expectEqual(y.values[0], 6);
    try std.testing.expectEqual(y.values[1], 15);
    try std.testing.expectEqual(y.values[2], 24);
    try std.testing.expectEqual(y.values[3], 33);
    try std.testing.expectEqual(y.values[4], 42);
    try std.testing.expectEqual(y.values[5], 51);
    try std.testing.expectEqual(y.values[6], 60);
    try std.testing.expectEqual(y.values[7], 69);
    try std.testing.expectEqual(y.values[8], 78);
    try std.testing.expectEqual(y.values[9], 87);
    try std.testing.expectEqual(y.values[10], 96);
    try std.testing.expectEqual(y.values[11], 105);

    try contraction("ijk->ji", &x, &z);

    try std.testing.expectEqual(z.values[0], 6);
    try std.testing.expectEqual(z.values[1], 42);
    try std.testing.expectEqual(z.values[2], 78);
    try std.testing.expectEqual(z.values[3], 15);
    try std.testing.expectEqual(z.values[4], 51);
    try std.testing.expectEqual(z.values[5], 87);
    try std.testing.expectEqual(z.values[6], 24);
    try std.testing.expectEqual(z.values[7], 60);
    try std.testing.expectEqual(z.values[8], 96);
    try std.testing.expectEqual(z.values[9], 33);
    try std.testing.expectEqual(z.values[10], 69);
    try std.testing.expectEqual(z.values[11], 105);

    try contraction("ijk->jk", &x, &z);

    try std.testing.expectEqual(z.values[0], 39);
    try std.testing.expectEqual(z.values[1], 42);
    try std.testing.expectEqual(z.values[2], 45);
    try std.testing.expectEqual(z.values[3], 48);
    try std.testing.expectEqual(z.values[4], 51);
    try std.testing.expectEqual(z.values[5], 54);
    try std.testing.expectEqual(z.values[6], 57);
    try std.testing.expectEqual(z.values[7], 60);
    try std.testing.expectEqual(z.values[8], 63);
    try std.testing.expectEqual(z.values[9], 66);
    try std.testing.expectEqual(z.values[10], 69);
    try std.testing.expectEqual(z.values[11], 72);

    factory.deinit();
}
// We're going to use insertion sort to figure out
// which stride is the smallest so we can create
// an efficient permutation-order array.

//fn optimalPermutation(comptime rank: usize, strides: []SizeAndStrideType) [rank]SizeAndStrideType {
//    var perm: [rank]SizeAndStrideType = undefined;
//    
//    var i: usize = 1;
//    var j: usize = undefined;
//    var k: usize = undefined;
//
//    while(i < rank) : (i += 1) {
//        perm[i] = i;
//    }
//    
//    while(i < rank) : (i += 1) {
//        k = strides[perm[i]];
//        j = i;
// 
//        while (j > 0 and strides[perm[j]] > k) {
//            perm[j] = j;
//            j -= 1;
//        }
//        perm[j] = i;
//    }
//    return perm;
//}

