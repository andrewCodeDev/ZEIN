
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
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;
const SizeAndStrideType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;
var GPA = @import("TensorFactory.zig");
const ReduceOp = @import("std").builtin.ReduceOp;
const math = @import("std").math;

pub const InnerProductPlan = @import("ExpressionParsing.zig").InnerProductPlan;
pub const defaultPermuation = @import("SizesAndStrides.zig").defaultPermutation;
pub const contractionParse = @import("ExpressionParsing.zig").contractionParse;
pub const innerProductParse = @import("ExpressionParsing.zig").innerProductParse;
pub const computeTensorIndex = @import("Tensor.zig").computeTensorIndex;


// The OpsPolicy controls the behavior of the math
// class. This includes behavior such as whether or
// not the class can allocate new tensors if output
// tensors are are not provided

pub const OpsError = error {
    UnequalSize,
    InvalidDimensions,
    InvalidSizes,
    SizeZeroTensor
};

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

pub inline fn recursivePermutate(
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

pub fn contraction(comptime expression: [] const u8, x: anytype, y: anytype) !void {

    if(!x.isValid() or !y.isValid()) {
        return TensorError.InvalidTensorLayout;
    }

    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ip = comptime contractionParse(XT.Rank, YT.Rank, expression);

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
        XT.ValueType, XT.SizesType, XT.Rank, YT.Rank, ip.lhs, ip.rhs, 0, x, y, &xc, &yc
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
        XT.ValueType, XT.SizesType, XT.Rank, YT.Rank, ip.lhs, ip.rhs, 0, x, y, &xc, &yc
    });
}

pub inline fn recursiveContraction(
    comptime VT: type, // value type
    comptime IT: type, // int type
    comptime XR: usize, // tensor x rank
    comptime YR: usize, // tensor y rank
    comptime xp: [XR]IT, // x permutation
    comptime yp: [YR]IT, // y permutation
    comptime I: usize, // starting index
    x: anytype, // source tensor
    y: anytype, // destination memory
    xc: *[XR]IT, // index container
    yc: *[YR]IT, // index container
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
                 VT, IT, XR, YR, xp, yp, (I + 1), x, y, xc, yc
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
                 VT, IT, XR, YR, xp, yp, (I + 1), x, y, xc, yc
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

pub fn innerProduct(
    comptime expression: [] const u8,
    x: anytype,
    y: anytype,
    z: anytype) !void {

    if(!x.isValid() or !y.isValid() or !z.isValid()) {
        return TensorError.InvalidTensorLayout;
    }

    // TODO: Add dimension checks for compatible indexing

    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ZT = @TypeOf(z.*);

    const plan = comptime innerProductParse(
        XT.Rank, YT.Rank, ZT.Rank, expression
    );

    var x_i: [XT.Rank]XT.SizesType = undefined;
    var y_i: [YT.Rank]YT.SizesType = undefined;
    var z_i: [ZT.Rank]ZT.SizesType = undefined;
    
    @memset(z.values, 0);
    
    @call(.always_inline, recursiveInnerProduct, .{
        XT.ValueType, XT.SizesType, 0, plan, x, y, z, &x_i, &y_i, &z_i 
    });
}

pub inline fn sizeSelector(
    comptime x_index: usize,
    comptime y_index: usize,
    comptime select: usize,
    x: anytype,
    y: anytype
) usize {
    if(select == 0) {
        return x.getSize(x_index);
    } else {
        return y.getSize(y_index);
    }
}

pub inline fn recursiveInnerProduct(
    comptime VT: type, // value type
    comptime IT: type, // int type
    comptime I: usize, // starting index
    comptime plan: anytype, // InnerProductPlan
    x: anytype, // lhs operand tensor
    y: anytype, // rhs operand tensor
    z: anytype, // output tensor
    xc: *[@TypeOf(x.*).Rank]IT, // index container
    yc: *[@TypeOf(y.*).Rank]IT, // index container
    zc: *[@TypeOf(z.*).Rank]IT, // index container
) void {

    const XT = @TypeOf(x.*);
    const YT = @TypeOf(x.*);
    const ZT = @TypeOf(x.*);

    const size = @call(.always_inline, sizeSelector,
        .{ plan.x_perm[I], plan.y_perm[I], plan.s_ctrl[I], x, y }
    );

    if(I < (plan.total - 1)) {
        var i: IT = 0;
        while(i < size) : (i += 1) {
            if(plan.x_perm[I] != plan.pass) { xc[plan.x_perm[I]] = i; }
            if(plan.y_perm[I] != plan.pass) { yc[plan.y_perm[I]] = i; }
            if(plan.z_perm[I] != plan.pass) { zc[plan.z_perm[I]] = i; }
            @call(.always_inline, recursiveInnerProduct, .{
                 VT, IT, (I + 1), plan, x, y, z, xc, yc, zc
            });
        }
    }

    else {
        var i: IT = 0;
        while(i < size) : (i += 1) {
            if(plan.x_perm[I] != plan.pass) { xc[plan.x_perm[I]] = i; }
            if(plan.y_perm[I] != plan.pass) { yc[plan.y_perm[I]] = i; }
            if(plan.z_perm[I] != plan.pass) { zc[plan.z_perm[I]] = i; }
            const x_n = computeTensorIndex(XT.Rank, XT.SizesType, &x.sizes_and_strides.strides, xc.*);
            const y_n = computeTensorIndex(YT.Rank, YT.SizesType, &y.sizes_and_strides.strides, yc.*);
            const z_n = computeTensorIndex(ZT.Rank, ZT.SizesType, &z.sizes_and_strides.strides, zc.*);
            z.values[z_n] += x.values[x_n] * y.values[y_n];
        }
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

pub fn add(x: anytype, y: anytype, z: anytype) !void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    if(!x.isValid() or !y.isValid() or !z.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    if(x.valueSize() != y.valueSize() or y.valueSize() != z.valueSize()) {
        return OpsError.UnequalSize;
    }
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        z.values[i] = x.values[i] + y.values[i1];
    }
}

pub fn addUnchecked(x: anytype, y: anytype, z: anytype) void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        z.values[i] = x.values[i] + y.values[i1];
    }
}

pub fn multiply(x: anytype, y: anytype, z: anytype) !void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    if(!x.isValid() or !y.isValid() or !z.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    if(x.valueSize() != y.valueSize() or y.valueSize() != z.valueSize()) {
        return OpsError.UnequalSize;
    }
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        z.values[i] = x.values[i] * y.values[i1];
    }
}

pub fn multiplyUnchecked(x: anytype, y: anytype, z: anytype) void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        z.values[i] = x.values[i] * y.values[i1];
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

