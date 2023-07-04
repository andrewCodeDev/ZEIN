
// DESIGN PHILOSOPHY June 6th, 2023 //

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


pub const OpsError = error {
    UnequalSize,
    InvalidDimensions,
    InvalidSizes,
    SizeZeroTensor,
    IntegerOverflow
};

inline fn uint(comptime size: usize) type {
    return switch(size) {
        64 => return u64,
        32 => return u32,
        16 => return u16,
         8 => return u8,
        else => @compileError("Invalid size passed to uint function.")
    };
}

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
        return reduceDispatch(ReduceOp.Add, addGeneric, x, initValue(ReduceOp.Add, @TypeOf(x.*).ValueType));
    } else {
         return OpsError.SizeZeroTensor;
    }
}
pub fn product(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
        return reduceDispatch(ReduceOp.Mul, mulGeneric, x, initValue(ReduceOp.Mul, @TypeOf(x.*).ValueType));
    } else {
        return OpsError.SizeZeroTensor;
    }
}
pub fn min(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
        return reduceDispatch(ReduceOp.Min, minGeneric, x, initValue(ReduceOp.Min, @TypeOf(x.*).ValueType));
    } else {
        return OpsError.SizeZeroTensor;
    }
}
pub fn max(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
       return reduceDispatch(ReduceOp.Max, maxGeneric, x, initValue(ReduceOp.Max, @TypeOf(x.*).ValueType));
    } else {
        return OpsError.SizeZeroTensor;
    }
}

// TODO: Address the issue with checked vs unchecked absGeneric at call sight
pub fn absmax(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
       return mapReduceDispatch(
            ReduceOp.Max, absGenericUnchecked, maxGeneric, x, initValue(ReduceOp.Max, @TypeOf(x.*).ValueType)
       );
    } else {
        return OpsError.SizeZeroTensor;
    }
}

// TODO: Address the issue with checked vs unchecked absGeneric at call sight
pub fn absmin(x: anytype) !@TypeOf(x.*).ValueType {
    if(x.valueSize() > 0) {
       return mapReduceDispatch(
            ReduceOp.Min, absGenericUnchecked, minGeneric, x, initValue(ReduceOp.Min, @TypeOf(x.*).ValueType)
       );
    } else {
        return OpsError.SizeZeroTensor;
    }
}

// TODO: Address the issue with checked vs unchecked absGeneric at call sight
pub fn absmaxUnchecked(x: anytype) @TypeOf(x.*).ValueType {
    return mapReduceDispatch(
         ReduceOp.Max, absGenericUnchecked, maxGeneric, x, initValue(ReduceOp.Max, @TypeOf(x.*).ValueType)
    );
}

// TODO: Address the issue with checked vs unchecked absGeneric at call sight
pub fn absminUnchecked(x: anytype) @TypeOf(x.*).ValueType {
    return mapReduceDispatch(
         ReduceOp.Min, absGenericUnchecked, minGeneric, x, initValue(ReduceOp.Min, @TypeOf(x.*).ValueType)
    );
}
// To complete the set, for those who like to live dangerously... the unchecked versions.

pub fn sumUnchecked(x: anytype) @TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Add, addGeneric, x, initValue(ReduceOp.Add, @TypeOf(x.*).ValueType));
}
pub fn productUnchecked(x: anytype) @TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Mul, mulGeneric, x, initValue(ReduceOp.Mul, @TypeOf(x.*).ValueType));
}
pub fn minUnchecked(x: anytype) @TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Min, minGeneric, x, initValue(ReduceOp.Min, @TypeOf(x.*).ValueType));
}
pub fn maxUnchecked(x: anytype) @TypeOf(x.*).ValueType {
    return reduceDispatch(ReduceOp.Max, maxGeneric, x, initValue(ReduceOp.Max, @TypeOf(x.*).ValueType));
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

//////////////////////////////////////////////////////////////
///////// BINARY ARITHMETIC FUNCTIONS ////////////////////////

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
    arithmeticDispatch(addGeneric, x, y, z);
}

pub fn addUnchecked(x: anytype, y: anytype, z: anytype) void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    arithmeticDispatch(addGeneric, x, y, z);
}

pub fn sub(x: anytype, y: anytype, z: anytype) !void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    if(!x.isValid() or !y.isValid() or !z.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    if(x.valueSize() != y.valueSize() or y.valueSize() != z.valueSize()) {
        return OpsError.UnequalSize;
    }
    arithmeticDispatch(subGeneric, x, y, z);
}

pub fn subUnchecked(x: anytype, y: anytype, z: anytype) void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    arithmeticDispatch(subGeneric, x, y, z);
}

pub fn mul(x: anytype, y: anytype, z: anytype) !void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    if(!x.isValid() or !y.isValid() or !z.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    if(x.valueSize() != y.valueSize() or y.valueSize() != z.valueSize()) {
        return OpsError.UnequalSize;
    }
    arithmeticDispatch(mulGeneric, x, y, z);
}

pub fn mulUnchecked(x: anytype, y: anytype, z: anytype) void {
    if(@TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    arithmeticDispatch(mulGeneric, x, y, z);
}

pub fn scale(x: anytype, y: @TypeOf(x), s: @TypeOf(x.*).ValueType) !void {
    if(!x.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        y.values[i] = @call(.always_inline, mulGeneric, .{ x.values[i], s });
    }
}

pub fn scaleUnchecked(x: anytype, y: @TypeOf(x), s: @TypeOf(x.*).ValueType) void {
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        y.values[i] = @call(.always_inline, mulGeneric, .{ x.values[i], s });
    }
}

pub fn bias(x: anytype, y: @TypeOf(x), s: @TypeOf(x.*).ValueType) !void {
    if(!x.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        y.values[i] = @call(.always_inline, addGeneric, .{ x.values[i], s });
    }
}

pub fn biasUnchecked(x: anytype, y: @TypeOf(x), s: @TypeOf(x.*).ValueType) void {
    var i: usize = 0;
    while(i < x.values.len) : (i += 1) {
        y.values[i] = @call(.always_inline, addGeneric, .{ x.values[i], s });
    }
}
inline fn quantizeGeneric(comptime int: type, x: anytype) int {
    return @intFromFloat(@round(x * comptime @as(@TypeOf(x), math.maxInt(int))));
}

pub fn quantize(x: anytype, y: anytype) @TypeOf(x.*).ValueType {
    const m = absmaxUnchecked(x);

    if(m > 1.0) {
        const s = 1.0 / m;
        var i: usize = 0;
        while(i < x.values.len) : (i += 1) {
            y.values[i] = quantizeGeneric(@TypeOf(y.*).ValueType, x.values[i] * s);
        }
    }
    else {
        var i: usize = 0;
        while(i < 100) : (i += 1) {
            y.values[i] = quantizeGeneric(@TypeOf(y.*).ValueType, x.values[i]);
        }
    }
    return m;
}


inline fn unquantizeGeneric(comptime float: type, x: anytype) float {
    return @as(float, @floatFromInt(x)) / comptime @as(float, @floatFromInt(math.maxInt(@TypeOf(x))));
}

pub fn unquantize(x: anytype, y: anytype, s: @TypeOf(y.*).ValueType) void {
    const FT = @TypeOf(y.*).ValueType;
    
    if(s > 1.0) {
        var i: usize = 0;
        while(i < x.values.len) : (i += 1) {
            y.values[i] = s * unquantizeGeneric(FT, x.values[i]);
        }
    }
    else {
        var i: usize = 0;
        while(i < 100) : (i += 1) {
            y.values[i] = unquantizeGeneric(FT, x.values[i]);
        }
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
    comptime GenericFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
    ) @TypeOf(x.*).ValueType {
    var i: usize = 0;
    var rdx = init;
    while(i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, GenericFunc, .{ rdx, x.values[i] });
    }
    return rdx;
}

fn vectorizedReduce(
    comptime N: usize, 
    comptime ReduceType: anytype, 
    comptime GenericFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
    ) @TypeOf(x.*).ValueType {

    const T = @TypeOf(x.*).ValueType;
    var i: usize = 0;
    var rdx = init;
    
    // reduce in size N chunks...
    while((i + N) < x.valueSize()) : (i += N) {
        const slice = x.values[i..N + i];
        const vec: @Vector(N, T) = slice[0..N].*; // needs compile time length
        rdx = @call(.always_inline, GenericFunc, .{ rdx, @reduce(ReduceType, vec) });
    }
    // reduce remainder...
    while(i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, GenericFunc, .{ rdx, x.values[i] });
    }
    return rdx;
}

fn reduceDispatch(    
    comptime ReduceType: anytype, 
    comptime GenericFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
) @TypeOf(x.*).ValueType {

    const size = x.valueSize();

    if(size < 128) {
        return loopReduce(GenericFunc, x, init);
    }
    else if(size < 256) {
        return vectorizedReduce(128, ReduceType, GenericFunc, x, init);
    }
    else if(size < 512) {
        return vectorizedReduce(256, ReduceType, GenericFunc, x, init);
    }
    else {
        return vectorizedReduce(512, ReduceType, GenericFunc, x, init);
    }
}

fn loopArithmetic(
    comptime BinaryFunc: anytype, 
    x: anytype,
    y: anytype,
    z: anytype,
    ) void {
    var i: usize = 0;
    while(i < x.valueSize()) : (i += 1) {
        z.values[i] = @call(.always_inline, BinaryFunc, .{ x.values[i], y.values[i] });
    }
}

fn vectorizedArithmetic(
    comptime N: usize, 
    comptime BinaryFunc: anytype, 
    x: anytype,
    y: anytype,
    z: anytype,
    ) void {
    const T = @TypeOf(x.*).ValueType;
    var i: usize = 0;
    var j: usize = N;
    var buffer: [N]T = undefined;
    while(j <= x.values.len) : ({i += 512; j += 512; }) {
        const v: @Vector(N, T) = x.values[0..N].*;
        const u: @Vector(N, T) = y.values[0..N].*;
        buffer = @call(.always_inline, BinaryFunc, .{v, u});
        @memcpy(z.values[i..j], &buffer);
    }
    while(i < x.values.len) : (i += 1) {
        z.values[i] = @call(.always_inline, BinaryFunc, .{ x.values[i], y.values[i] });
    }
}

fn arithmeticDispatch(    
    comptime BinaryFunc: anytype, 
    x: anytype,
    y: anytype,
    z: anytype,
) void {

    const size = x.valueSize();

    if(size < 128) {
        return loopArithmetic(BinaryFunc, x, y, z);
    }
    else if(size < 256) {
        return vectorizedArithmetic(128, BinaryFunc, x, y, z);
    }
    else if(size < 512) {
        return vectorizedArithmetic(256, BinaryFunc, x, y, z);
    }
    else {
        return vectorizedArithmetic(512, BinaryFunc, x, y, z);
    }
}

fn loopMapReduce(
    comptime UnaryFunc: anytype, 
    comptime BinaryFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
    ) @TypeOf(x.*).ValueType {
    var i: usize = 0;
    var rdx = init;
    while(i < x.valueSize()) : (i += 1) {
        const u = @call(.always_inline, UnaryFunc, .{ x.values[i] });
        rdx = @call(.always_inline, BinaryFunc, .{ rdx, u }); 
    }
    return rdx;
}

fn vectorizedMapReduce(
    comptime N: usize, 
    comptime ReduceType: anytype, 
    comptime UnaryFunc: anytype, 
    comptime BinaryFunc: anytype, 
    x: anytype,
    init: @TypeOf(x.*).ValueType
    ) @TypeOf(x.*).ValueType {

    const T = @TypeOf(x.*).ValueType;
    var i: usize = 0;
    var rdx = init;
    
    // reduce in size N chunks...
    while((i + N) < x.valueSize()) : (i += N) {
        const slice = x.values[i..N + i];
        var vec: @Vector(N, T) =  slice[0..N].*; // needs compile time length
        vec = @call(.always_inline, UnaryFunc, .{ vec });
        rdx = @call(.always_inline, BinaryFunc, .{ rdx, @reduce(ReduceType, vec) });
    }
    // reduce remainder...
    while(i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, BinaryFunc, .{ rdx, x.values[i] });
    }
    return rdx;
}

fn mapReduceDispatch(    
    comptime ReduceType: anytype, 
    comptime UnaryFunc: anytype,
    comptime BinaryFunc: anytype,
    x: anytype,
    init: @TypeOf(x.*).ValueType
) @TypeOf(x.*).ValueType {

    const size = x.valueSize();

    if(size < 128) {
        return loopMapReduce(UnaryFunc, BinaryFunc, x, init);
    }
    else if(size < 256) {
        return vectorizedMapReduce(128, ReduceType, UnaryFunc, BinaryFunc, x, init);
    }
    else if(size < 512) {
        return vectorizedMapReduce(256, ReduceType, UnaryFunc, BinaryFunc, x, init);
    }
    else {
        return vectorizedMapReduce(512, ReduceType, UnaryFunc, BinaryFunc, x, init);
    }
}

inline fn addGeneric(x: anytype, y: anytype) @TypeOf(x) {
    return x + y;
}
inline fn mulGeneric(x: anytype, y: anytype) @TypeOf(x) {
    return x * y;
}
inline fn subGeneric(x: anytype, y: anytype) @TypeOf(x) {
    return x - y;
}
inline fn divGeneric(x: anytype, y: anytype) @TypeOf(x) {
    return x / y;
}
inline fn maxGeneric(x: anytype, y: anytype) @TypeOf(x) {
    return @max(x, y);
}
inline fn minGeneric(x: anytype, y: anytype) @TypeOf(x) {
    return @min(x, y);
}

pub inline fn absGenericUnchecked(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    return switch (comptime @typeInfo(T)) {
        .Float => {
            return @fabs(x);
        },
        .Int => |info| {
            if(comptime info.signedness == true) {
                const mask = x >> (comptime @bitSizeOf(@TypeOf(x)) - 1);
                return (x + mask) ^ mask;
            } else {
                return x;  
            }
        },
        .Vector => |info| {
            return switch(comptime @typeInfo(info.child)) {
                .Float => {
                    return @fabs(x);
                },
                else => {
                    @compileError("Absolute value for integer vectors unimplemented.");
                }
            };
        },
        else => @compileError("Invalid type passed to absGeneric function: " ++ @typeName(T))
    };
}

pub inline fn absGeneric(x: anytype) !@TypeOf(x) {    
    const T = @TypeOf(x);
    return switch (comptime @typeInfo(T)) {
        .Int => |info| {
            if(info.signedness) {
                if(x == math.minInt(T)) return OpsError.IntegerOverflow;
            }
            return @call(.always_inline, absGenericUnchecked, .{ x });
        },
        else => @call(.always_inline, absGenericUnchecked, .{ x })
    };
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

