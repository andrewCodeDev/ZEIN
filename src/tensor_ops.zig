// DESIGN PHILOSOPHY June 6th, 2023 //

// The goal for V1 is simple. Provide reliable (albeit naive) functionality
// that focuses on correctness first. Once that is established, V2 can use
// V1 as a reference for future versions, therefore creating a baseline
// for correctness. As such, the current goal is to provide a complete set
// of functionalities and replace them with more optimal solutions over time.

const std = @import("std");
const ReduceOp = std.builtin.ReduceOp;
const math = std.math;

const Util = @import("utility.zig");
const Tensor = @import("./tensor.zig").Tensor;
const TensorError = @import("./tensor.zig").TensorError;
const Rowwise = @import("./sizes_and_strides.zig").Rowwise;
const Colwise = @import("./sizes_and_strides.zig").Colwise;
const SizeType = @import("./sizes_and_strides.zig").SizeAndStride.ValueType;

pub const InnerProductPlan = @import("./expression_parsing.zig").InnerProductPlan;
pub const defaultPermuation = @import("./sizes_and_strides.zig").defaultPermutation;
pub const contractionParse = @import("./expression_parsing.zig").contractionParse;
pub const innerProductParse = @import("./expression_parsing.zig").innerProductParse;
pub const outerProductParse = @import("./expression_parsing.zig").outerProductParse;
pub const computeTensorIndex = @import("./tensor.zig").computeTensorIndex;

pub const OpsError = error{ UnequalSize, InvalidDimensions, InvalidSizes, SizeZeroTensor, IntegerOverflow };

inline fn reduceInit(comptime op: ReduceOp, comptime T: type) T {

    const info = @typeInfo(T);

    return switch (op) {
        .Add => 0, // implicit cast
        .Mul => 1, // implicit cast
        .Min => if (comptime info == .Int)
            math.maxInt(T) else math.floatMax(T),
        .Max => if (comptime info == .Int)
            math.minInt(T) else -math.floatMax(T),
        else => @compileError("reduceInit: unsupported op"),
    };
}

pub fn sum(x: anytype) @TypeOf(x.*).ValueType {
    std.debug.assert(x.valueSize() > 0);
    return simdReduce(ReduceOp.Mul, addGeneric, x, reduceInit(ReduceOp.Add, @TypeOf(x.*).ValueType));
}
pub fn product(x: anytype) @TypeOf(x.*).ValueType {
    std.debug.assert(x.valueSize() > 0);
    return simdReduce(ReduceOp.Mul, mulGeneric, x, reduceInit(ReduceOp.Mul, @TypeOf(x.*).ValueType));
}

pub fn min(x: anytype) @TypeOf(x.*).ValueType {
    std.debug.assert(x.valueSize() > 0);
    return simdReduce(ReduceOp.Min, minGeneric, x, reduceInit(ReduceOp.Min, @TypeOf(x.*).ValueType));
}
pub fn max(x: anytype) @TypeOf(x.*).ValueType {
    std.debug.assert(x.valueSize() > 0);
    return simdReduce(ReduceOp.Max, maxGeneric, x, reduceInit(ReduceOp.Max, @TypeOf(x.*).ValueType));
}

// TODO: Address the issue with checked vs unchecked absGeneric at call sight
pub fn absmax(x: anytype) @TypeOf(x.*).ValueType {
    return simdMapReduce(ReduceOp.Max, absGenericUnchecked, maxGeneric, x, reduceInit(ReduceOp.Max, @TypeOf(x.*).ValueType));
}

// TODO: Address the issue with checked vs unchecked absGeneric at call sight
pub fn absmin(x: anytype) @TypeOf(x.*).ValueType {
    return simdMapReduce(ReduceOp.Min, absGenericUnchecked, maxGeneric, x, reduceInit(ReduceOp.Min, @TypeOf(x.*).ValueType));
}

// TODO: does this belong here?
pub fn fill(
    x: anytype, 
    init: @TypeOf(x.*).ValueType,
    step: @TypeOf(x.*).ValueType
) void {
    var incr = init;
    for (x.values) |*value| {
        value.* = incr;
        incr += step;
    }
}

//////////////////////////////////////////////////////////////
///////// BINARY ARITHMETIC FUNCTIONS ////////////////////////

fn elementwiseCheck(x: anytype, y: anytype, z: anytype) void {
    if (comptime @TypeOf(x) != @TypeOf(y) or @TypeOf(y) != @TypeOf(z)) {
        @compileError("Mismatched tensor types for addition.");
    }
    std.debug.assert(x.isValid() and y.isValid() and z.isValid());
    std.debug.assert(x.valueSize() == y.valueSize() and y.valueSize() == z.valueSize());
}

pub fn add(x: anytype, y: anytype, z: anytype) void {
    elementwiseCheck(x, y, z);
    simdArithmetic(addGeneric, x, y, z);
}

// <>--------------------------------------------------------<>

pub fn sub(x: anytype, y: anytype, z: anytype) void {
    elementwiseCheck(x, y, z);
    simdArithmetic(subGeneric, x, y, z);
}

// <>--------------------------------------------------------<>

// TODO: should this be called mul? It's actually a hadamard
pub fn mul(x: anytype, y: anytype, z: anytype) void {
    elementwiseCheck(x, y, z);
    simdArithmetic(mulGeneric, x, y, z);
}

// <>--------------------------------------------------------<>

// TODO: scale seems like a bad name?
pub fn scale(x: anytype, y: @TypeOf(x), s: @TypeOf(x.*).ValueType) void {
    std.debug.assert(x.isValid() and y.isValid());
    std.debug.assert(x.valueSize() == y.valueSize());
    simdScalarBroadcast(mulGeneric, x, y, s);
}

// <>--------------------------------------------------------<>

pub fn bias(x: anytype, y: @TypeOf(x), b: @TypeOf(x.*).ValueType) void {
    std.debug.assert(x.isValid() and y.isValid());
    std.debug.assert(x.valueSize() == y.valueSize());
    simdScalarBroadcast(addGeneric, x, y, b);
}

// <>--------------------------------------------------------<>

inline fn quantizeGeneric(comptime int: type, x: anytype) int {
    return @intFromFloat(@round(x * comptime @as(@TypeOf(x), math.maxInt(int))));
}

pub fn quantize(x: anytype, y: anytype) @TypeOf(x.*).ValueType {
    const m = absmax(x);

    if (m > 1.0) {
        const s = 1.0 / m;
        var i: usize = 0;
        while (i < x.values.len) : (i += 1) {
            y.values[i] = quantizeGeneric(@TypeOf(y.*).ValueType, x.values[i] * s);
        }
    } else {
        var i: usize = 0;
        while (i < 100) : (i += 1) {
            y.values[i] = quantizeGeneric(@TypeOf(y.*).ValueType, x.values[i]);
        }
    }
    return m;
}

// <>--------------------------------------------------------<>

inline fn unquantizeGeneric(comptime float: type, x: anytype) float {
    return @as(float, @floatFromInt(x)) / comptime @as(float, @floatFromInt(math.maxInt(@TypeOf(x))));
}

pub fn unquantize(x: anytype, y: anytype, s: @TypeOf(y.*).ValueType) void {
    const FT = @TypeOf(y.*).ValueType;

    if (s > 1.0) {
        var i: usize = 0;
        while (i < x.values.len) : (i += 1) {
            y.values[i] = s * unquantizeGeneric(FT, x.values[i]);
        }
    } else {
        var i: usize = 0;
        while (i < 100) : (i += 1) {
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
    n: *IT, // scratch counter
) void {
    if (I == (R - 1)) {
        // we only need to make this once really...
        const x_ss: @Vector(R, IT) = x.*.sizes_and_strides.strides;

        var i: IT = 0;
        var n_i = n.*;
        while (i < x.*.getSize(I)) : ({
            i += 1;
            n_i += 1;
        }) {
            c[I] = i;
            const x_c: @Vector(R, IT) = c.*;
            const x_i = @reduce(ReduceOp.Add, x_c * x_ss);

            y[n_i] = x.*.values[x_i];
        }
        n.* += i;
    } else {
        var i: IT = 0;
        while (i < x.*.getSize(I)) : (i += 1) {
            c[I] = i;

            @call(.always_inline, recursivePermutate, .{ VT, IT, R, (I + 1), x, y, c, n });
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

pub fn contraction(comptime expression: []const u8, x: anytype, y: anytype) void {
    std.debug.assert(x.isValid() and y.isValid());

    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ip = comptime contractionParse(XT.Rank, YT.Rank, expression);

    if (comptime Util.debug) {    

        for (0..YT.Rank) |i| {
            std.debug.assert(x.getSize(ip.lhs[i]) == y.getSize(ip.rhs[i]));
        }
    }

    // TODO: @memset assumes host device memory...
    @memset(y.values, 0);

    unreachable; // TODO
}

//pub inline fn recursiveContraction(
//    comptime VT: type, // value type
//    comptime XR: usize, // tensor x rank
//    comptime YR: usize, // tensor y rank
//    comptime I: usize, // starting index
//    x: anytype, // source tensor
//    y: anytype, // destination memory
//    xc: []SizeType, // index container
//    yc: []SizeType, // index container
//) void {
    //if (XR <= YR) {
    //    @compileError("Contraction must go from a larger tensor to a smaller one.");
    //}

    //if (I < YR) {
    //    const x_perm_index = xp[I];
    //    const y_perm_index = yp[I];

    //    // this first branch loads up the x and y indices
    //    // and passes them to the next loop. In this case,
    //    // I is still in bounds of both x and y ranks.

    //    var i: IT = 0;
    //    while (i < x.getSize(x_perm_index)) : (i += 1) {
    //        xc[x_perm_index] = i;
    //        yc[y_perm_index] = i;

    //        @call(.always_inline, recursiveContraction, .{ VT, IT, XR, YR, xp, yp, (I + 1), x, y, xc, yc });
    //    }
    //} else if ((YR <= I) and (I < (XR - 1))) {

    //    // the second branch deals with values of I that are
    //    // out-of-bounds for y rank, but still in-bounds for
    //    // the x rank.

    //    const x_perm_index = xp[I];

    //    var i: IT = 0;
    //    while (i < x.getSize(x_perm_index)) : (i += 1) {
    //        xc[x_perm_index] = i;

    //        @call(.always_inline, recursiveContraction, .{ VT, IT, XR, YR, xp, yp, (I + 1), x, y, xc, yc });
    //    }
    //} else {

    //    // the third branch deals with summing up the contracted
    //    // indices and writing them to the related y index

    //    const x_ss: @Vector(XR, IT) = x.*.sizes_and_strides.strides;

    //    const x_perm_index = xp[I];

    //    var i: IT = 0;
    //    var t: VT = 0;
    //    while (i < x.getSize(x_perm_index)) : (i += 1) {
    //        xc[x_perm_index] = i;
    //        const x_c: @Vector(XR, IT) = xc.*;
    //        const x_i = @reduce(ReduceOp.Add, x_c * x_ss);
    //        t += x.values[x_i]; // accumulate summations
    //    }
    //    const y_ss: @Vector(YR, IT) = y.sizes_and_strides.strides;
    //    const y_c: @Vector(YR, IT) = yc.*;
    //    const y_i = @reduce(ReduceOp.Add, y_c * y_ss);
    //    y.*.values[y_i] += t;
    //}
//}

// <>--------------------------------------------------------<>

// TODO: Add explanation for this crazy thing...

pub fn innerProduct(comptime expression: []const u8, x: anytype, y: anytype, z: anytype) void {
    std.debug.assert(!x.isValid() or !y.isValid() or !z.isValid());        

    const plan = comptime innerProductParse(
        @TypeOf(x.*).Rank, @TypeOf(y.*).Rank, @TypeOf(z.*).Rank, expression
    );

    if (comptime Util.debug) {
        for (0..plan.total) |i| {
            if (plan.x_perm[i] != plan.pass and plan.y_perm[i] != plan.pass) {
                std.debug.assert(x.getSize(plan.x_perm[i]) != y.getSize(plan.y_perm[i]));
            }
        }
    }
    @memset(z.values, 0);

    innerProductImpl(plan, x, y, z);
}

// naive unrolling of inner product
// directly accumulate the indices 
// TODO:
//   turn this version into a last resort
//   and only dispatch if a better option
//   isn't available due to dimensions
fn innerProductImpl(
    comptime plan: anytype, // InnerProductPlan
    x: anytype, // lhs operand tensor
    y: anytype, // rhs operand tensor
    z: anytype, // output tensor
) void {
    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ZT = @TypeOf(z.*);

    // index containers for tensor computation
    var xc: [XT.Rank]SizeType = undefined;
    var yc: [YT.Rank]SizeType = undefined;
    var zc: [ZT.Rank]SizeType = undefined;

    inline for (0..plan.total) |I| {
        
        const size = if (plan.s_ctrl[I] == 0) 
            x.getSize(plan.x_perm[I]) else y.getSize(plan.y_perm[I]);

        for (0..size) |i| {

            if (comptime plan.x_perm[I] != plan.pass) { xc[plan.x_perm[I]] = i; }
            if (comptime plan.y_perm[I] != plan.pass) { yc[plan.y_perm[I]] = i; }
            if (comptime plan.z_perm[I] != plan.pass) { zc[plan.z_perm[I]] = i; }

            if (comptime I == (plan.total - 1)) {
                const x_n = computeTensorIndex(XT.Rank, XT.SizesType, x.getStrides(), &xc);
                const y_n = computeTensorIndex(YT.Rank, YT.SizesType, y.getStrides(), &yc);
                const z_n = computeTensorIndex(ZT.Rank, ZT.SizesType, z.getStrides(), &zc);
                z.values[z_n] += x.values[x_n] * y.values[y_n];
            }
        }
    }
}

// <>--------------------------------------------------------<>

// TODO: Add explanation for this crazy thing...

pub fn outerProduct(comptime expression: []const u8, x: anytype, y: anytype, z: anytype) void {
    if (!x.isValid() or !y.isValid() or !z.isValid()) {
        return TensorError.InvalidTensorLayout;
    }
    const XT = @TypeOf(x.*);
    const YT = @TypeOf(y.*);
    const ZT = @TypeOf(z.*);

    const plan = comptime outerProductParse(XT.Rank, YT.Rank, ZT.Rank, expression);

    if (Util.debug) {
        for (plan.x_perm, plan.y_perm, plan.z_perm) |xp, yp, zp| {
            if (xp != plan.pass and x.getSize(xp) != z.getSize(zp))
                return OpsError.InvalidDimensions;
            if (yp != plan.pass and y.getSize(yp) != z.getSize(zp))
                return OpsError.InvalidDimensions;
        }
    }

    @memset(z.values, 0);

    unreachable; //TODO
}

//pub inline fn recursiveOuterProduct(
//    comptime VT: type, // value type
//    comptime I: usize, // starting index
//    comptime plan: anytype, // InnerProductPlan
//    x: anytype, // lhs operand tensor
//    y: anytype, // rhs operand tensor
//    z: anytype, // output tensor
//) void {
//    const XT = @TypeOf(x.*);
//    const YT = @TypeOf(y.*);
//    const ZT = @TypeOf(z.*);
//
//    const size = @call(.always_inline, sizeSelector, .{ plan.x_perm[I], plan.y_perm[I], plan.s_ctrl[I], x, y });
//
//    if (I < (plan.total - 1)) {
//        var i: IT = 0;
//        while (i < size) : (i += 1) {
//            if (comptime plan.x_perm[I] != plan.pass) {
//                xc[plan.x_perm[I]] = i;
//            }
//            if (comptime plan.y_perm[I] != plan.pass) {
//                yc[plan.y_perm[I]] = i;
//            }
//            zc[plan.z_perm[I]] = i;
//            @call(.always_inline, recursiveInnerProduct, .{ VT, IT, (I + 1), plan, x, y, z, xc, yc, zc });
//        }
//    } else {
//        var i: IT = 0;
//        while (i < size) : (i += 1) {
//            if (comptime plan.x_perm[I] != plan.pass) {
//                xc[plan.x_perm[I]] = i;
//            }
//            if (comptime plan.y_perm[I] != plan.pass) {
//                yc[plan.y_perm[I]] = i;
//            }
//            zc[plan.z_perm[I]] = i;
//            const x_n = computeTensorIndex(XT.Rank, XT.SizesType, &x.sizes_and_strides.strides, xc.*);
//            const y_n = computeTensorIndex(YT.Rank, YT.SizesType, &y.sizes_and_strides.strides, yc.*);
//            const z_n = computeTensorIndex(ZT.Rank, ZT.SizesType, &z.sizes_and_strides.strides, zc.*);
//            z.values[z_n] += x.values[x_n] * y.values[y_n];
//        }
//    }
//}

// <>--------------------------------------------------------<>

fn simdReduce(
    comptime ReduceType: anytype,
    comptime BinaryFunc: anytype,
    x: anytype, 
    init: @TypeOf(x.*).ValueType
) @TypeOf(x.*).ValueType  {
    const T = @TypeOf(x.*).ValueType;
    var i: usize = 0;
    var rdx = init;

    // reduce in size N chunks...
    if (comptime std.simd.suggestVectorLength(T)) |N| {
        while ((i + N) < x.valueSize()) : (i += N) {
            const vec: @Vector(N, T) = x.values[i..i + N][0..N].*; // needs compile time length
            rdx = @call(.always_inline, BinaryFunc, .{ rdx, @reduce(ReduceType, vec) });
        }
    }

    // reduce remainder...
    while (i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, BinaryFunc, .{ rdx, x.values[i] });
    }
    return rdx;
}

// <>--------------------------------------------------------<>

fn simdArithmetic(
    comptime BinaryFunc: anytype,
    x: anytype,
    y: anytype,
    z: anytype,
) void {

    const T = @TypeOf(x.*).ValueType;
    var i: usize = 0;
    
    if (comptime std.simd.suggestVectorLength(T)) |N| {
        var j: usize = N;    
        while(j <= x.len) : ({i += N; j += N; }) {
            const v: @Vector(N, T) = x.values[i..j][0..N].*;
            const u: @Vector(N, T) = y.values[i..j][0..N].*;
            z[i..j][0..N].* = @call(.always_inline, BinaryFunc, .{v, u});
        }
    }

    while (i < x.len) : (i += 1) {
        z.values[i] = @call(.always_inline, BinaryFunc, .{ x.values[i], y.values[i] });
    }
}

// <>--------------------------------------------------------<>

// TODO: limited in terms of what "map" can be
fn simdMapReduce(
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
    if (comptime std.simd.suggestVectorLength(T)) |N| {
        while ((i + N) < x.valueSize()) : (i += N) {
            var vec: @Vector(N, T) = x.values[i..i + N][0..N].*;
            vec = @call(.always_inline, UnaryFunc, .{vec});
            rdx = @call(.always_inline, BinaryFunc, .{ rdx, @reduce(ReduceType, vec) });
        }
    }

    // reduce remainder...
    while (i < x.valueSize()) : (i += 1) {
        rdx = @call(.always_inline, BinaryFunc, .{ rdx, x.values[i] });
    }
    return rdx;
}

// <>--------------------------------------------------------<>

fn simdScalarBroadcast(
    comptime BinaryFunc: anytype,
    x: anytype,
    y: anytype,
    s: @TypeOf(x.*).ValueType
) void {
    const T = @TypeOf(x.*).ValueType;

    var i: usize = 0;

    // broadcast in size N chunks...
    if (comptime std.simd.suggestVectorLength(T)) |N| {
        const u: @Vector(N, T) = @splat(s);

        var j: usize = N;
        while (j <= x.values.len) : ({ i += N; j += N; }) {
            const v: @Vector(N, T) = x.values[i..j][0..N].*;
            y.values[i..j][0..N].* = @call(.always_inline, BinaryFunc, .{ v, u });
        }
    }

    // broadcast remainder...
    while (i < x.values.len) : (i += 1) {
        y.values[i] = @call(.always_inline, BinaryFunc, .{ x.values[i], s });
    }
}

// <>--------------------------------------------------------<>

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

// <>--------------------------------------------------------<>

pub inline fn absGenericUnchecked(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    switch (comptime @typeInfo(T)) {
        .Float => {
            return @abs(x);
        },
        .Int => |info| {
            if (comptime info.signedness == true) {
                const mask = x >> (comptime @bitSizeOf(T) - 1);
                return (x + mask) ^ mask;
            } else {
                return x;
            }
        },
        .Vector => |info| {
            switch (comptime @typeInfo(info.child)) {
                .Float => {
                    return @abs(x);
                },
                else => {
                    @compileError("Absolute value for integer vectors unimplemented.");
                },
            }
        },
        else => @compileError("Invalid type passed to absGeneric function: " ++ @typeName(T)),
    }
}

pub inline fn absGeneric(x: anytype) !@TypeOf(x) {
    const T = @TypeOf(x);
    return switch (comptime @typeInfo(T)) {
        .Int => |info| {
            if (info.signedness) {
                if (x == math.minInt(T)) return OpsError.IntegerOverflow;
            }
            return @call(.always_inline, absGenericUnchecked, .{x});
        },
        else => @call(.always_inline, absGenericUnchecked, .{x}),
    };
}
