// Here we find the heart of Zein - Tensors. Before proceeding, please read the following:

///////////////////////////////////
// DESIGN PHILOSOPHY (June 3, 2023)

// MEMORY, OWNDERSHIP, AND REFERENCING //

// There is no plan to make a distinction between a tensor and a "view" of a tensor.
// Tensors here are, by design, a way to view data. As such, a different "tensored" view
// of the same data is just another tensor that shares underlying memory.

// !!! THIS STRONGLY IMPLIES THAT TENSORS DO NOT *OWN* DATA, THEY VIEW IT !!!

// If anything can be said to "own" memory, it is the allocator. Allocators are going
// to play an important role in this library (as they do in Zig more generally).

// To create a tensor that has initialized memory is the job of a factory.
// The design of such a tensor factory, as it were, will be handled in a source
// file dedicated to that exact job. It is very important that we do not cross
// responsibilities in this system.

// TENSORS AS THEY RELATE TO ARRAYS //

// Because of the design descisions outlined above, users should be able to easily
// make a tensor with their desired dimensions to wrap existing arrays and manipulate
// them as if they were tensors themselves. This means that a tensor can act like
// an adapter to already existing memory.

// Because of this, there is not a current plan to enforce that tensors must be of
// one type or another. It is my hope to provide a generic tensor based interface
// that can be used on a variety of objects at the user's caution.

// At some point, it may be important to then provide a generic functional interface
// to provide for further use cases such as generically holding objects that users
// create themselves. While this is an interesting goal, the scope of V1 is currently
// focused on integer and floating point numbers. User provided types will have to
// be reviewed as time goes forward.

const std = @import("std");

const Util = @import("utility.zig");

// STD import files...
const ReduceOp = @import("std").builtin.ReduceOp;

const arrayProduct = @import("./utility.zig").arrayProduct;

// Zein import files...
pub const SizeAndStride = @import("./sizes_and_strides.zig").SizeAndStride;
pub const SizesAndStrides = @import("./sizes_and_strides.zig").SizesAndStrides;
pub const OrderType = @import("./sizes_and_strides.zig").OrderType;
pub const Rowwise = @import("./sizes_and_strides.zig").Rowwise;
pub const Colwise = @import("./sizes_and_strides.zig").Colwise;
const Permutate = @import("./permutate.zig");

// Tensor Utilities...
pub const TensorError = error{ InvalidTensorLayout, InvalidPermutation, AllocSizeMismatch, CapacityMismatch, RankMismatch };

pub inline fn computeTensorIndex(
    comptime rank: usize, 
    comptime size_type: type, 
    strides: []const size_type, 
    indices: []const size_type
) size_type {
    return switch(rank) {
        1 => indices[0], // direct index... just an array
        2 => indices[0] * strides[0] + indices[1] * strides[1],
        else => blk: { // inner product between indices and strides
            const s: @Vector(rank, size_type) = strides[0..rank];
            const i: @Vector(rank, size_type) = indices[0..rank];
            break :blk @reduce(ReduceOp.Add, s * i);
        },
    };
}

///////////////////////////
// Tensor Implementation //

pub fn Tensor(comptime value_type: type, comptime rank: usize, comptime order: OrderType) type {
    if (63 < rank) {
        @compileError("Tensors of rank 64 or greater are not supported.");
    }

    if (0 == rank) {
        @compileError("Tensors of rank zero are not supported.");
    }

    return struct {
        pub const Rank = rank;

        pub const Order = order;

        pub const SizesType = SizeAndStride.ValueType;

        pub const ValueType = value_type;

        pub const ValueSlice = []ValueType;

        pub const SizesAndStridesType = SizesAndStrides(Rank, Order);

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        values: ValueSlice,

        sizes_and_strides: SizesAndStridesType,

        pub fn init(values: ?ValueSlice, sizes: ?[Rank]SizesType) Self {
            return Self{
                .values = if (values) |vs| (vs) else &[_]ValueType{},
                .sizes_and_strides = SizesAndStridesType.init(sizes),
            };
        }

        pub fn sliceSizes(self: ConstSelfPtr, i: usize, j: usize) []const SizesType {
            return &self.sizes_and_strides.sizes[i..j];
        }
        pub fn sliceStrides(self: ConstSelfPtr, i: usize, j: usize) []const SizesType {
            return &self.sizes_and_strides.strides[i..j];
        }
        pub fn slicePermutation(self: ConstSelfPtr, i: usize, j: usize) []const SizesType {
            return &self.sizes_and_strides.permutation[i..j];
        }

        pub fn getSizes(self: ConstSelfPtr) []const SizesType {
            return &self.sizes_and_strides.sizes;
        }
        pub fn getStrides(self: ConstSelfPtr) []const SizesType {
            return &self.sizes_and_strides.strides;
        }
        pub fn getPermutation(self: ConstSelfPtr) []const SizesType {
            return &self.sizes_and_strides.permutation;
        }

        pub fn valueCapacity(self: ConstSelfPtr) usize {
            return arrayProduct(Rank, SizesType, &self.sizes_and_strides.sizes);
        }

        pub fn valueSize(self: ConstSelfPtr) usize {
            return self.values.len;
        }

        pub fn isValid(self: ConstSelfPtr) bool {
            return self.valueSize() != 0 and self.valueSize() == self.valueCapacity();
        }

        pub fn swap(self: SelfPtr, other: SelfPtr) void {
            self.swapValues(other);
            self.swapSizesAndStrides(other);
        }

        pub fn swapValues(self: SelfPtr, other: SelfPtr) void {
            // to assure that sizes and strides are not
            // invalidated, we check size and capacity
            std.debug.assert(self.valueSize() == other.valueSize());
            std.debug.assert(self.isValid() and other.isValid());

            const values = self.values;
            self.values = other.values;
            other.values = values;
        }

        pub fn swapSizesAndStrides(self: SelfPtr, other: SelfPtr) void {
            // we only want to compute these once...

            if (comptime Util.debug) {
                const capacity_a = self.valueCapacity();
                const capacity_b = other.valueCapacity();
                // tensors can have different SizesAndStrides
                // and still share the total value capcity
                std.debug.assert(capacity_a == capacity_b);
                // check that both tensors are at capacity without additional computation
                std.debug.assert(
                    self.valueSize() == capacity_a and other.valueSize() == capacity_b
                );
            }

            // there is probably a faster way to do this
            const tmp = self.sizes_and_strides;
            self.sizes_and_strides = other.sizes_and_strides;
            other.sizes_and_strides = tmp;
        }

        pub fn permutate(self: SelfPtr, comptime expression: []const u8) Self {
            // create a permutated tensor that shares the same underlying memory
            std.debug.assert(self.isValid());

            var tmp = self.*; // share values
            Permutate.permutate(Rank, Order, expression, &tmp.sizes_and_strides);
            return tmp;
        }

        pub fn getValue(self: ConstSelfPtr, indices: [rank]SizesType) ValueType {
            const n = computeTensorIndex(Rank, SizesType, self.getStrides(), &indices);
            return self.values[n];
        }

        pub fn setValue(self: ConstSelfPtr, value: ValueType, indices: [rank]SizesType) void {
            const n = computeTensorIndex(Rank, SizesType, self.getStrides(), &indices);
            self.values[n] = value;
        }

        pub inline fn getSize(self: ConstSelfPtr, i: usize) SizesType {
            return self.sizes_and_strides.sizes[i];
        }

        pub inline fn getStride(self: ConstSelfPtr, i: usize) SizesType {
            return self.sizes_and_strides.strides[i];
        }
    };
}

test "Initialization" {
    const expect = std.testing.expect;

    var x = Tensor(u32, 3, Rowwise).init(null, .{ 10, 20, 30 });

    const total: usize = 10 * 20 * 30;

    try expect(total == x.valueCapacity());
}

test "Tensor Swapping" {
    const expect = std.testing.expect;

    const x_values = try std.heap.page_allocator.alloc(i8, 100);
    defer std.heap.page_allocator.free(x_values);

    const y_values = try std.heap.page_allocator.alloc(i8, 100);
    defer std.heap.page_allocator.free(y_values);

    var x = Tensor(i8, 2, Rowwise).init(x_values, .{ 10, 10 });
    var y = Tensor(i8, 2, Rowwise).init(y_values, .{ 10, 10 });

    x.swap(&y);

    try expect(x.values.ptr == y_values.ptr);
    try expect(y.values.ptr == x_values.ptr);

    const total: usize = 10 * 10;

    try expect(total == x.valueCapacity());
    try expect(total == y.valueCapacity());
}

test "Tensor Transpose" {
    const expect = std.testing.expect;

    var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    var x = Tensor(i32, 2, Rowwise).init(&data, .{ 3, 3 });

    try expect(x.isValid());

    try expect(x.getValue(.{ 0, 0 }) == 1);
    try expect(x.getValue(.{ 0, 1 }) == 2);
    try expect(x.getValue(.{ 0, 2 }) == 3);
    try expect(x.getValue(.{ 1, 0 }) == 4);
    try expect(x.getValue(.{ 1, 1 }) == 5);
    try expect(x.getValue(.{ 1, 2 }) == 6);
    try expect(x.getValue(.{ 2, 0 }) == 7);
    try expect(x.getValue(.{ 2, 1 }) == 8);
    try expect(x.getValue(.{ 2, 2 }) == 9);

    var y = x.permutate("ij->ji");

    try expect(y.getValue(.{ 0, 0 }) == 1);
    try expect(y.getValue(.{ 0, 1 }) == 4);
    try expect(y.getValue(.{ 0, 2 }) == 7);
    try expect(y.getValue(.{ 1, 0 }) == 2);
    try expect(y.getValue(.{ 1, 1 }) == 5);
    try expect(y.getValue(.{ 1, 2 }) == 8);
    try expect(y.getValue(.{ 2, 0 }) == 3);
    try expect(y.getValue(.{ 2, 1 }) == 6);
    try expect(y.getValue(.{ 2, 2 }) == 9);
}
