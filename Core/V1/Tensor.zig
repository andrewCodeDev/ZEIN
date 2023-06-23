
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

// STD import files...
const ReduceOp = @import("std").builtin.ReduceOp;

const arrayProduct = @import("Utility.zig").arrayProduct;

// Zein import files...
pub const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
pub const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
pub const OrderType = @import("SizesAndStrides.zig").OrderType;
pub const Rowwise = @import("SizesAndStrides.zig").Rowwise;
pub const Colwise = @import("SizesAndStrides.zig").Colwise;

const Permutate = @import("Permutate.zig");

// Tensor Utilities...
pub const TensorError = error {
    InvalidTensorLayout,
    InvalidPermutation,
    AllocSizeMismatch,
    CapacityMismatch,
    RankMismatch
};

pub inline fn computeTensorIndex(
    comptime rank: usize,
    comptime value_type: type, 
    strides: *const [rank]value_type,
    indices: [rank]value_type) value_type {
    const i : @Vector(rank, value_type) = indices;
    const s : @Vector(rank, value_type) = strides.*;
    return @reduce(ReduceOp.Add, s * i);
}

///////////////////////////
// Tensor Implementation //

pub fn Tensor(comptime value_type: type, comptime rank: usize, comptime order: OrderType) type {

    if(63 < rank){
        @compileError("Tensors of rank 64 or greater are not supported.");
    }

    if(0 == rank){
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

        values : ValueSlice,

        sizes_and_strides : SizesAndStridesType,
        
        pub fn init(values: ?ValueSlice, sizes: ?[Rank]SizesType) Self {
            return Self {
                .values = if (values) |vs| (vs) else &[_]ValueType{ },
                .sizes_and_strides = SizesAndStridesType.init(sizes),
            };
        }

        pub fn sliceSizes(self: ConstSelfPtr, i: usize, j: usize) [] const SizesType {
            return &self.sizes_and_strides.sizes[i..j];
        }
        pub fn sliceStrides(self: ConstSelfPtr, i: usize, j: usize) [] const SizesType {
            return &self.sizes_and_strides.strides[i..j];
        }
        pub fn slicePermutation(self: ConstSelfPtr, i: usize, j: usize) [] const SizesType {
            return &self.sizes_and_strides.permutation[i..j];
        }

        pub fn getSizes(self: ConstSelfPtr) [] const SizesType {
            return &self.sizes_and_strides.sizes;
        }
        pub fn getStrides(self: ConstSelfPtr) [] const SizesType {
            return &self.sizes_and_strides.strides;
        }
        pub fn getPermutation(self: ConstSelfPtr) [] const SizesType {
            return &self.sizes_and_strides.permutation;
        }
        
        pub fn valueCapacity(self: ConstSelfPtr) usize {
            return arrayProduct(Rank, SizesType, &self.sizes_and_strides.sizes);
        }
        pub fn valueSize(self: ConstSelfPtr) usize {
            return self.values.len;
        }

        // This is a critical function that users should call
        // before using their tensor for operations. This check 
        // ensures that the sizes and strides will enable them
        // to properly access all of the memory within their 
        // tensor and not step overbounds with proper indexing.

        // Default "checked" functions will call this implicity,
        // but please understand that the getValue and setValue
        // do not call this. There is a lengthy comment about
        // that above them - suffice to say, use this check before
        // indexing into your tensor!

        pub fn isValid(self: ConstSelfPtr) bool {
            return self.valueSize() != 0 and self.valueSize() == self.valueCapacity();
        }

        /////////////////////////////////////////
        // Unchecked Functions Implementations //

        // I understand that these functions are verbose,
        // but they are meant to communicate to the user
        // that an operation they are about to do could
        // invalidate a tensor in some way.
        
        // to use this function safely, check that both tensor value
        // sizes are the same and that both tensors are at capacity
        pub fn swapValuesUnchecked(self: SelfPtr, other: SelfPtr) void {
            var values = self.values;
            var index = self.alloc_index;

            // assign values and index from other
            self.values = other.values;
            self.alloc_index = other.alloc_index;

            // assign values and index to other
            other.values = values;
            other.alloc_index = index;
        }

        // to use this function safely, check that the both tensors are at capacity
        pub fn swapSizesAndStridesUnchecked(self: SelfPtr, other: SelfPtr) void {
            // there is probably a faster way to do this
            var tmp = self.sizes_and_strides;
            self.sizes_and_strides = other.sizes_and_strides;
            other.sizes_and_strides = tmp;
        }

        // to use this function safely, check that both tensors are at capacity
        pub fn swapUnchecked(self: SelfPtr, other: SelfPtr) void {
            self.swapValuesUnchecked(other);
            self.swapSizesAndStridesUnchecked(other);
        }

        // to use this function safely, check that the source is
        // valid to ensure that the resulting tensor is also valid.
        pub fn permutateUnchecked(self: SelfPtr, comptime expression: [] const u8) Self {
            var tmp = self.*; // share values and alloc_index!
            Permutate.permutate(Rank, Order, expression, &tmp.sizes_and_strides);
            return tmp;
        }

        ///////////////////////////////////////
        // Checked Functions Implementations //

        // Checked functions only succeed if their guard clauses
        // are true. Otherwise, they return errors and do not
        // perform the operation. This is to prevent leaving
        // tensors in an invalid state after the operation.

        pub fn swapValues(self: SelfPtr, other: SelfPtr) !void {
            // to assure that sizes and strides are not
            // invalidated, we check size and capacity
            if(self.valueSize() != other.valueSize()){
                return TensorError.AllocSizeMismatch;
            }
            if(!self.isValid() or !other.isValid()) {
                return TensorError.InvalidTensorLayout;
            }
            self.swapValuesUnchecked(other);
        }

        pub fn swapSizesAndStrides(self: SelfPtr, other: SelfPtr) !void {
            // we only want to compute these once...
            const capacity_a = self.valueCapactiy();
            const capacity_b = other.valueCapactiy();

            // tensors can have different SizesAndStrides
            // and still share the total value capcity
            if(capacity_a != capacity_b){
                return TensorError.CapacityMismatch;
            }
            // check that both tensors are at capacity without additional computation
            if(self.valueSize() != capacity_a  or other.valueSize() != capacity_b) {
                return TensorError.InvalidTensorLayout;
            }
            self.swapSizesAndStridesUnchecked(other);
        }

        pub fn swap(self: SelfPtr, other: SelfPtr) !void {
            // Two tensors do not need to be the same size to be swapped.
            // They only need to both be valid tensors to prevent invalidation.
            if(!self.isValid() or !other.isValid()) {
                return TensorError.InvalidTensorLayout;
            }
            self.swapUnchecked(other);
        }

        pub fn permutate(self: SelfPtr, comptime expression: [] const u8) !Self {
            // create a permutated tensor that shares the same underlying memory
            if(!self.isValid()) {
                return TensorError.InvalidTensorLayout;
            }
            return self.permutateUnchecked(expression);
        }

        /////////////////////////////////////////////////
        // !!! Value Getter and Setters are UNCHECKED !!!

        // I debated with myself about having a Unchecked/Default version of
        // these functions, but I can't justify the cost. Here's why...
        // it is too cumbersome to put a "try" infront of these and will
        // cause tensor expressions to be extremely awkward.

        // I have rarely seen C++ implementations of tensors that use
        // the std::vector::at function (which is range checked). Instead,
        // almost everyone opts for the std::vector::operator[].
        // This is because it's natural to write x[0] + y[0].

        // Also, we could check *all kinds of things* here. For instance,
        // are you at capacity? Even if you're not, you could have your
        // strides zeroed out and a value.len == 0. Should we test for
        // that too? Then, we could test that you are within range
        // and that means each index is valid for each size axis. This is
        // simply untenable. In practice, what that means is that everyone 
        // will just use the unchecked function call instead (I know I would).
        
        // Due to this, it is critical that the user takes the (very common)
        // burden of checking that their indices are within range. Atop that,
        // the user should also call isValid before using their tensors
        // to ensure that everything lines up before using a tensor to be
        // certain that they are doing valid indexing.
        pub fn getValue(self: ConstSelfPtr, indices: [rank]SizesType) ValueType {
            const n = computeTensorIndex(
                Rank, SizesType, &self.sizes_and_strides.strides, indices
            );
            return self.values[n];
        }

        pub fn setValue(self: ConstSelfPtr, value: ValueType, indices: [rank]SizesType) void {
            const n = computeTensorIndex(
                Rank, SizesType, &self.sizes_and_strides.strides, indices
            );            
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

    const expect = @import("std").testing.expect;

    var x = Tensor(u32, 3, Rowwise).init(
            null, .{ 10, 20, 30 }
        );    

    const total: usize = 10 * 20 * 30;

    try expect(total == x.valueCapacity());        
}

test "Tensor Transpose" {
    const expect = @import("std").testing.expect;

    var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    var x = Tensor(i32, 2, Rowwise).init(&data, .{ 3, 3 });    

    try expect(x.isValid());

    try expect(x.getValue(.{0,0}) == 1);
    try expect(x.getValue(.{0,1}) == 2);
    try expect(x.getValue(.{0,2}) == 3);
    try expect(x.getValue(.{1,0}) == 4);
    try expect(x.getValue(.{1,1}) == 5);
    try expect(x.getValue(.{1,2}) == 6);
    try expect(x.getValue(.{2,0}) == 7);
    try expect(x.getValue(.{2,1}) == 8);
    try expect(x.getValue(.{2,2}) == 9);

    var y = try x.permutate("ij->ji");

    try expect(y.getValue(.{0,0}) == 1);
    try expect(y.getValue(.{0,1}) == 4);
    try expect(y.getValue(.{0,2}) == 7);
    try expect(y.getValue(.{1,0}) == 2);
    try expect(y.getValue(.{1,1}) == 5);
    try expect(y.getValue(.{1,2}) == 8);
    try expect(y.getValue(.{2,0}) == 3);
    try expect(y.getValue(.{2,1}) == 6);
    try expect(y.getValue(.{2,2}) == 9);
}