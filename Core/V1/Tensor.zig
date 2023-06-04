
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
const exitProgram = @import("std").os.exit;
const ReduceOp = @import("std").builtin.ReduceOp;

// Zein import files...
pub const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
pub const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
const OrderType = @import("SizesAndStrides.zig").OrderType;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;

const Permutate = @import("Permutate.zig");

// Tensor Utilities...
const TensorError = error {
    SizeAndCapacityMismatch,
    InvalidPermutation,
    AllocSizeMismatch,
    CapacityMismatch,
    RankMismatch
};

fn sliceProduct(slice: [] const SizeAndStride.ValueType) SizeAndStride.ValueType {
    var total: SizeAndStride.ValueType = 1;
    for(slice) |n| { 
        total *= n;
    }
    return total * @boolToInt(0 < slice.len);
}

fn checkBitwisePermutation(comptime rank: usize, permutation: *const [rank]u32) bool {
    // O(N) operation to check for valid permutations.

    // All indices of the SizesAndStrides must be
    // checked before we can permutate. Otherwise,
    // this could mean that a transpose operation
    // could leave a tensor in an invalid state.

    // bitwise limit to check if an index is out of bounds
    const limit: usize = ((rank + 1) << 1);

    // storage for bitwise OR operations checks
    var checked: usize = 0;

    // bit shifting zero by one is a no-op
    // this is a work-around for indexing
    var is_zero: usize = 0;
    
    for(permutation.*) |i| { 
        checked |= (i << 1); 
        is_zero |= @boolToInt((i == 0));
    }
    checked += is_zero;
    
    return (checked < limit) and @popCount(checked) == rank;
}

inline fn computeTensorIndex(
    comptime rank: usize,
    comptime value_type: type, 
    strides: *const [rank]u32,
    indices: [rank]u32) u32 {
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

        pub const ValueType = value_type;

        pub const ValueSlice = []ValueType;

        pub const SizesAndStridesType = SizesAndStrides(Rank, Order);

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        values : ValueSlice,
        sizes_and_strides : SizesAndStridesType,

        pub fn init(
            values: ?ValueSlice,
            sizes : ?[Rank]u32,
        ) Self {
            return Self {
                .values = if (values) |vs| (vs) else &[_]ValueType{},
                .sizes_and_strides = SizesAndStridesType.init(sizes),
            };
        }

        pub fn getSizes(self: ConstSelfPtr) [] const u32 {
            return &self.*.sizes_and_strides.sizes;
        }
        pub fn getStrides(self: ConstSelfPtr) [] const u32 {
            return &self.*.sizes_and_strides.strides;
        }
        
        pub fn valueCapacity(self: ConstSelfPtr) usize {
            return sliceProduct(self.*.getSizes());
        }
        pub fn valueSize(self: ConstSelfPtr) usize {
            return self.*.values.len;
        }

        pub fn atCapacity(self: ConstSelfPtr) bool {
            return self.*.valueSize() == self.*.valueCapacity();
        }

        /////////////////////////////////////////
        // Unchecked Functions Implementations //

        // I understand that these functions are verbose,
        // but they are meant to communicate to the user
        // that an operation they are about to do could
        // invalidate a tensor in some way.
        
        // to use this function safely, check that both slice lenghts are
        // the same and that the capacity is equal to the value length
        pub fn setValuesUnchecked(self: SelfPtr, values: ValueSlice) void {
            self.*.values = values;
        }

        // to use this function safely, check that both tensor value
        // sizes are the same and that both tensors are at capacity
        pub fn swapValuesUnchecked(self: SelfPtr, other: SelfPtr) void {
            var tmp = self.*.values;
            self.*.values = other.*.values;
            other.*.values = tmp;
        }

        // to use this function safely, check that the both tensors are at capacity
        pub fn swapSizesAndStridesUnchecked(self: SelfPtr, other: SelfPtr) void {
            // there is probably a faster way to do this
            var tmp = self.*.sizes_and_strides;
            self.*.sizes_and_strides = other.*.sizes_and_strides;
            other.*.sizes_and_strides = tmp;
        }

        // to use this function safely, check that both tensors are at capacity
        pub fn swapUnchecked(self: SelfPtr, other: SelfPtr) void {
            self.*.swapValuesUnchecked(other);
            self.*.swapSizesAndStridesUnchecked(other);
        }

        // to use this function safely, check that each index from 0..Rank is present
        pub fn permutateUnchecked(self: SelfPtr, permutation: [Rank]SizeAndStride.ValueType) void {
            Permutate.permutateInput(Rank, Order, &self.*.sizes_and_strides, permutation);
        }

        // to use this function safely, ensure that no indices are out of bounds
        fn getValueUnchecked(self: ConstSelfPtr, indices: [rank]u32) u32 {
            const n = computeTensorIndex(
                Rank, SizeAndStride.ValueType, &self.*.sizes_and_strides.strides, indices
            );
            return self.*.values[n];
        }

        // to use this function safely, ensure that no indices are out of bounds
        fn setValueUnchecked(self: ConstSelfPtr, value: ValueType, indices: [rank]u32) void {
            const n = computeTensorIndex(
                Rank, SizeAndStride.ValueType, &self.*.sizes_and_strides.strides, indices
            );
            self.*.values[n] = value;
        }

        ///////////////////////////////////////
        // Checked Functions Implementations //

        // Checked functions only succeed if their guard clauses
        // are true. Otherwise, they return false and do not
        // perform the operation. This is to prevent leaving
        // tensors in an invalid state after the operation.

        pub fn setValues(self: SelfPtr, values: ValueSlice) bool {
            // to assure that sizes and strides are not
            // invalidated, we check size and capacity
            if(self.*.valueCapacity() != values.len){
                return TensorError.AllocSizeMismatch;
            }
            self.*.setValuesUnchecked(self, values);
            return true;
        }

        pub fn swapValues(self: SelfPtr, other: SelfPtr) !void {
            // to assure that sizes and strides are not
            // invalidated, we check size and capacity
            if(self.*.valueSize() != other.*.valueSize()){
                return TensorError.AllocSizeMismatch;
            }
            if(!self.*.atCapacity() or !other.*.atCapacity()) {
                return TensorError.SizeAndCapacityMismatch;
            }
            self.*.swapValuesUnchecked(other);
        }

        pub fn swapSizesAndStrides(self: SelfPtr, other: SelfPtr) !void {
            // we only want to compute these once...
            const capacity_a = self.valueCapactiy();
            const capacity_b = other.valueCapactiy();

            // tensors can have different SizesAndStrides
            // and still share the total value capcity
            if(capacity_b != capacity_b){
                return TensorError.CapacityMismatch;
            }
            // check that both tensors are at capacity without additional computation
            if(self.*.ValueSize() != capacity_a  or other.*.ValueSize() != capacity_b) {
                return TensorError.SizeAndCapacityMismatch;
            }
            self.*.swapSizesAndStridesUnchecked(other);
        }

        pub fn swap(self: SelfPtr, other: SelfPtr) !void {
            // Two tensors do not need to be the same size to be
            // swapped, we require that they are both at capcity
            if(!self.*.atCapacity() or !other.*.atCapacity()) {
                return TensorError.SizeAndCapacityMismatch;
            }
            self.*.swapUnchecked(other);
        }

        fn getValue(self: ConstSelfPtr, indices: [rank]u32) ValueType {
            // We find ourselves at a difficult decision to make...
            // How to handle this error? It is too cumbersome to check
            // a return type for an error and make serious use of this
            // function. Because of this, the at function will simply
            // call for program termination for invalid memory access.
            const n = computeTensorIndex(
                Rank, SizeAndStride.ValueType, &self.*.sizes_and_strides.strides, indices
            );
            if (self.*.valueSize() <= n) {
                exitProgram(1);
            }
            return self.*.values[n];
        }

        fn setValue(self: ConstSelfPtr, value: ValueType, indices: [rank]u32) void {
            // We find ourselves at a difficult decision to make...
            // How to handle this error? It is too cumbersome to check
            // a return type for an error and make serious use of this
            // function. Because of this, the at function will simply
            // call for program termination for invalid memory access.
            const n = computeTensorIndex(
                Rank, SizeAndStride.ValueType, &self.*.sizes_and_strides.strides, indices
            );
            if (self.*.valueSize() <= n) {
                exitProgram(1);
            }
            self.*.values[n] = value;
        }

        pub fn permutate(self: SelfPtr, permutation: [rank]SizeAndStride.ValueType) !void {
            // check that all indices are accounted for
            if(!checkBitwisePermutation(Rank, &permutation)){
                return TensorError.InvalidPermutation;
            }
            Permutate.permutateInput(Rank, Order, &self.*.sizes_and_strides, &permutation);
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

test "Bitwise-Permutation" {
    const expect = @import("std").testing.expect;

    // valid permutation checks...
    try expect(checkBitwisePermutation(3, &.{ 0, 1, 2 }));
    try expect(checkBitwisePermutation(3, &.{ 0, 2, 1 }));
    try expect(checkBitwisePermutation(3, &.{ 1, 0, 2 }));
    try expect(checkBitwisePermutation(3, &.{ 1, 2, 0 }));
    try expect(checkBitwisePermutation(3, &.{ 2, 0, 1 }));
    try expect(checkBitwisePermutation(3, &.{ 2, 1, 0 }));
    
    // invalid permutation checks...
    try expect(!checkBitwisePermutation(3, &.{ 0, 1, 0 }));
    try expect(!checkBitwisePermutation(3, &.{ 0, 2, 6 }));
    try expect(!checkBitwisePermutation(3, &.{ 0, 0, 0 }));
    try expect(!checkBitwisePermutation(3, &.{ 6, 7, 8 }));
    try expect(!checkBitwisePermutation(3, &.{ 1, 2, 2 }));
    try expect(!checkBitwisePermutation(3, &.{ 1, 2, 3 }));
}

test "Tensor at function" {
    const expect = @import("std").testing.expect;

    var data = [9]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    var x = Tensor(i32, 2, Rowwise).init(
            &data, .{ 3, 3 }
        );    

    try expect(x.getValue(.{0,0}) == 1);
    try expect(x.getValue(.{0,1}) == 2);
    try expect(x.getValue(.{0,2}) == 3);
    try expect(x.getValue(.{1,0}) == 4);
    try expect(x.getValue(.{1,1}) == 5);
    try expect(x.getValue(.{1,2}) == 6);
    try expect(x.getValue(.{2,0}) == 7);
    try expect(x.getValue(.{2,1}) == 8);
    try expect(x.getValue(.{2,2}) == 9);

    try x.permutate(.{1, 0});

    try expect(x.getValue(.{0,0}) == 1);
    try expect(x.getValue(.{0,1}) == 4);
    try expect(x.getValue(.{0,2}) == 7);
    try expect(x.getValue(.{1,0}) == 2);
    try expect(x.getValue(.{1,1}) == 5);
    try expect(x.getValue(.{1,2}) == 8);
    try expect(x.getValue(.{2,0}) == 3);
    try expect(x.getValue(.{2,1}) == 6);
    try expect(x.getValue(.{2,2}) == 9);
}