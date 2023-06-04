
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

                // ACCESSING TENSOR DATA AND ELEMENTS //

// Tensors, in this system, should not to be mistaken for arrays. Arrays do allow a user
// to access individual elements and modify them. Tensors (for our purpose) are meant
// to be used to manipulate data at a batch level. If a user wants to manipulate an
// individual element, they certainly can by simply using the underlying memory array
// that a tensor is currently viewing.

// Broadly speaking, the handling of individual elements will be done by algorithms
// that work at the scope of tensors. Because of this, Zein should supply a varienty
// of robust batch level operations that allow for these manipulations.

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

pub const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;

pub const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;

const Transpose = @import("Transpose.zig");

// more imports coming soon as they are implemented...

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
    
    return (checked < limit) and (@popCount(checked) == rank);
}

pub fn Tensor(comptime value_type: type, comptime rank: usize) type {

    if(64 <= rank){
        @compileError("Tensors of rank 64 or greater are not supported.");
    }

    return struct {

        pub const Rank = rank;

        pub const ValueType = value_type;

        pub const ValueSlice = []ValueType;

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        value_slice : ValueSlice,
        sizes_and_strides : SizesAndStrides(Rank),

        pub fn init(
            value_slice: ?ValueSlice,
            sizes_and_strides: ?[Rank]SizeAndStride,
        ) Self {
            return Self {
                .value_slice = if (value_slice) |vs| (vs) else &[_]ValueType{},
                .sizes_and_strides = SizesAndStrides(Rank).init(sizes_and_strides),
            };
        }

        pub fn getSizes(self: ConstSelfPtr) [] const u32 {
            return self.*.sizes_and_strides.sliceSizes();
        }
        pub fn getStrides(self: ConstSelfPtr) [] const u32 {
            return self.*.sizes_and_strides.sliceStrides();
        }
        
        pub fn valueCapacity(self: ConstSelfPtr) usize {
            return sliceProduct(self.*.getSizes());
        }
        pub fn valueSize(self: ConstSelfPtr) usize {
            return self.*.value_slice.len;
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
            self.*.value_slice = values;
        }

        // to use this function safely, check that both tensor value
        // sizes are the same and that both tensors are at capacity
        pub fn swapValuesUnchecked(self: SelfPtr, other: SelfPtr) void {
            var tmp = self.*.value_slice;
            self.*.value_slice = other.*.value_slice;
            other.*.value_slice = tmp;
        }

        // to use this function safely, check that the both tensors are at capacity
        pub fn swapSizesAndStridesUnchecked(self: SelfPtr, other: SelfPtr) void {
            // there is probably a faster way to do this
            var tmp = self.*.sizes_and_strides;
            self.*.sizes_and_strides = other.*.sizes_and_strides;
            other.*.sizes_and_strides = tmp;
        }

        // to use this function safely, check that both tensors are at capacity
        pub fn swapTensorsUnchecked(self: SelfPtr, other: SelfPtr) void {
            self.*.swapValuesUnchecked(other);
            self.*.swapSizesAndStridesUnchecked(other);
        }

        // to use this function safely, check that each axis index is present
        pub fn transposeUnchecked(self: SelfPtr, permutation: [Rank]u32) void {
            Transpose.transposeInput(Rank, &self.*.sizes_and_strides, &permutation);
        }

        ///////////////////////////////////////
        // Checked Functions Implementations //

        // Checked functions only succeed if their guard clauses
        // are true. Otherwise, they return false and do not
        // perform the operation. This is to prevent leaving
        // tensors in an invalid state after the operation.

        pub fn setValuesChecked(self: SelfPtr, values: ValueSlice) bool {
            // to assure that sizes and strides are not
            // invalidated, we check size and capacity
            if(self.*.valueCapacity() != values.len){
                return false;
            }
            self.*.setValuesUnchecked(self, values);
            return true;
        }

        pub fn swapValuesChecked(self: SelfPtr, other: SelfPtr) bool {
            // to assure that sizes and strides are not
            // invalidated, we check size and capacity
            if(self.*.valueSize() != other.*.valueSize()){
                return false;
            }
            if(!self.*.atCapacity() or !other.*.atCapacity()) {
                return false;
            }
            self.*.swapValuesUnchecked(other);
            return true;
        }

        pub fn swapSizesAndStridesChecked(self: SelfPtr, other: SelfPtr) bool {
            // we only want to compute these once...
            const capacity_a = self.valueCapactiy();
            const capacity_b = other.valueCapactiy();

            // tensors can have different SizesAndStrides
            // and still share the total value capcity
            if(capacity_a != capacity_b){
                return false;
            }
            // check that both tensors are at capacity without additional computation
            if(self.*.ValueSize() != capacity_a  or other.*.ValueSize() != capacity_b) {
                return false;
            }
            self.*.swapSizesAndStridesUnchecked(other);
            return true;
        }

        pub fn swapTensorsChecked(self: SelfPtr, other: SelfPtr) bool {
            // Two tensors do not need to be the same size to be
            // swapped, we require that they are both at capcity
            if(!self.*.atCapacity() or !other.*.atCapacity()) {
                return false;
            }
            self.*.swapTensorsUnchecked(other);
            return true;
        }

        pub fn transposeChecked(self: SelfPtr, permutation: [] const SizeAndStride.Valuetype) bool {
            // check that all indices are accounted for
            if(Rank != permutation.len){
                return false;
            }
            if(!checkBitwisePermutation(Rank, &permutation)){
                return false;
            }
            Transpose.transposeInput(Rank, &self.*.sizes_and_strides, &permutation);
            return true;
        }
    };
}

test "Initialization" {

    const expect = @import("std").testing.expect;

    var x = Tensor(u32, 3).init(null, [_]SizeAndStride{
        .{ .size = 10, .stride = 10 },
        .{ .size = 20, .stride = 20 },
        .{ .size = 30, .stride = 30 }
    });

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