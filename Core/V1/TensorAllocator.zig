
// TensorAllocator Implementation file. Before proceeding, please read the following:

///////////////////////////////////
// DESIGN PHILOSOPHY (June 5, 2023)

// The TensorAllocator is an adapter around a provided allocator type.
// It is primarily here to ensure provide automatic sizing to a given tensor.
// In the future, the TensorAllocator will handle things like concatenation
// because that is ultimately a memory operation.

// Fundamentally, this class can be avoided if you intend to use your own
// allocations to assign to tensor values. The allocatoins will still be 
// checked if using the default functions.

// Allocators still need to have the deinit() function called as per usual.

const Allocator = @import("std").mem.Allocator;

const Tensor = @import("Tensor.zig").Tensor;

const SizesAndStridesType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;

// Zein import files...
const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
const OrderType = @import("SizesAndStrides.zig").OrderType;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;

pub const AllocatorError = error {
    UnknownObject,
    TensorSizeZero,
    TensorHasAlloc
};

const sliceProduct = @import("Utility.zig").sliceProduct;

pub fn TensorAllocator(comptime value_type: type) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;
        
        const ConstSelfPtr = *const Self;
        
        const ValueType = value_type;

        const SizesType = SizesAndStridesType;
        
        allocator: Allocator,    

        pub fn init(allocator: Allocator) Self {
            return Self{ .allocator = allocator };
        }

        pub fn allocValues(self: SelfPtr, size: usize) ![]ValueType {
            var values = try self.*.allocator.alloc(ValueType, size);
            return values;
        }

        pub fn freeValues(self: SelfPtr, values: []ValueType) void {
            self.*.allocator.free(values);
        }

        pub fn allocToTensor(self: SelfPtr, tensor: anytype) !void {
            if(tensor.*.valueSize() != 0) {
                return AllocatorError.TensorHasAlloc;
            }
            if(tensor.*.valueCapacity() == 0) {
                return AllocatorError.TensorSizeZero;
            } 
            const size = sliceProduct(SizesType, tensor.*.sliceSizes());
                        
            tensor.*.values = try self.*.allocValues(size);
        }
        
        pub fn freeFromTensor(self: SelfPtr, tensor: anytype) void {
            self.*.freeValues(tensor.*.releaseValues());            
        }

        // I feel this function is justified because it not only
        // provides syntactic sugar for the creating tensors, but
        // it also ensures that the tensor's construction is valid.

        pub fn allocTensor(
            self: SelfPtr, 
            comptime rank: usize,
            comptime order: OrderType,
            sizes: [rank]SizesType) !Tensor(ValueType, rank, order) {

            const size = sliceProduct(SizesType, &sizes);

            if(size == 0) {
                return AllocatorError.TensorSizeZero;
            } 
            var data = try self.*.allocValues(size);
                        
            return Tensor(ValueType, rank, order).init(data, sizes);
        }

        pub fn copyTensor(self: SelfPtr, tensor: anytype) !@TypeOf(tensor.*) {
            const T = @TypeOf(tensor.*);
            
            var values = try self.allocValues(tensor.*.valueSize());

            @memcpy(values, tensor.*.values);

            return T { 
                .values = values, .sizes_and_strides = tensor.*.sizes_and_strides
            };
        }
    };
}

test "Allocate and Free" {

    const std = @import("std");

    const expect = std.testing.expect;
    
    var GPA = std.heap.GeneralPurposeAllocator(.{ }){ };

    var factory = TensorAllocator(f32).init(GPA.allocator());

    /////////////////////////////////////////
    { // assign into to tensor //////////////
        var X = Tensor(f32, 2, Rowwise).init(
            null, .{ 10, 10 }
        );

        // create 100 elements... 10x10
        try factory.allocToTensor(&X);
        try expect(X.valueSize() == 100);

        // tensor slice should be reset
        factory.freeFromTensor(&X);
        try expect(X.valueSize() == 0);
    }

    /////////////////////////////////////////
    { // assign indirectly to tensor ////////
        var X = Tensor(f32, 2, Rowwise).init(
            null, .{ 10, 10 }
        );

        // create 100 elements... 10x10
        var values = try factory.allocValues(X.valueCapacity());
        try X.setValues(values);
        try expect(X.valueSize() == 100);

        // tensor slice should be reset
        factory.freeValues(X.releaseValues());
        try expect(X.valueSize() == 0);
    }

    /////////////////////////////////////////
    { // assign directly to tensor //////////
        var X = try factory.allocTensor(2, Rowwise, .{ 10, 10 });

        // create 100 elements... 10x10
        try expect(X.valueSize() == 100);

        // tensor slice should be reset
        factory.freeFromTensor(&X);
        try expect(X.valueSize() == 0);
    }
    /////////////////////////////////////////
    { // assign directly to tensor //////////
        var X = try factory.allocTensor(2, Rowwise, .{ 10, 10 });
        var Y = try factory.copyTensor(&X);

        // create 100 elements... 10x10
        try expect(X.valueSize() == 100);
        try expect(Y.valueSize() == 100);

        // tensor slice should be reset
        factory.freeFromTensor(&X);
        factory.freeFromTensor(&Y);

        try expect(X.valueSize() == 0);
        try expect(Y.valueSize() == 0);
    }
    
    if (GPA.deinit() == .leak) { @panic("LEAK DETECTED"); }
}
