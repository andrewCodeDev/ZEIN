
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

const std = @import("std");

const Allocator = @import("std").mem.Allocator;

const Tensor = @import("Tensor.zig").Tensor;

const SizesAndStridesType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;

// Zein import files...
const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
const OrderType = @import("SizesAndStrides.zig").OrderType;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;

const sliceProduct = @import("Utility.zig").sliceProduct;

pub const AllocatorError = error {
    UnknownObject,
    TensorSizeZero,
    TensorHasAlloc,
    WrongAllocator,
    IndexAlreadyFreed,
    InvalidIndex
};

// used to keep track of tensor allocations
const ArrayList = @import("std").ArrayList;

const Mutex = @import("std").Thread.Mutex;

// this is a current work around until more work can
// go into the allocator itself - right now, it uses
// an ArrayList of allocations to free memory. Unfortunately,
// that means that it also needs an allocator itself.
// at some point, this should be replaced, but it
// works for now. I'm setting the BufferSize to something
// large enough to handle anything reasonable (and then some...)

// GPA for null initialized tensor allocators.
const GPA = std.heap.GeneralPurposeAllocator(.{ });

// A number large enough that it shouldn't matter.
const BufferSize = 100;
var BufferMutex = Mutex{};
var GPABuffer: [BufferSize]?GPA = undefined;
var GPAUsed: usize = 0;

fn constructGpaIndex() usize {
    BufferMutex.lock();

    defer BufferMutex.unlock();

    var i: usize = 0;

    while(i < BufferSize) : (i += 1){
        if (GPABuffer[i] == null) { 
            GPABuffer[i] = GPA{};
            return i; 
        }
    }
    @panic("Too many tensor allocator instances.");
}

/////////////////////////////////////////
// only call this after calling deinit!!!
fn nullifyGpaIndex(index: usize) void {
    BufferMutex.lock();

    GPABuffer[index] = null;

    BufferMutex.unlock();
}


pub fn TensorAllocator(comptime value_type: type) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;
        
        const ConstSelfPtr = *const Self;
        
        const ValueType = value_type;

        const SizesType = SizesAndStridesType;

        const StorageType = ArrayList(?[]ValueType);

        const IndexedAlloc = struct {
            index: usize,
            alloc: []ValueType,
        };

        gpa_index: usize,
        
        allocator: Allocator,    

        storage: StorageType,

        pub fn init(allocator: ?Allocator) Self {

            const index = constructGpaIndex();

            var gpa: *GPA = undefined;

            if(GPABuffer[index]) |*a| {
                gpa = a;
            }
            return Self { 
                .gpa_index = index,
                .allocator = if(allocator)|a| a else gpa.allocator(),
                .storage = StorageType.init(gpa.allocator()),
            };
        }

        pub fn deinit(self: SelfPtr) void {
            // free all existing allocations if they are not null
            for(self.storage.items) |item| {
                if(item) |alloc| { self.allocator.free(alloc); }
            }
            self.storage.deinit();
            
            const index = self.gpa_index;

            if(GPABuffer[index]) |*a| {
                if (a.deinit() == .leak) { @panic("LEAK DETECTED"); }
            }
            nullifyGpaIndex(index);
        }

        pub fn allocValues(self: SelfPtr, size: usize, old_index: ?usize) !IndexedAlloc {

            // I'm using the old_index parameter to be able to re-use
            // data members from the storage array. If for some reason
            // the user is freeing and reallocating frequently, it
            // would cause the storage to grow linearly over time.

            // The public member functions need to do the heavy
            // lifting of seeing if it's safe to assign new values

            if(size == 0) {
                return AllocatorError.TensorSizeZero;
            }
            var alloc = try self.allocator.alloc(ValueType, size);

            // if old_index is not null, check to see if it fits
            // fits the storage capacity to re-use existing item
            if(old_index) |index| {
                if(index <= self.storage.items.len) {
                    // reuse existing storage item if null
                    if(self.storage.items[index] == null) {
                        self.storage.items[index] = alloc;
                        return IndexedAlloc{ .index = index, .alloc = alloc };
                    }
                }
            }
            // if old_index was null or was larger than
            // the storage capacity, append a new item
            const index = self.storage.items.len;
            try self.storage.append(alloc);
            return IndexedAlloc{ .index = index, .alloc = alloc };
        }

        pub fn freeValues(self: SelfPtr, values: []ValueType, index: usize) !void {

            // we need to check that the index is valid
            // for the storage that we have and then
            // see if the pointers are the same.

            if(self.storage.items.len <= index) {
                return AllocatorError.WrongAllocator;
            }
            if(self.storage.items[index]) |alloc| {
                if(alloc.ptr != values.ptr){
                    return AllocatorError.WrongAllocator;
                }
                self.storage.items[index] = null;
                self.allocator.free(values);
            }
            else {
                return AllocatorError.IndexAlreadyFreed;
            }
        }

        pub fn allocToTensor(self: SelfPtr, tensor: anytype) !void {

            // pretty simple, make a new allocation
            // and then assign the storage index
            if(tensor.*.valueSize() != 0) {
                return AllocatorError.TensorHasAlloc;
            }            
            var indexed_alloc = try self.allocValues(
                tensor.*.valueCapacity(), tensor.*.alloc_index 
            );
            tensor.*.values = indexed_alloc.alloc;
            tensor.*.alloc_index = indexed_alloc.index;
        }
        
        pub fn freeFromTensor(self: SelfPtr, tensor: anytype) !void {

            // this function leaves the storage index intact so
            // the same tensor can request allocations again
            if(tensor.*.alloc_index) |index| {
                try self.freeValues(tensor.*.values, index);
                tensor.*.values = &[_]ValueType{ };
            }
            else {
                return AllocatorError.WrongAllocator;
            }
        }

        pub fn allocTensor(
            self: SelfPtr, 
            comptime rank: usize,
            comptime order: OrderType,
            sizes: [rank]SizesType) !Tensor(ValueType, rank, order) {

            const size = sliceProduct(SizesType, &sizes);

            if(size == 0) {
                return AllocatorError.TensorSizeZero;
            } 
            var indexed_alloc = try self.allocValues(size, null);
                        
            return Tensor(ValueType, rank, order) {
                .values = indexed_alloc.alloc,
                .sizes_and_strides = SizesAndStrides(rank, order).init(sizes),
                .alloc_index = indexed_alloc.index,
            };
        }

        pub fn copyTensor(self: SelfPtr, tensor: anytype) !@TypeOf(tensor.*) {

            const T = @TypeOf(tensor.*);
            
            var indexed_alloc = try self.allocValues(tensor.*.valueSize(), null);

            @memcpy(indexed_alloc.alloc, tensor.*.values);

            return T {
                .values = indexed_alloc.alloc,
                .sizes_and_strides = tensor.*.sizes_and_strides,
                .alloc_index = indexed_alloc.index,
            };
        }
    };
}

test "Allocate and Free" {

    const expect = std.testing.expect;

    // null uses the general purpose allocator.
    // It also means that it will call deinit
    // on the gpa allocator when we call deinit.
    var factory = TensorAllocator(f32).init(null);

    /////////////////////////////////////////
    { // assign into to tensor //////////////
        var X = Tensor(f32, 2, Rowwise).init(null, .{ 10, 10 });

        // create 100 elements... 10x10
        try factory.allocToTensor(&X);
        try expect(X.valueSize() == 100);

        // tensor slice should be reset
        try factory.freeFromTensor(&X);
        try expect(X.valueSize() == 0);
    }

    /////////////////////////////////////////
    { // assign directly to tensor //////////
        var X = try factory.allocTensor(2, Rowwise, .{ 10, 10 });

        // create 100 elements... 10x10
        try expect(X.valueSize() == 100);

        // tensor slice should be reset
        try factory.freeFromTensor(&X);
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
        try factory.freeFromTensor(&X);
        try factory.freeFromTensor(&Y);

        try expect(X.valueSize() == 0);
        try expect(Y.valueSize() == 0);
    }

    
    // make 3 tensors and do not free them
    var X = try factory.allocTensor(2, Rowwise, .{ 10, 10 });
    var Y = try factory.allocTensor(2, Rowwise, .{ 10, 10 });
    var Z = try factory.allocTensor(2, Rowwise, .{ 10, 10 });

    // trivial operation to avoid compile error
    X.setValue(3, .{0, 1});
    Y.setValue(3, .{0, 1});
    Z.setValue(3, .{0, 1});

    // factory will free X Y Z automatically
    // and this will panic if test fails
    factory.deinit();
}
