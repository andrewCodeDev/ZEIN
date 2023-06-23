
// TensorFactory Implementation file. Before proceeding, please read the following:

///////////////////////////////////
// DESIGN PHILOSOPHY (June 5, 2023)

// The TensorFactory is an adapter around a provided allocator type.
// It is primarily here to ensure provide automatic sizing to a given tensor.
// In the future, the TensorFactory will handle things like concatenation
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

const TrackingMode = enum {
    start, 
    stop,
    free
};

// let m = current mode;
// if m == start:
//    free: dealocate memory, m -> free
//    stop: no-op, m -> stop
//
// if m == stop: 
//    free: dealocate memory, m -> free
//    start: no-op, m -> start
//
// if m == free: 
//    m -> start: no-op, m -> start
//    m -> stop: no-op, m -> free

pub fn TensorFactory(comptime value_type: type) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;
        
        const ConstSelfPtr = *const Self;
        
        const ValueType = value_type;

        const ValueSlice = []ValueType;

        const SizesType = SizesAndStridesType;

        const TrackingData = ArrayList([]ValueType);

        gpa_index: usize,
        allocator: Allocator,
        tracking_data: TrackingData,
        tracking_mode: TrackingMode,

        pub fn init(allocator: ?Allocator) Self {

            const index = constructGpaIndex();

            var gpa: *GPA = undefined;

            if(GPABuffer[index]) |*a| {
                gpa = a;
            }
            return Self { 
                .gpa_index = index,
                .allocator = if(allocator)|a| a else gpa.allocator(),
                .tracking_data = TrackingData.init(gpa.allocator()),
                .tracking_mode = TrackingMode.free,
            };
        }

        pub fn deinit(self: SelfPtr) void {
            self.tracking(.free);

            self.tracking_data.deinit();

            if(GPABuffer[self.gpa_index]) |*a| {
                if (a.deinit() == .leak) { @panic("LEAK DETECTED"); }
            }
            nullifyGpaIndex(self.gpa_index);
        }

        fn allocValues(self: SelfPtr, size: usize) !ValueSlice {

            var alloc = try self.allocator.alloc(ValueType, size);

            if(self.tracking_mode == .start){
                try self.tracking_data.append(alloc);
            }
            return alloc;
        }

        fn freeValues(self: SelfPtr, values: ValueSlice) !void {

            if(self.tracking_mode == .start) {
                for(self.tracking_data.items) |data| {
                    if(values.ptr == data.ptr) {
                        return;
                    }
                }
                try self.tracking_data.append(values);  
            }
            else {
                self.allocator.free(values);
            }
        }

        pub fn tracking(self: SelfPtr, mode: TrackingMode) void {

            if(self.tracking_mode == .free and mode == .stop) {
                return; // free is inherently not tracking, so stay free
            }
            if((self.tracking_mode == .start or self.tracking_mode == .stop) and mode == .free) {
                while(self.tracking_data.items.len > 0) { 
                    self.allocator.free(self.tracking_data.pop());
                }
            }
            self.tracking_mode = mode;
        }

        pub fn allocToTensor(self: SelfPtr, tensor: anytype) !void {
            if(tensor.*.valueSize() != 0) {
                return AllocatorError.TensorHasAlloc;
            }            
            tensor.values = try self.allocValues(tensor.valueCapacity());
        }
        
        pub fn freeFromTensor(self: SelfPtr, tensor: anytype) !void {
            try self.freeValues(tensor.*.values);
            tensor.values = &[_]ValueType{ };
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
            var alloc = try self.allocValues(size);
                        
            return Tensor(ValueType, rank, order) {
                .values = alloc,
                .sizes_and_strides = SizesAndStrides(rank, order).init(sizes),
            };
        }

        pub fn copyTensor(self: SelfPtr, tensor: anytype) !@TypeOf(tensor.*) {

            const T = @TypeOf(tensor.*);
            
            var alloc = try self.allocValues(tensor.valueSize());

            @memcpy(alloc, tensor.values);

            return T{
                .values = alloc, 
                .sizes_and_strides = tensor.sizes_and_strides,
            };
        }
    };
}

test "Allocate and Free" {

    const expect = std.testing.expect;

    // null uses the general purpose allocator.
    // It also means that it will call deinit
    // on the gpa allocator when we call deinit.
    var factory = TensorFactory(f32).init(null);

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

    factory.tracking(.start); // beging tracking allocations
    
    ///////////////////////////////////////
    { // assign directly to tensor //////////
        var X = try factory.allocTensor(2, Rowwise, .{ 10, 10 });
        var Y = try factory.copyTensor(&X);

        // create 100 elements... 10x10
        try expect(X.valueSize() == 100);
        try expect(Y.valueSize() == 100);

        try factory.freeFromTensor(&X);
        // do not free y... use deinit

        // tensor slice should be reset
        try expect(X.valueSize() == 0);
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
