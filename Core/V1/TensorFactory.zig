
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

// Zein import files...
const std = @import("std");
const Allocator = @import("std").mem.Allocator;
const Tensor = @import("Tensor.zig").Tensor;
const TensorError = @import("Tensor.zig").TensorError;
const SizesAndStridesType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;

const SizeAndStride = @import("SizesAndStrides.zig").SizeAndStride;
const SizesAndStrides = @import("SizesAndStrides.zig").SizesAndStrides;
const OrderType = @import("SizesAndStrides.zig").OrderType;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;
const Ops = @import("TensorOps.zig");
const OpsError = @import("TensorOps.zig").OpsError;
const contractionParse = @import("ExpressionParsing.zig").contractionParse;
const innerProductParse = @import("ExpressionParsing.zig").innerProductParse;
const contractedRank = @import("ExpressionParsing.zig").contractedRank;
const sliceProduct = @import("Utility.zig").sliceProduct;

const LinearCachingAllocator = @import("LinearCachingAllocator.zig").LinearCachingAllocator;

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
const LCA = LinearCachingAllocator;

// A number large enough that it shouldn't matter.
const BufferSize = 100;
var BufferMutex = Mutex{};
var LCABuffer: [BufferSize]?LCA = undefined;
var LCAUsed: usize = 0;

fn constructLCA() Allocator {
    BufferMutex.lock();
    defer BufferMutex.unlock();
    var i: usize = 0;
    while(i < BufferSize) : (i += 1){
        if (LCABuffer[i] == null) { 
            LCABuffer[i] = LCA{ };
            return LCABuffer[i].?.allocator(); 
        }
    }
    @panic("Too many tensor allocator instances.");
}

fn nullifyLCA(expired: *anyopaque) void {
    BufferMutex.lock();
    defer BufferMutex.unlock();
    for(0..BufferSize) |i| {
        if(LCABuffer[i]) |*lca| {
            if (@intFromPtr(lca) == @intFromPtr(expired)) { 
                LCABuffer[i].?.deinit();
                LCABuffer[i] = null;
            }
        }
    }
}

fn isDefaultLCA(ptr: *anyopaque) bool {
    const buffer_a = @intFromPtr(&LCABuffer[0]);
    const buffer_b = @intFromPtr(&LCABuffer[BufferSize - 1]);
    const check = @intFromPtr(ptr);
    // check if we are inbounds of the static array
    return (buffer_a <= check) or (check <= buffer_b);
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
//    start: no-op, m -> start
//    stop: no-op, m -> free

pub fn TensorFactory(comptime value_type: type) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;
        
        const ConstSelfPtr = *const Self;
        
        const ValueType = value_type;

        const ValueSlice = []ValueType;

        const SizesType = SizesAndStridesType;

        const TrackingData = ArrayList([]ValueType);

        allocator: Allocator,
        tracking_data: TrackingData,
        tracking_mode: TrackingMode,

        pub fn init(allocator: ?Allocator) Self {

            return Self { 
                .allocator = if(allocator) |a| a else constructLCA(),
                .tracking_data = TrackingData.init(std.heap.page_allocator),
                .tracking_mode = TrackingMode.free,
            };
        }

        pub fn deinit(self: SelfPtr) void {
            self.tracking(.free);

            self.tracking_data.deinit();

            // check if we need to nullify an LCA
            if(isDefaultLCA(self.allocator.ptr)) {
                nullifyLCA(self.allocator.ptr);
            }
        }

        ///////////////////////////////////
        // private allocation functions ///

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

        ///////////////////////////////////
        // Change the tracking mode ///////

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

        //////////////////////////////////
        // Tensor Allocation functions ///

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

        /////////////////////////////
        // Factory Math Functions ///

        pub fn add(self: SelfPtr, x: anytype, y: anytype) !@TypeOf(x.*) {
            if(@TypeOf(x.*) != @TypeOf(y.*)) {
                @compileError("Addition requires operands to be of the same type.");
            }
            if(!x.isValid() or !y.isValid()){
                return TensorError.InvalidTensorLayout;
            }
            if(x.valueSize() != y.valueSize()){
                return OpsError.UnequalSize;
            }
            var z = try self.allocTensor(
                @TypeOf(x.*).Rank, @TypeOf(x.*).Order, x.sizes_and_strides.sizes
            );
            Ops.addUnchecked(x, y, &z); 
            return z;
        }

        pub fn sub(self: SelfPtr, x: anytype, y: anytype) !@TypeOf(x.*) {
            if(@TypeOf(x.*) != @TypeOf(y.*)) {
                @compileError("Addition requires operands to be of the same type.");
            }
            if(!x.isValid() or !y.isValid()){
                return TensorError.InvalidTensorLayout;
            }
            if(x.valueSize() != y.valueSize()){
                return OpsError.UnequalSize;
            }
            var z = try self.allocTensor(
                @TypeOf(x.*).Rank, @TypeOf(x.*).Order, x.sizes_and_strides.sizes
            );
            Ops.subUnchecked(x, y, &z); 
            return z;
        }

        pub fn mul(self: SelfPtr, x: anytype, y: anytype) !@TypeOf(x.*) {
            if(@TypeOf(x.*) != @TypeOf(y.*)) {
                @compileError("Multipication requires operands to be of the same type.");
            }
            if(!x.isValid() or !y.isValid()){
                return TensorError.InvalidTensorLayout;
            }
            if(x.valueSize() != y.valueSize()){
                return OpsError.UnequalSize;
            }
            var z = try self.allocTensor(
                @TypeOf(x.*).Rank, @TypeOf(x.*).Order, x.sizes_and_strides.sizes
            );
            Ops.mulUnchecked(x, y, &z); 
            return z;
        }

        pub fn bias(self: SelfPtr, x: anytype, b: @TypeOf(x.*).ValueType) !@TypeOf(x.*) {
            if(!x.isValid()){
                return TensorError.InvalidTensorLayout;
            }
            var y = try self.allocTensor(
                @TypeOf(x.*).Rank, @TypeOf(x.*).Order, x.sizes_and_strides.sizes
            );
            Ops.biasUnchecked(x, &y, b); 
            return y;
        }

        pub fn scale(self: SelfPtr, x: anytype, s: @TypeOf(x.*).ValueType) !@TypeOf(x.*) {
            if(!x.isValid()){
                return TensorError.InvalidTensorLayout;
            }
            var y = try self.allocTensor(
                @TypeOf(x.*).Rank, @TypeOf(x.*).Order, x.sizes_and_strides.sizes
            );
            Ops.scaleUnchecked(x, &y, s); 
            return y;
        }

        pub fn contraction(
            self: SelfPtr, 
            comptime expression: [] const u8, 
            x: anytype
            ) !Tensor(ValueType, contractedRank(expression), @TypeOf(x.*).Order) {

            if(!x.isValid()) {
                return TensorError.InvalidTensorLayout;
            }
            const XRank = @TypeOf(x.*).Rank;
            const YRank = comptime contractedRank(expression);

            const ip = comptime contractionParse(XRank, YRank, expression);

            var y_ss: [YRank]SizesType = undefined;
            {
                var i: usize = 0;
                while(i < YRank) : (i += 1) {
                    y_ss[i] = x.sizes_and_strides.sizes[ip.lhs[i]];
                }
            }
            var y = try self.allocTensor(YRank, @TypeOf(x.*).Order, y_ss);

            var xc: [XRank]SizesType = undefined;
            var yc: [YRank]SizesType = undefined;
            
            @memset(y.values, 0);
            
            @call(.always_inline, Ops.recursiveContraction, .{
                ValueType, SizesType, XRank, YRank, ip.lhs, ip.rhs, 0, x, &y, &xc, &yc
            });
            return y;
        }

        pub fn innerProduct(
            self: SelfPtr, 
            comptime expression: [] const u8, 
            x: anytype,
            y: anytype
            ) !Tensor(ValueType, contractedRank(expression), @TypeOf(x.*).Order) {

            if(!x.isValid() or !y.isValid()) {
                return TensorError.InvalidTensorLayout;
            }
            const XRank = @TypeOf(x.*).Rank;
            const YRank = @TypeOf(y.*).Rank;
            const ZRank = comptime contractedRank(expression);
            const plan = comptime innerProductParse(XRank, YRank, ZRank, expression);

            var z_sizes: [ZRank]SizesType = undefined;
            {
                var i: usize = 0;
                while(i < plan.total) : (i += 1) {
                    if(plan.z_perm[i] == plan.pass) {
                        continue;
                    } else if(plan.s_ctrl[i] == 0){
                        z_sizes[plan.z_perm[i]] = x.getSize(plan.x_perm[i]);
                    } else {
                        z_sizes[plan.z_perm[i]] = y.getSize(plan.y_perm[i]);
                    }
                }
            }

            var z = try self.allocTensor(ZRank, @TypeOf(x.*).Order, z_sizes);
            var x_i: [XRank]SizesType = undefined;
            var y_i: [YRank]SizesType = undefined;
            var z_i: [ZRank]SizesType = undefined;
            
            @memset(z.values, 0);
            
            @call(.always_inline, Ops.recursiveInnerProduct, .{
                ValueType, SizesType, 0, plan, x, y, &z, &x_i, &y_i, &z_i
            });
            return z;
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

test "vectorized reduce" {

    var factory = TensorFactory(i32).init(null);

    factory.tracking(.start);
    
    var x = try factory.allocTensor(2, Rowwise, .{ 100, 100 });
    
    @memset(x.values, 1);

    { // reduce sum of 10'000 elements
        const y = try Ops.sum(&x);
        try std.testing.expectEqual(y, 10000);
    }
    { // reduce product of 10'000 elements
        const y = try Ops.product(&x);
        try std.testing.expectEqual(y, 1);
    }
    { // reduce max of 10'000 elements
        x.setValue(999, .{24, 62});
        const y = try Ops.max(&x);
        try std.testing.expectEqual(y, 999);
    }
    { // reduce max of 10'000 elements
        x.setValue(-999, .{92, 10});
        const y = try Ops.min(&x);
        try std.testing.expectEqual(y, -999);
    }
    factory.deinit();
}

test "contraction" {

    var factory = TensorFactory(i32).init(null);

    factory.tracking(.start);

    var x = try factory.allocTensor(3, Rowwise, .{ 3, 4, 3 });

    @memset(x.values, 1);

    var y = try factory.contraction("ijk->i", &x);

    try std.testing.expectEqual(y.values[0], 12);
    try std.testing.expectEqual(y.values[1], 12);
    try std.testing.expectEqual(y.values[2], 12);

    var z = try factory.contraction("ijk->j", &x);

    try std.testing.expectEqual(z.values[0], 9);
    try std.testing.expectEqual(z.values[1], 9);
    try std.testing.expectEqual(z.values[2], 9);
    try std.testing.expectEqual(z.values[3], 9);

    factory.deinit();
}


test "contraction 2" {
    var factory = TensorFactory(i32).init(null);

    defer factory.deinit();

    factory.tracking(.start);

    var x = try factory.allocTensor(3, Rowwise, .{ 3, 4, 3 });
    var y = try factory.allocTensor(2, Rowwise, .{ 3, 4 });
    var z = try factory.allocTensor(2, Rowwise, .{ 4, 3 });

    Ops.fill(&x, 1, 1);

    try Ops.contraction("ijk->ij", &x, &y);

    try std.testing.expectEqual(y.values[0], 6);
    try std.testing.expectEqual(y.values[1], 15);
    try std.testing.expectEqual(y.values[2], 24);
    try std.testing.expectEqual(y.values[3], 33);
    try std.testing.expectEqual(y.values[4], 42);
    try std.testing.expectEqual(y.values[5], 51);
    try std.testing.expectEqual(y.values[6], 60);
    try std.testing.expectEqual(y.values[7], 69);
    try std.testing.expectEqual(y.values[8], 78);
    try std.testing.expectEqual(y.values[9], 87);
    try std.testing.expectEqual(y.values[10], 96);
    try std.testing.expectEqual(y.values[11], 105);

    try Ops.contraction("ijk->ji", &x, &z);

    try std.testing.expectEqual(z.values[0], 6);
    try std.testing.expectEqual(z.values[1], 42);
    try std.testing.expectEqual(z.values[2], 78);
    try std.testing.expectEqual(z.values[3], 15);
    try std.testing.expectEqual(z.values[4], 51);
    try std.testing.expectEqual(z.values[5], 87);
    try std.testing.expectEqual(z.values[6], 24);
    try std.testing.expectEqual(z.values[7], 60);
    try std.testing.expectEqual(z.values[8], 96);
    try std.testing.expectEqual(z.values[9], 33);
    try std.testing.expectEqual(z.values[10], 69);
    try std.testing.expectEqual(z.values[11], 105);

    try Ops.contraction("ijk->jk", &x, &z);

    try std.testing.expectEqual(z.values[0], 39);
    try std.testing.expectEqual(z.values[1], 42);
    try std.testing.expectEqual(z.values[2], 45);
    try std.testing.expectEqual(z.values[3], 48);
    try std.testing.expectEqual(z.values[4], 51);
    try std.testing.expectEqual(z.values[5], 54);
    try std.testing.expectEqual(z.values[6], 57);
    try std.testing.expectEqual(z.values[7], 60);
    try std.testing.expectEqual(z.values[8], 63);
    try std.testing.expectEqual(z.values[9], 66);
    try std.testing.expectEqual(z.values[10], 69);
    try std.testing.expectEqual(z.values[11], 72);
}

test "inner product 1" {

    var factory = TensorFactory(i32).init(null);

    factory.tracking(.start);

    defer factory.deinit();

    var x = try factory.allocTensor(2, Rowwise, .{ 2, 2 });
    var y = try factory.allocTensor(2, Rowwise, .{ 2, 2 });
    var z = try factory.allocTensor(2, Rowwise, .{ 2, 2 });

    Ops.fill(&x, 1, 0);
    Ops.fill(&y, 1, 1);

    try Ops.innerProduct("ij,jk->ik", &x, &y, &z);

    try std.testing.expectEqual(z.values[0], 4);
    try std.testing.expectEqual(z.values[1], 6);
    try std.testing.expectEqual(z.values[2], 4);
    try std.testing.expectEqual(z.values[3], 6);

    try Ops.innerProduct("ij,jk->ki", &x, &y, &z);

    try std.testing.expectEqual(z.values[0], 4);
    try std.testing.expectEqual(z.values[1], 4);
    try std.testing.expectEqual(z.values[2], 6);
    try std.testing.expectEqual(z.values[3], 6);
}

test "inner product 2" {

    var factory = TensorFactory(i32).init(null);

    factory.tracking(.start);

    defer factory.deinit();

    var x = try factory.allocTensor(3, Rowwise, .{ 2, 3, 2 });
    var y = try factory.allocTensor(3, Rowwise, .{ 2, 3, 2 });

    Ops.fill(&x, 0, 1);
    Ops.fill(&y, 0, 1);

    var z = try factory.innerProduct("ijk,kjm->im", &x, &y);

    try std.testing.expectEqual(z.values[0], 100);
    try std.testing.expectEqual(z.values[1], 115);
    try std.testing.expectEqual(z.values[2], 280);
    try std.testing.expectEqual(z.values[3], 331);

    var w = try factory.innerProduct("ikj,jkl->kl", &x, &y);

    try std.testing.expectEqual(w.values[0],  48);
    try std.testing.expectEqual(w.values[1],  62);
    try std.testing.expectEqual(w.values[2], 116);
    try std.testing.expectEqual(w.values[3], 138);
    try std.testing.expectEqual(w.values[4], 216);
    try std.testing.expectEqual(w.values[5], 246);
}

test "arithmetic 1" {

    var factory = TensorFactory(i64).init(null);

    factory.tracking(.start);

    defer factory.deinit();

    var x = try factory.allocTensor(1, Rowwise, .{ 100_000 });
    var y = try factory.allocTensor(1, Rowwise, .{ 100_000 });
    Ops.fill(&x, 1, 0);
    Ops.fill(&y, 2, 0);

    // factory versions...
    {
        var z = try factory.add(&x, &y);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 300_000);
    }
    {
        var z = try factory.mul(&x, &y);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 200_000);
    }
    {
        var z = try factory.sub(&x, &y);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == -100_000);
    }
    {
        var b: i64 = 4;
        var z = try factory.bias(&x, b);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 500_000);
    }
    {
        var b: i64 = 4;
        var z = try factory.scale(&x, b);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 400_000);
    }

    var z = try factory.allocTensor(1, Rowwise, .{ 100_000 });

    // free versions...
    {
        try Ops.add(&x, &y, &z);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 300_000);
    }
    {
        try Ops.mul(&x, &y, &z);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 200_000);
    }
    {
        try Ops.sub(&x, &y, &z);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == -100_000);
    }
    {
        var b: i64 = 4;
        try Ops.bias(&x, &z, b);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 500_000);
    }
    {
        var b: i64 = 4;
        try Ops.scale(&x, &z, b);
        const s = try Ops.sum(&z);
        try std.testing.expect(s == 400_000);
    }
}
