

const TensorError = @import("Tensor.zig").TensorError;
const TensorAllocator = @import("TensorAllocator.zig").TensorAllocator;
const Rowwise = @import("SizesAndStrides.zig").Rowwise;
const Colwise = @import("SizesAndStrides.zig").Colwise;
const SizeAndStrideType = @import("SizesAndStrides.zig").SizeAndStride.ValueType;
const defaultPermuation = @import("SizesAndStrides.zig").defaultPermutation;

const ReduceOp = @import("std").builtin.ReduceOp;
const mem = @import("std").mem;



// The OpsPolicy controls the behavior of the math
// class. This includes behavior such as whether or
// not the class can allocate new tensors if output
// tensors are are not provided

pub const OpsPolicy = struct {
    // Flag to allocate more scratch memory.
    // Some ops work best with scratch memory.
    alloc_scratch: bool = true,

    // Flag to check arguments for validity.
    validate_args: bool = true,
};

pub fn TensorOps(comptime alloc_type: type, comptime policy: OpsPolicy) type {

    return struct {

        const Self = @This();

        const SelfPtr = *Self;

        const ConstSelfPtr = *const Self;

        const SizesType = SizeAndStrideType;

        const AllocType = alloc_type;

        const Policy = policy;

        // The allocator data member is here incase
        // a user does not provide enough memory
        allocator: TensorAllocator(AllocType),

        // Scratch memory for operations
        scratch: []AllocType = &[_]AllocType{ },

        pub fn init(allocator: TensorAllocator(AllocType)) Self {
            return Self { .allocator = allocator };
        }

        pub fn scratchSize(self: ConstSelfPtr) usize {
            return self.*.scratch.len;
        }

        pub fn releaseScratch(self: SelfPtr) []AllocType {
            var tmp = self.*.scratch;
            self.*.scratch = &[_]AllocType{};
            return tmp;
        }        

        pub fn resizeScratch(self: SelfPtr, size: usize) !void {
            if(self.*.scratch.len == size) {
                return;
            }
            if(self.*.scratchSize() != 0) {
                self.*.allocator.freeValues(self.*.releaseScratch());
            }
            self.*.scratch = try self.*.allocator.allocValues(size);
        }

        //pub fn add(self: SelfPtr, X: anytype, Y: anytype, Z: anytype) !void {
        //    _ = self;
        //    const XT = @TypeOf(X.*);
        //    const YT = @TypeOf(Y.*);

        //    if(XT != YT) {
        //        @compileError("Cannot add tensors of different types.");
        //    }
        //    if(XT.ValueType != AllocType) {
        //        @compileError("Cannot add tensors of different value types.");
        //    }
        //    if(Policy.validate_args) {
        //        try expect(X.*.isValid() and Y.*.isValid() and Z.*.isValid());
        //        try expect(X.*.valueSize() == Y.*.valueSize());
        //        try expect(X.*.valueSize() == Z.*.valueSize());
        //    }            
        //    @compileError("Needs Implementation");
        //}

        //pub fn multiply(self: SelfPtr, X: anytype, Y: anytype, Z: anytype) !void {
        //    _ = self;
        //    const XT = @TypeOf(X.*);
        //    const YT = @TypeOf(Y.*);

        //    if(XT != YT) {
        //        @compileError("Cannot multiply tensors of different types.");
        //    }
        //    if(XT.ValueType != AllocType) {
        //        @compileError("Cannot multiply tensors of different value types.");
        //    }
        //    if(Policy.validate_args) {
        //        try expect(X.*.isValid() and Y.*.isValid() and Z.*.isValid());
        //        try expect(X.*.valueSize() == Y.*.valueSize());
        //        try expect(X.*.valueSize() == Z.*.valueSize());
        //    }
        //    @compileError("Needs Implementation");
        //}

        pub fn permutateValues(self: SelfPtr, x: anytype, permutation: [@TypeOf(x.*).Rank]SizesType) !void {

            const XT = @TypeOf(x.*);

            if(Policy.validate_args) {
                if (!x.*.isValid()) { 
                    return TensorError.InvalidTensorLayout; 
                }
                try x.permutate(permutation);
            }
            else {
                x.permutateUnchecked(permutation);
            }
            
            if(Policy.alloc_scratch) {

                if(self.*.scratchSize() < x.valueSize()){
                    try self.*.resizeScratch(x.*.valueSize());
                }

                var c: [XT.Rank]SizesType = undefined;
                var n: XT.SizesType = 0;

                @call(.always_inline, recursivePermutateValues, .{
                     XT.ValueType, SizesType, XT.Rank, 0, x, self.*.scratch, &c, &n
                });

                @memcpy(x.*.values, self.*.scratch[0..x.*.valueSize()]);
            }
            else {
                @compileError("Non-scratch memory version of permutateValues is not implemented.");
            } 
        }
    };
}

/////////////////////////////////////////////////////////////
// This is the naive version of a general tensor permutation.
// In the future, I plan on making more optimal versions of
// this, but it's reliable and decent enough to get started.

inline fn recursivePermutateValues(
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
        var i: IT = 0;
        var n_i = n.*;
        while(i < x.*.getSize(I)) : ({ i += 1; n_i += 1; }) {

            c[I] = i;
    
            const x_i = x_blk: {
                const x_c1 : @Vector(R, IT) = c.*;
                const x_ss : @Vector(R, IT) = x.*.sizes_and_strides.strides;
                break :x_blk @reduce(ReduceOp.Add, x_c1 * x_ss);
            }; 

            y[n_i] = x.*.values[x_i];
        }
        n.* += i;
    }

    else {
        var i: IT = 0;
        while(i < x.*.getSize(I)) : (i += 1) {
            
            c[I] = i; 
            
            @call(.always_inline, recursivePermutateValues, .{
                 VT, IT, R, (I + 1), x, y, c, n 
            });
        }
    }
}