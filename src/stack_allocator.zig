///////////////////////////////////////////////////////////////
//// Motivation and Explanation for StackAllocator ////////////

//    The StackAllocator has a congtiguous memory buffer (size in bytes)
//    that it attempts to utilize before deferring to it's backing_allocator.
//
//    It can only roll-back the used capacity if what is being freed was the
//    last thing to be allocated (like a typical stack, first-in-last-out).
//
//    To free all of the memory from the stack, deallocate the items in
//    reverse order to what they were allocated in.
//
//    Resize will only work if you are attempting to resize the last
//    allocated item (item on top of the stack).
//
//    If you overflow the stack, the StackAllocator will defer to using
//    its backing_allocator.
//

const std = @import("std");

pub fn StackBuffer(comptime size: usize) type {
    return struct {
        const Self = @This();
        const Size = size;

        items: [Size]u8 = undefined,
        used: usize = 0,

        pub fn withdraw(self: *Self, n: usize) ?[]u8 {
            if ((n + self.used) <= self.items.len) {
                const data = self.items[self.used .. self.used + n];
                self.used += n;
                return data;
            }
            return null;
        }

        pub inline fn owns(self: *const Self, data: []u8) bool {
            const lhs = @intFromPtr(&self.items[0]);
            const rhs = @intFromPtr(&self.items[self.items.len - 1]);
            const ptr = @intFromPtr(data.ptr);
            return (lhs <= ptr) and (ptr <= rhs);
        }

        pub inline fn isTop(self: *const Self, data: []u8) bool {
            // can only pop values off the top of the stack
            if (self.used < data.len) {
                return false;
            }
            // check to see if we can back up the values
            return (@intFromPtr(&self.items[self.used - data.len]) == @intFromPtr(data.ptr));
        }

        pub fn canResize(self: *const Self, data: []u8, n: usize) bool {
            // can only resize values at the top of the stack
            if (!self.isTop(data)) {
                return false;
            }
            const old_used = self.used - data.len;
            const new_used = old_used + n;
            return new_used <= Size;
        }

        pub fn deposit(self: *Self, data: []u8) bool {
            if (!self.owns(data)) {
                return false;
            }
            // check to see if we can back up the values
            if (self.isTop(data)) {
                self.used -= data.len;
            }
            return true;
        }
    };
}

////////////////////////////////////////////////////////
//////// StackAllocator Implementation /////////////////

pub fn StackAllocator(comptime size: usize) type {
    return struct {
        const Self = @This();
        const Size = size;

        stack_buffer: StackBuffer(Size),
        backing_allocator: std.mem.Allocator,

        // TODO: Create a dummy mutex that can be swapped via policy
        mutex: std.Thread.Mutex = std.Thread.Mutex{},

        pub fn init(backing_allocator: std.mem.Allocator) Self {
            return Self{
                .backing_allocator = backing_allocator,
                .stack_buffer = .{},
            };
        }

        pub fn allocator(self: *Self) std.mem.Allocator {
            return .{
                .ptr = self,
                .vtable = &.{
                    .alloc = alloc,
                    .resize = resize,
                    .free = free,
                },
            };
        }

        pub fn alloc(ctx: *anyopaque, len: usize, log2_ptr_align: u8, ret_addr: usize) ?[*]u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));

            self.mutex.lock();

            defer self.mutex.unlock();

            if (self.stack_buffer.withdraw(len)) |data| {
                return data.ptr;
            }
            return self.backing_allocator.rawAlloc(len, log2_ptr_align, ret_addr);
        }

        pub fn resize(
            ctx: *anyopaque,
            old_mem: []u8,
            log2_align: u8,
            new_len: usize,
            ret_addr: usize,
        ) bool {
            const self: *Self = @ptrCast(@alignCast(ctx));

            self.mutex.lock();

            defer self.mutex.unlock();

            if (!self.stack_buffer.owns(old_mem)) {
                return self.backing_allocator.rawResize(old_mem, log2_align, new_len, ret_addr);
            }
            return self.stack_buffer.canResize(old_mem, new_len);
        }

        pub fn free(
            ctx: *anyopaque,
            old_mem: []u8,
            log2_align: u8,
            ret_addr: usize,
        ) void {
            const self: *Self = @ptrCast(@alignCast(ctx));

            self.mutex.lock();

            defer self.mutex.unlock();

            // if we do not own the memory, we'll try
            // to free it using the backing allocator
            if (!self.stack_buffer.deposit(old_mem)) {
                self.backing_allocator.rawFree(old_mem, log2_align, ret_addr);
            }
        }
    };
}

/////////////////////////////////////////////////////////
/////// StackAllocator Testing Section //////////////////

test "basic stack properties" {
    var GPA = std.heap.GeneralPurposeAllocator(.{}){};
    var stack_allocator = StackAllocator(100).init(GPA.allocator());
    var allocator = stack_allocator.allocator();

    { // reverse-order stack popping
        const a = try allocator.alloc(u8, 10);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 10);
        const b = try allocator.alloc(u8, 10);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 20);

        allocator.free(b);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 10);
        allocator.free(a);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 0);
    }

    { // unordered stack popping
        const a = try allocator.alloc(u8, 10);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 10);
        const b = try allocator.alloc(u8, 10);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 20);

        allocator.free(a);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 20);
        allocator.free(b);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 10);
    }

    if (GPA.deinit() == .leak) @panic("MEMORY LEAK DETECTED!!");
}

test "basic stack resize" {
    var GPA = std.heap.GeneralPurposeAllocator(.{}){};
    var stack_allocator = StackAllocator(100).init(GPA.allocator());
    var allocator = stack_allocator.allocator();

    { // resize checking
        const a = try allocator.alloc(u8, 10);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 10);
        const b = try allocator.alloc(u8, 10);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 20);

        // a cannot resize because it is not on the top of the stack
        try std.testing.expect(!allocator.resize(a, 20));

        // b can resize because it is on the top of the stack
        try std.testing.expect(allocator.resize(b, 20));

        // b should be able to take the remaining memory
        try std.testing.expect(allocator.resize(b, 90));

        // b should not be able to take more than remainder
        try std.testing.expect(!allocator.resize(b, 91));
    }

    if (GPA.deinit() == .leak) @panic("MEMORY LEAK DETECTED!!");
}

test "stack-overflow allocation" {
    var GPA = std.heap.GeneralPurposeAllocator(.{}){};
    var stack_allocator = StackAllocator(100).init(GPA.allocator());
    var allocator = stack_allocator.allocator();

    { // overflow the full memory stack
        const a = try allocator.alloc(u8, 100);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 100);

        const b = try allocator.alloc(u8, 100);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 100);

        allocator.free(a);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 0);
        allocator.free(b);
        try std.testing.expectEqual(stack_allocator.stack_buffer.used, 0);
    }

    if (GPA.deinit() == .leak) @panic("MEMORY LEAK DETECTED!!");
}
