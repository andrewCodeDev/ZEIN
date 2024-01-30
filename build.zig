const std = @import("std");

pub fn build(b: *std.Build) void {
    // parse target and optimization options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // define primary (static library) artifact
    const lib = b.addStaticLibrary(.{
        .name = "ZEIN",
        .root_source_file = .{ .path = "src/root.zig" },
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // create secondary (unit testing) artifact
    const unit_tests = b.addTest(.{
        .root_source_file = .{ .path = "src/root.zig" },
        .target = target,
        .optimize = optimize,
    });
    const run_tests = b.addRunArtifact(unit_tests);

    // define test step dependency
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
