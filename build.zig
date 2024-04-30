const std = @import("std");

const Mode = enum {
    gui,
    solve,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Solve

    const solve_exe = b.addExecutable(.{
        .name = "kakuro",
        .root_source_file = .{ .path = "kakuro.zig" },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const solve_opts = b.addOptions();
    solve_exe.root_module.addOptions("build_options", solve_opts);
    solve_opts.addOption(Mode, "mode", .solve);

    b.installArtifact(solve_exe);
    const solve_cmd = b.addRunArtifact(solve_exe);
    solve_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        solve_cmd.addArgs(args);
    }
    const solve_step = b.step("solve", "Run TUI solver");
    solve_step.dependOn(&solve_cmd.step);

    // GUI

    const gui_exe = b.addExecutable(.{
        .name = "kakuro-gui",
        .root_source_file = .{ .path = "kakuro.zig" },
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const gui_opts = b.addOptions();
    gui_exe.root_module.addOptions("build_options", gui_opts);
    gui_opts.addOption(Mode, "mode", .gui);

    const raylib_zig = b.dependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });
    const raylib = raylib_zig.module("raylib");
    const raylib_math = raylib_zig.module("raylib-math");
    const raylib_artifact = raylib_zig.artifact("raylib");
    gui_exe.linkLibrary(raylib_artifact);
    gui_exe.root_module.addImport("raylib", raylib);
    gui_exe.root_module.addImport("raylib-math", raylib_math);

    b.installArtifact(gui_exe);
    const gui_cmd = b.addRunArtifact(gui_exe);
    const gui_step = b.step("gui", "Run TUI gui");
    gui_step.dependOn(&gui_cmd.step);
    // gui_cmd.step.dependOn(b.getInstallStep());
}
