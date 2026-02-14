const std = @import("std");

const Mode = enum {
    gui,
    solve,
};

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const bounded_array = b.dependency("bounded_array", .{ .target = target, .optimize = optimize });

    // Solve
    const solve_mod = b.addModule("kakuro", .{
        .root_source_file = b.path("src/kakuro.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    solve_mod.addImport("bounded_array", bounded_array.module("bounded_array"));
    const solve_exe = b.addExecutable(.{
        .name = "kakuro",
        .root_module = solve_mod,
    });
    const solve_opts = b.addOptions();
    solve_exe.root_module.addOptions("build_options", solve_opts);
    solve_opts.addOption(Mode, "mode", .solve);

    const solve_run = b.addRunArtifact(solve_exe);
    const solve_install = b.addInstallArtifact(solve_exe, .{});
    solve_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        solve_run.addArgs(args);
    }
    const solve_step = b.step("solve", "Run TUI solver");
    solve_step.dependOn(&solve_run.step);
    solve_step.dependOn(&solve_install.step);


    // GUI
    const gui_mod = b.addModule("kakuro-gui", .{
        .root_source_file = b.path("src/kakuro.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    gui_mod.addImport("bounded_array", bounded_array.module("bounded_array"));
    const gui_exe = b.addExecutable(.{
        .name = "kakuro-gui",
        .root_module = gui_mod,
    });

    const gui_opts = b.addOptions();
    gui_exe.root_module.addOptions("build_options", gui_opts);
    gui_opts.addOption(Mode, "mode", .gui);

    const raylib_zig = b.lazyDependency("raylib_zig", .{
        .target = target,
        .optimize = optimize,
    });
    if (raylib_zig) |rlz| {
        const raylib = rlz.module("raylib");
        const raylib_artifact = rlz.artifact("raylib");
        gui_exe.linkLibrary(raylib_artifact);
        gui_exe.root_module.addImport("raylib", raylib);
    }

    // b.installArtifact(gui_exe);
    const gui_run = b.addRunArtifact(gui_exe);
    const gui_install = b.addInstallArtifact(gui_exe, .{});
    const gui_step = b.step("gui", "Run TUI gui");
    gui_step.dependOn(&gui_run.step);
    gui_step.dependOn(&gui_install.step);

    // Diff
    const diff_mod = b.addModule("diff", .{
        .root_source_file = b.path("src/diff.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const diff_exe = b.addExecutable(.{
        .name = "diff",
        .root_module = diff_mod,
    });
    const diff_step = b.step("diff", "Compare two reports generated from `solve --report`");
    const diff_run = b.addRunArtifact(diff_exe);
    const diff_install = b.addInstallArtifact(diff_exe, .{});
    if (b.args) |args| {
        diff_run.addArgs(args);
    }
    diff_step.dependOn(&diff_run.step);
    diff_step.dependOn(&diff_install.step);
}
