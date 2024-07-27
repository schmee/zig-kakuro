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
        .root_source_file = b.path("src/kakuro.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    const solve_opts = b.addOptions();
    solve_exe.root_module.addOptions("build_options", solve_opts);
    solve_opts.addOption(Mode, "mode", .solve);

    const solve_cmd = b.addRunArtifact(solve_exe);
    const solve_install = b.addInstallArtifact(solve_exe, .{});
    solve_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        solve_cmd.addArgs(args);
    }
    const solve_step = b.step("solve", "Run TUI solver");
    solve_step.dependOn(&solve_cmd.step);
    solve_step.dependOn(&solve_install.step);


    // GUI
    const gui_exe = b.addExecutable(.{
        .name = "kakuro-gui",
        .root_source_file = b.path("src/kakuro.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
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
    const gui_cmd = b.addRunArtifact(gui_exe);
    const gui_install = b.addInstallArtifact(gui_exe, .{});
    const gui_step = b.step("gui", "Run TUI gui");
    gui_step.dependOn(&gui_cmd.step);
    gui_step.dependOn(&gui_install.step);
}
