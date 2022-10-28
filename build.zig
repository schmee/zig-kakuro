const std = @import("std");
const raylib = @import("raylib-zig/lib.zig"); //call .Pkg() with the folder raylib-zig is in relative to project build.zig

const Mode = enum {
    gui,
    solve,
};

pub fn build(b: *std.build.Builder) void {
    const target = b.standardTargetOptions(.{});
    const mode = b.standardReleaseOptions();

    // Solve

    const solve_exe = b.addExecutable("kakuro", "kakuro.zig");
    const solve_opts = b.addOptions();
    solve_exe.addOptions("build_options", solve_opts);
    solve_opts.addOption(Mode, "mode", .solve);

    solve_exe.setTarget(target);
    solve_exe.setBuildMode(mode);
    solve_exe.linkLibC();

    const solve_cmd = solve_exe.run();
    solve_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        solve_cmd.addArgs(args);
    }
    const solve_step = b.step("solve", "Run TUI solver");
    solve_step.dependOn(&solve_cmd.step);


    // GUI

    const gui_exe = b.addExecutable("kakuro-gui", "kakuro.zig");
    gui_exe.setTarget(target);
    gui_exe.setBuildMode(mode);
    gui_exe.linkLibC();

    raylib.link(gui_exe, false);
    raylib.addAsPackage("raylib", gui_exe);
    raylib.math.addAsPackage("raylib-math", gui_exe);

    const gui_opts = b.addOptions();
    gui_exe.addOptions("build_options", gui_opts);
    gui_opts.addOption(Mode, "mode", .gui);

    const gui_cmd = gui_exe.run();
    gui_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        gui_cmd.addArgs(args);
    }
    const gui_step = b.step("gui", "Run the GUI");
    gui_step.dependOn(&gui_cmd.step);
}
