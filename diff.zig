const std = @import("std");

const Entity = struct {
    line1: Line,
    line2: Line,
    diff: Diff,
};

pub fn main() !void {
    var allocator = std.heap.c_allocator;

    const argv = std.os.argv;
    if (argv.len != 3) {
        return error.IllegalInput;
    }

    const f1_path = std.mem.span(argv[1]);
    const f2_path = std.mem.span(argv[2]);
    const f1 = try std.fs.cwd().openFile(f1_path, .{ .read = true });
    defer f1.close();
    const f2 = try std.fs.cwd().openFile(f2_path, .{ .read = true });
    defer f2.close();


    const f1_content = blk: {
        const stat = try f1.stat();
        var buf = try allocator.alloc(u8, stat.size);
        _ = try f1.read(buf);
        break :blk buf;
    };
    var f1_lines = std.mem.split(u8, f1_content, "\n");
    _ = f1_lines.next(); // skip header

    const f2_content = blk: {
        const stat = try f2.stat();
        var buf = try allocator.alloc(u8, stat.size);
        _ = try f2.read(buf);
        break :blk buf;
    };
    var f2_lines = std.mem.split(u8, f2_content, "\n");
    _ = f2_lines.next(); // skip header

    var w = std.io.getStdOut().writer();

    var max_iters_diff: isize = 0;
    var max_iters_diff_index: isize = -1;
    var max_time_ns_diff: isize = 0;
    var max_time_ns_diff_index: isize = -1;

    var entities = std.ArrayList(Entity).init(allocator);
    while (f1_lines.next()) |f1_line| {
        if (std.mem.eql(u8, f1_line, "")) break;
        const parsed1 = parseLine(f1_line);
        const f2_line = f2_lines.next().?;
        const parsed2 = parseLine(f2_line);
        const diff = diffLines(parsed1, parsed2);

        const entity = Entity{
            .line1 = parsed1,
            .line2 = parsed2,
            .diff = diff,
        };
        try entities.append(entity);

        try printDiff(allocator, w, entity);

        const index = parsed1.index;

        const iters_diff_abs = try std.math.absInt(diff.iters_abs);
        if (iters_diff_abs > max_iters_diff) {
            max_iters_diff = iters_diff_abs;
            max_iters_diff_index = index;
        }

        const time_ns_diff_abs = try std.math.absInt(diff.time_ns_abs);
        if (time_ns_diff_abs > max_time_ns_diff) {
            max_time_ns_diff = time_ns_diff_abs;
            max_time_ns_diff_index = index;
        }

    }

    try w.writeAll("\n======================================================\n\n");

    try w.print("max iters diff\n", .{});
    if (max_iters_diff_index != -1) {
        try printDiff(allocator, w, entities.items[@intCast(usize, max_iters_diff_index - 1)]);
    } else {
        try w.print("no iters diffs\n", .{});
    }

    try w.print("max time_ns diff\n", .{});
    if (max_time_ns_diff_index != -1) {
        try printDiff(allocator, w, entities.items[@intCast(usize, max_time_ns_diff_index - 1)]);
    } else {
        try w.print("no time diffs\n", .{});
    }

    try w.writeAll("\n======================================================\n\n");

    var iters_unchanged: usize = 0;
    var iters_improvements: usize = 0;
    var iters_regressions: usize = 0;
    for (entities.items) |entity| {
        if (entity.diff.iters_rel < 100) {
            iters_improvements += 1;
        } else if (entity.diff.iters_rel > 100) {
            iters_regressions += 1;
        } else {
            iters_unchanged += 1;
        }
    }
    const iters_percent_unchanged = @intToFloat(f32, iters_unchanged) / @intToFloat(f32, entities.items.len) * 100;
    const iters_percent_improvements = @intToFloat(f32, iters_improvements) / @intToFloat(f32, entities.items.len) * 100;
    const iters_percent_regressions = @intToFloat(f32, iters_regressions) / @intToFloat(f32, entities.items.len) * 100;
    try w.print("iters: unchanged {d} improvements {d} regressions {d} => unchanged {d:.2}% improvements {d:.2}% regressions {d:.2}%\n", .{
        iters_unchanged, iters_improvements, iters_regressions, iters_percent_unchanged, iters_percent_improvements, iters_percent_regressions
    });

    for (entities.items) |entity| {
        if (entity.diff.iters_rel < 100) {
            try w.writeAll(ANSI_GREEN);
        } else if (entity.diff.iters_rel > 100) {
            try w.writeAll(ANSI_RED);
        }
        try w.print("{d} ", .{entity.line1.index});
        try w.writeAll(ANSI_RESET);
    }
    try w.writeAll("\n");

    var time_ns_unchanged: usize = 0;
    var time_ns_improvements: usize = 0;
    var time_ns_regressions: usize = 0;
    for (entities.items) |entity| {
        if (entity.diff.time_ns_rel < 100) {
            time_ns_improvements += 1;
        } else if (entity.diff.time_ns_rel > 100) {
            time_ns_regressions += 1;
        } else {
            time_ns_unchanged += 1;
        }
    }
    const time_ns_percent_unchanged = @intToFloat(f32, time_ns_unchanged) / @intToFloat(f32, entities.items.len) * 100;
    const time_ns_percent_improvements = @intToFloat(f32, time_ns_improvements) / @intToFloat(f32, entities.items.len) * 100;
    const time_ns_percent_regressions = @intToFloat(f32, time_ns_regressions) / @intToFloat(f32, entities.items.len) * 100;
    try w.print("time_ns: unchanged {d} improvements {d} regressions {d} => unchanged {d:.2}% improvements {d:.2}% regressions {d:.2}%\n", .{
        time_ns_unchanged, time_ns_improvements, time_ns_regressions, time_ns_percent_unchanged, time_ns_percent_improvements, time_ns_percent_regressions
    });

    for (entities.items) |entity| {
        const time_ns_abs = try std.math.absInt(entity.diff.time_ns_abs);
        // Ignore microsecond differences to denoise the diff
        const should_diff = time_ns_abs > 20_000;
        if (should_diff and entity.diff.time_ns_rel < 100) {
            try w.writeAll(ANSI_GREEN);
        } else if (should_diff and entity.diff.time_ns_rel > 100) {
            try w.writeAll(ANSI_RED);
        }
        try w.print("{d} ", .{entity.line1.index});
        try w.writeAll(ANSI_RESET);
    }
    try w.writeAll("\n");
}

const Diff = struct {
    iters_abs: isize,
    iters_rel: f32,
    time_ns_abs: isize,
    time_ns_rel: f32,
};

fn diffLines(parsed1: Line, parsed2: Line) Diff {
    const iters_diff = parsed2.iters - parsed1.iters;
    const iters_relative = 100 * @intToFloat(f32, parsed2.iters) / @intToFloat(f32, parsed1.iters);

    const time_ns_diff = parsed2.time_ns - parsed1.time_ns;
    const time_ns_relative = 100 * @intToFloat(f32, parsed2.time_ns) / @intToFloat(f32, parsed1.time_ns);

    return .{
        .iters_abs = iters_diff,
        .iters_rel = iters_relative,
        .time_ns_abs = time_ns_diff,
        .time_ns_rel = time_ns_relative,
    };
}

fn printDiff(allocator: std.mem.Allocator, w: anytype, entity: Entity) !void {
    const parsed1 = entity.line1;
    const parsed2 = entity.line2;
    const diff = entity.diff;

    try w.print("index {d} ", .{parsed1.index});

    try w.print("iters ", .{});
    try w.print("{s:>10}", .{convertIters(allocator, parsed1.iters)});
    try w.print("{s:>10}", .{convertIters(allocator, parsed2.iters)});
    if (diff.iters_abs < 0) {
        try w.writeAll(ANSI_GREEN);
    } else if (diff.iters_abs > 0) {
        try w.writeAll(ANSI_RED);
    }
    try w.print("{d:>10}", .{diff.iters_abs});
    try w.print("{s:>5}", .{""});
    try w.print("{d:>7.2}%", .{diff.iters_rel});
    try w.writeAll(ANSI_RESET);

    try w.print(" time ms", .{});
    try w.print("{s:>10}", .{convertTimeNs(allocator, parsed1.time_ns)});
    try w.print("{s:>10}", .{convertTimeNs(allocator, parsed2.time_ns)});
    const time_ns_abs = try std.math.absInt(diff.time_ns_abs);
    const should_diff = time_ns_abs > 20_000;
    if (should_diff and diff.time_ns_abs < 0) {
        try w.writeAll(ANSI_GREEN);
    } else if (should_diff and diff.time_ns_abs > 0) {
       try w.writeAll(ANSI_RED);
    }
    try w.print("    ", .{});
    try w.print("{d:<6.2}%   ", .{diff.time_ns_rel});
    try w.print("{s:<5}   ", .{std.fmt.fmtDurationSigned(diff.time_ns_abs)});
    try w.writeAll(ANSI_RESET);
    try w.print("\n", .{});
}

fn convertIters(allocator: std.mem.Allocator, iters: isize) []const u8 {
    return if (iters == -1)
        "-"
    else
        std.fmt.allocPrint(allocator, "{d}", .{@intCast(usize, iters)}) catch unreachable;
}

fn convertTimeNs(allocator: std.mem.Allocator, time_ns: isize) []const u8 {
    return if (time_ns == -1)
        "-"
    else
        std.fmt.allocPrint(allocator, "{}", .{std.fmt.fmtDurationSigned(time_ns)}) catch unreachable;
}


const esc = "\x1B";
const csi = esc ++ "[";
const ANSI_RED = csi ++ "31m";
const ANSI_GREEN = csi ++ "32m";
const ANSI_RESET = csi ++ "0m";

const Line = struct {
    index: isize = 0,
    iters: isize = 0,
    time_ns: isize = 0,
};

fn parseLine(line: []const u8) Line {
    var values = std.mem.split(u8, line, ",");
    return .{
        .index = std.fmt.parseInt(isize, values.next().?, 10) catch unreachable,
        .iters = std.fmt.parseInt(isize, values.next().?, 10) catch unreachable,
        .time_ns = std.fmt.parseInt(isize, values.next().?, 10) catch unreachable,
    };
}
