const std = @import("std");
const rl = @import("raylib");
const build_options = @import("build_options");

const assert = debug.assert;
const debug = std.debug;
const fmt = std.fmt;
const fs = std.fs;
const heap = std.heap;
const io = std.io;
const log = std.log;
const math = std.math;
const mem = std.mem;
const rand = std.rand;
const testing = std.testing;
const time = std.time;

const Allocator = mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;
const ArrayList = std.ArrayList;
const Atomic = std.atomic.Atomic;
const AutoHashMap = std.AutoHashMap;
const StaticBitSet = std.bit_set.StaticBitSet;
const DynamicBitSet = std.bit_set.DynamicBitSet;
const Random = std.rand.Random;


// ================================ SOLVER STUFF ================================
// ==============================================================================

/// Terminology:
///
/// Cell: a square on the board where you place numbers.
/// Candidates: the numbers which are valid to place in a cell.
/// Line: a row or column on the board.
/// Neighbour: all the cells that share a row or column with a cell.
///
/// This solver basically consists of "depth first search + constraint
/// propagation". The "depth first search" part is implemented in `search` and
/// the "constraint propagation" part is `Line.remove_candidate_and_propagate`,
/// so those are both good places to start to see how the solver works.

pub const log_level: std.log.Level = .info;
const has_gui = build_options.mode == .gui;

const WALL: u8 = 10;
const max_candidates: u8 = 9;

/// A Kakuro game, partitioned into into a static (AuxData) and a dynamic (State) part.
const Kakuro = struct {
    state: State,
    aux_data: AuxData,

    fn deinit(self: *@This(), allocator: Allocator) void {
        self.state.deinit(allocator);
        self.aux_data.deinit(allocator);
    }
};

/// The _dynamic_ data for Kakuro board, the state of the board at a point in
/// time. This is the main data structure used in solving so it is essential
/// that the size is kept small.
///
/// See `AuxData` for static data.
const State = struct {
    /// The candidates for each cell on the board, indexed from top left to
    /// bottom right.
    candidates: []Candidates,

    /// Where the last move was made on the board, used during search as the
    /// starting point of propagation.
    move_index: ?u16 = null,

    /// Number of cells filled, used to quickly test for terminal states (i.e
    /// all cells are filled).
    n_filled: u16,

    const Self = @This();

    fn place(self: *Self, index: usize, val: u8) void {
        assert(self.candidates[index].is_candidate(val));
        var cnds = &self.candidates[index];
        cnds.set_unique(val);
        cnds.set_filled();
        self.n_filled += 1;
    }

    fn get(self: Self, index: usize) u8 {
        return self.candidates[index].get_unique();
    }

    fn is_filled(self: Self, index: usize) bool {
        return self.candidates[index].is_filled();
    }

    fn is_terminal(self: Self, n_cells: usize) bool {
        return self.n_filled == n_cells;
    }

    fn is_solved(self: Self, precomputed_lines: PrecomputedLines) bool {
        for (precomputed_lines.lines) |line| {
            if (!line.is_solved(self)) return false;
        }
        return true;
    }

    fn init(allocator: Allocator, aux_data: AuxData) !Self {
        const candidates = blk: {
            var n_cells: usize = 0;
            for (aux_data.board) |c| {
                if (c != WALL) n_cells += 1;
            }
            var all_candidates = try allocator.alloc(Candidates, n_cells);
            mem.set(Candidates, all_candidates, .{ .cs = 0b111_111_111 });
            break :blk all_candidates;
        };

        var state = Self {
            .candidates = candidates,
            .n_filled = 0,
            .move_index = null,
        };

        // Apply upper and lower bounds on all cells as a pre-pass.
        for (aux_data.precomputed_lines.lines) |line| {
            const mask = line_constraints_mask(line.len, line.constraint);
            var i: u8 = 0;
            while (i < line.len) : (i += 1) {
                const index = line.indices[i];
                var cnds = &state.candidates[index];
                cnds.mask(mask);
            }
        }

        return state;
    }

    fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.candidates);
    }

    fn clone(self: Self, allocator: Allocator, move_index: ?u16) !State {
        return Self {
            .candidates = try allocator.dupe(Candidates, self.candidates),
            .n_filled = self.n_filled,
            .move_index = move_index,
        };
    }
};

/// The _static_ data for Kakuro board, data that is the same at all points in
/// time for a particular Kakuro board. Kept separate from `State` to minimize
/// the data copied around during a search.
///
/// See `State` for dynamic data.
const AuxData = struct {
    board: []const u8,
    n_rows: usize,
    n_cols: usize,
    n_cells: usize,
    precomputed_lines: PrecomputedLines,

    fn deinit(self: *@This(), allocator: Allocator) void {
        self.precomputed_lines.deinit(allocator);
        allocator.free(self.board);
    }
};

const sum_1_to_9: u8 = 45;

inline fn lower_bound(n: u8, constraint: u8) u8 {
    const bound = sum_1_to_9 - ((11 - n) * (10 - n)) / 2;
    return if (bound >= constraint) 0 else constraint - bound;
}

inline fn upper_bound(val: u8, constraint: u8) u8 {
    const bound = (val * (val - 1)) / 2;
    return if (bound > constraint) 0 else constraint - bound;
}

/// The state of a cell on the board. The state is represented as a bitmap of
/// candidates:
///
/// Bits 1-9: set to 1 if the number at the position is a valid placement, 0 otherwise.
/// Bit 10: set to 1 if cell is filled, 0 otherwise.
///
/// A cell filled with the number `x` will have bit `x` and bit 9 set to 1, and
/// all other bits set to 0.
///
/// Even though this struct is quite small the solver spends roughly 30% of
/// its time copying arrays of candidates around, so anything to make this
/// struct smaller is worth considering.
const Candidates = struct {
    const Self = @This();
    const Int = u10;

    cs: Int,

    fn count(self: Self) u8 {
        return @popCount(Int, self.cs & ~(@as(u10, 1) << 9));
    }

    fn is_candidate(self: Self, val: u8) bool {
        assert(val > 0 and val <= max_candidates);
        return ((self.cs >> @intCast(u4, val - 1)) & 1) == 1;
    }

    fn get_unique(self: Self) u8 {
        assert(self.count() == 1);
        return @ctz(Int, self.cs) + 1;
    }

    fn set_unique(self: *Self, val: u8) void {
        assert(self.is_candidate(val));
        self.cs = (@as(Int, 1) << @intCast(u4, val - 1));
    }

    fn set_filled(self: *Self) void {
        self.cs |= @as(Int, 1) << 9;
    }

    fn is_filled(self: Self) bool {
        return self.cs >> 9 == 1;
    }

    fn mask(self: *Self, m: Int) void {
        self.cs &= m;
    }

    fn print(self: Self) void {
        var i: u8 = 1;
        debug.print("[", .{});
        while (i <= max_candidates) : (i += 1) {
            const v: u8 = if (self.is_candidate(i)) 1 else 0;
            debug.print("{d} ", .{v});
        }
        debug.print("]\n", .{});
    }
};

const PrecomputedLines = struct {
    lines: []Line,
    index_to_lines: []LinePair,

    const Self = @This();

    fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.lines);
        allocator.free(self.index_to_lines);
    }
};

/// Sets all bits above `n` to 0.
fn zero_above_inclusive(n: Candidates.Int) Candidates.Int {
    // TODO figure out how to remove this branch
    if (n >= max_candidates) return 0b111_111_111;
    const one = @as(Candidates.Int, 1);
    return 0b111_111_111 & ((one << @intCast(u4, n))  -| 1);
}

/// Sets all bits below `n` to 0.
fn zero_below_inclusive(n: Candidates.Int) Candidates.Int {
    const all_ones = ~@as(Candidates.Int, 0);
    return 0b111_111_111 & math.shl(Candidates.Int, all_ones, n -| 1);
}

/// Computes a bitmask where the 0's are the candidates that are NOT valid
/// for the given line length and constraint.
fn line_constraints_mask(line_len: u8, line_constraint: u8) Candidates.Int {
    const ub = upper_bound(line_len, line_constraint);
    const lb = lower_bound(line_len, line_constraint);
    const ma = zero_above_inclusive(ub);
    const mb = zero_below_inclusive(lb);
    return 0b111_111_111 & ma & mb;
}

/// Same as `line_constraints_mask` but also removes a specific candidate.
fn candidates_mask(line_len: u8, line_constraint: u8, candidate_to_remove: u8) Candidates.Int {
    const start_mask = line_constraints_mask(line_len, line_constraint);
    const candidate_mask = ~@as(Candidates.Int, 0) ^ (@as(Candidates.Int, 1) << @intCast(u4, candidate_to_remove - 1));
    const out = start_mask & candidate_mask;
    // log.debug("candidates_mask >> len {d} constraint {d} val {d} ub {d} lb {d} ma {b:9} mb {b:9} val_mask {b:9} out {b:9}\n", .{line_len, line_constraint, val_to_remove, ub, lb, ma, mb, val_mask, out});
    return out;
}

/// Represents a row or column on the board. Only contains static data,
/// uses `indices` to fetch data from the current `State` during solving.
const Line = struct {
    /// The cells that comprise this line -> indices into `State.candidates`.
    indices: [9]u16,
    len: u8,
    constraint: u8,

    const Self = @This();

    fn is_solved(self: Self, state: State) bool {
        var i: u8 = 0;
        var sum: u8 = 0;
        var used = StaticBitSet(9).initEmpty();

        while (i < self.len) : (i += 1) {
            const index = self.indices[i];
            if (!state.is_filled(index)) return false;

            const val = state.get(index);
            if (used.isSet(val - 1)) return false;

            used.set(val - 1);
            sum += val;
        }
        return sum == self.constraint;
    }

    /// The heart of the solver. After a number has been placed in a cell, this
    /// function figures out which candidates are no longer valid in that line
    /// and applies it to all neighbours. If any of those neighbours only have
    /// one possible candidate after this step, the procedure repeats for that
    /// cell.
    fn remove_candidate_and_propagate(self: Self, state: *State, stack: *PropagateStack, candidate: u8, run_context: *RunContext, should_draw_propagation: bool) !bool {
        // Compute the number of filled cells and their sum.
        var sum: u8 = 0;
        var n_filled: u8 = 0;
        {
            var i: u8 = 0;
            while (i < self.len) : (i += 1) {
                const index = self.indices[i];
                const cnds = state.candidates[index];

                if (cnds.is_filled()) {
                    const val = state.get(index);
                    sum += @intCast(u8, val);
                    n_filled += 1;
                }
            }
        }

        const n_empty = self.len - n_filled;
        if (n_empty > 0) {
            const current_constraint = self.constraint - sum;
            const candidates_to_remove = candidates_mask(n_empty, current_constraint, candidate);
            var i: u8 = 0;
            while (i < self.len) : (i += 1) {
                const index = self.indices[i];
                // debug.print("i {d} index {d} cnds {b:10} filled {s}\n", .{i, index, state.candidates[index].cs, state.is_filled(index)});
                if (state.is_filled(index)) continue;

                var cnds = &state.candidates[index];
                if (has_gui and should_draw_propagation) {
                    const old = cnds.*;
                    cnds.mask(candidates_to_remove);
                    if (old.cs != cnds.cs) {
                        try run_context.markPropagation(index, old, cnds.*);
                    }
                    std.time.sleep(run_context.sleep_time_ns / 3);
                } else {
                    cnds.mask(candidates_to_remove);
                }
                switch (cnds.count()) {
                    0 => return false,
                    1 => try stack.append(index),
                    else => {},
                }
            }
        }

        return true;
    }
};

const LinePair = struct {
    row: Line,
    col: Line,
};

inline fn cell_index(n_cols: usize, row: usize, col: usize) usize {
    return (row * n_cols) + col;
}

fn compute_lines(
    allocator: Allocator,
    desc: Description,
    n_cells: usize,
) !PrecomputedLines {
    const n_rows = desc.n_rows;
    const n_cols = desc.n_cols;
    const board = desc.board;
    const row_constraints = desc.row_constraints;
    const col_constraints = desc.col_constraints;

    var index_to_lines_rows = try allocator.alloc(Line, n_cells);
    defer allocator.free(index_to_lines_rows);

    var index_to_lines_cols = try allocator.alloc(Line, n_cells);
    defer allocator.free(index_to_lines_cols);

    const board_size = board.len;

    const index_to_cell_index = try allocator.alloc(u16, board_size);
    {
        var cindex: u16 = 0;
        for (board) |c, i| {
            if (c != WALL) {
                index_to_cell_index[i] = cindex;
                cindex += 1;
            }
        }
    }

    const IndicesType = [9]u16;

    var i: usize = 0;
    var indices = mem.zeroes(IndicesType);
    var len: u8 = 0;

    // Compute all rows
    var lines = try ArrayList(Line).initCapacity(allocator, n_cells);
    while (i < n_rows) : (i += 1) {
        var j: usize = 0;
        while (j < n_cols) : (j += 1) {
            const index = cell_index(n_cols, i, j);

            // debug.print("ROW >>> c {d} i {d} j {d} index {d} len {d}\n", .{board[index], i, j, index, len});

            if (board[index] == WALL and len == 0)
                continue;

            if (board[index] != WALL) {
                indices[len] = index_to_cell_index[index];
                len += 1;
            }

            if ((board[index] == WALL or j == n_cols - 1) and len > 0) {
                const prev_index = cell_index(n_cols, i, j - 1);

                // Edge case: a length 1 row on the edge of the board.
                // In this case, we grab the row constraint from the current
                // cell rather than the previous cell.
                const constraint_index = if (board[prev_index] == WALL)
                    index
                else
                    prev_index;

                const row = Line {
                    .indices = indices,
                    .len = len,
                    .constraint = row_constraints[constraint_index],
                };
                lines.appendAssumeCapacity(row);
                var k: u8 = 0;
                while (k < len) : (k += 1) {
                    index_to_lines_rows[indices[k]] = row;
                }
                indices = mem.zeroes(IndicesType);
                len = 0;
            }
        }
    }

    // debug.print("curr cell {d}\n", .{curr_cell});

    // Compute all cols
    i = 0;
    indices = mem.zeroes(IndicesType);
    len = 0;
    while (i < n_cols) : (i += 1) {
        var j: usize = 0;
        while (j < n_rows) : (j += 1) {
            const index = cell_index(n_cols, j, i);

            // debug.print("COL >>> c {d} i {d} j {d} index {d} len {d}\n", .{board[index], i, j, index, len});

            if (board[index] == WALL and len == 0)
                continue;

            if (board[index] != WALL) {
                indices[len] = index_to_cell_index[index];
                len += 1;
            }

            if ((board[index] == WALL or j == n_rows - 1) and len > 0) {
                const prev_index = cell_index(n_cols, j - 1, i);

                // Edge case: a length 1 col on the edge of the board.
                // In this case, we grab the col constraint from the current
                // cell rather than the previous cell.
                const constraint_index = if (board[prev_index] == WALL)
                    index
                else
                    prev_index;

                const col = Line {
                    .indices = indices,
                    .len = len,
                    .constraint = col_constraints[constraint_index],
                };
                lines.appendAssumeCapacity(col);
                var k: u8 = 0;
                while (k < len) : (k += 1) {
                    index_to_lines_cols[indices[k]] = col;
                }
                indices = mem.zeroes(IndicesType);
                len = 0;
            }
        }
    }

    var index_to_lines = try allocator.alloc(LinePair, n_cells);
    {
        var j: usize = 0;
        while (j < n_cells) : (j += 1) {
            const row = index_to_lines_rows[j];
            const col = index_to_lines_cols[j];
            index_to_lines[j] = LinePair {
                .row = row,
                .col = col,
            };
        }
    }

    return PrecomputedLines {
        .lines = lines.toOwnedSlice(),
        .index_to_lines = index_to_lines,
    };
}

/// A stack that only allows unique items. Attempting to add a number
/// already in the stack will silently do nothing.
const PropagateStack = struct {
    stack: ArrayList(usize),
    in_stack: DynamicBitSet,

    const Self = @This();

    fn append(self: *Self, val: usize) !void {
        if (self.in_stack.isSet(val)) return;

        try self.stack.append(val);
        self.in_stack.set(val);
    }

    fn pop(self: *Self) usize {
        const val = self.stack.pop();
        self.in_stack.unset(val);
        return val;
    }

    fn init(allocator: Allocator, size: usize) !Self {
        return Self{
            .stack = ArrayList(usize).init(allocator),
            .in_stack = try DynamicBitSet.initEmpty(allocator, size),
        };
    }

    fn deinit(self: *Self) void {
        self.stack.deinit();
        self.in_stack.deinit();
    }

    fn clear(self: *Self) void {
        self.stack.clearRetainingCapacity();

        // Copied from the source of DynamicBitSet since it lacks a
        // `clear` method.
        const bit_length = self.in_stack.unmanaged.bit_length;
        const num_masks = (bit_length + (@bitSizeOf(usize) - 1)) / @bitSizeOf(usize);
        for (self.in_stack.unmanaged.masks[0..num_masks]) |*mask| {
            mask.* = 0;
        }
    }
};

fn propagate(stack: *PropagateStack, state: *State, precomputed_lines: PrecomputedLines, run_context: *RunContext) !bool {
    defer stack.clear();

    if (state.move_index) |move_index| try stack.append(move_index);

    const should_draw_propagation = if (has_gui)
        @atomicLoad(u8, &run_context.should_draw_propagation, .SeqCst) == 1
    else
        false;

    while (stack.stack.items.len > 0) {
        const index = stack.pop();
        assert(!state.is_filled(index));

        const val = state.candidates[index].get_unique();
        state.place(index, val);

        const lines = precomputed_lines.index_to_lines[index];

        const row_consistent = try lines.row.remove_candidate_and_propagate(state, stack, val, run_context, should_draw_propagation);
        if (!row_consistent) return false;

        const col_consistent = try lines.col.remove_candidate_and_propagate(state, stack, val, run_context, should_draw_propagation);
        if (!col_consistent) return false;
    }

    return true;
}

const SearchOpts = struct {
    max_iters: usize,
};

const SearchResult = struct {
    allocator: Allocator,
    solution: ?*State,
    iters: usize,
    start_time: i64,
    end_time: i64,

    const Self = @This();

    fn deinit(self: *Self) void {
        if (self.solution) |s| {
            self.allocator.destroy(s);
        }
    }

    fn elapsed(self: Self) i64 {
        return self.end_time - self.start_time;
    }

    fn iters_per_second(self: Self) f64 {
        return @intToFloat(f64, self.iters) / (@intToFloat(f64, self.elapsed()) / time.ns_per_s);
    }
};

fn search(allocator: Allocator, _stack: *ArrayList(State), state: State, aux_data: AuxData, opts: SearchOpts, run_context: *RunContext) !SearchResult {
    var stack = _stack.*;

    try stack.append(state);
    var rewind_clone: *State = undefined;
    if (has_gui) {
        rewind_clone = try allocator.create(State);
        rewind_clone.* = try state.clone(allocator, state.move_index);
        try run_context.rewinds.add(allocator, rewind_clone);
    }

    var iters: usize = 0;
    var current: State = undefined;
    var result = SearchResult {
        .allocator = allocator,
        .solution = null,
        .iters = undefined,
        .start_time = @intCast(i64, time.nanoTimestamp()),
        .end_time = undefined,
    };

    if (has_gui) {
        @atomicStore(?*const State, &run_context.state, &state, .SeqCst);
    }

    var propagate_stack = try PropagateStack.init(allocator, aux_data.n_cells);
    defer propagate_stack.deinit();

    {
        var i: usize = 0;
        while (i < aux_data.n_cells) : (i += 1) {
            if (state.is_filled(i)) continue;

            if (state.candidates[i].count() == 1)
                propagate_stack.append(i) catch unreachable;
        }
    }

    const precomputed_lines = aux_data.precomputed_lines;
    while (stack.items.len > 0 and iters < opts.max_iters) {
        if (has_gui and !run_context.running.load(.SeqCst))
            break;
        // if (iters % 10 == 0)
        //     std.log.info("search stack {d} iters {d}\n", .{stack.items.len, iters});
        iters += 1;
        if (has_gui) {
            run_context.iters.store(iters, .SeqCst);
        }

        current = stack.pop();
        defer {
            if (has_gui) {
                if (run_context.paused == 1) {
                    run_context.pauseMutex.lock();
                    run_context.pauseCond.wait(&run_context.pauseMutex);
                    run_context.pauseMutex.unlock();
                }
                run_context.clearPropagations();
            }
            current.deinit(allocator);
        }


        if (has_gui) {
            std.time.sleep(run_context.sleep_time_ns / run_context.sleep_time_multiplier);
        }

        const consistent = try propagate(&propagate_stack, &current, precomputed_lines, run_context);

        if (has_gui) {
            rewind_clone = try allocator.create(State);
            rewind_clone.* = try current.clone(allocator, current.move_index);
            try run_context.rewinds.add(allocator, rewind_clone);
            @atomicStore(?*const State, &run_context.state, &current, .SeqCst);
            run_context.consistent.store(consistent, .SeqCst);
        }

        if (!consistent) continue;

        if (current.is_terminal(aux_data.n_cells)) {
            assert(current.is_solved(precomputed_lines));

            var solution_ptr = try allocator.create(State);
            solution_ptr.* = try current.clone(allocator, 0);
            result.solution = solution_ptr;
            result.iters = iters;
            result.end_time = @intCast(i64, time.nanoTimestamp());

            if (has_gui) {
                var draw_solution = try allocator.create(State);
                draw_solution.* = try current.clone(allocator, null);
                @atomicStore(?*const State, &run_context.state, draw_solution, .SeqCst);
            }

            return result;
        }

        // Find the best candidate for the next move
        var best_index: u16 = 0;
        var best_cell: Candidates = undefined;
        var best_count: u8 = 10;
        for (current.candidates) |cnds, i| {
            if (current.is_filled(i)) continue;
            const count = cnds.count();

            if (count > 1 and count < best_count) {
                best_index = @intCast(u16, i);
                best_cell = cnds;
                best_count = count;
            }
        }

        var candidates = try std.BoundedArray(u8, 9).init(0);
        {
            var i: u8 = 1;
            while (i <= max_candidates) : (i += 1) {
                if (!best_cell.is_candidate(i)) continue;
                try candidates.append(i);
            }
        }

        for (candidates.slice()) |i| {
            var copy = try current.clone(allocator, best_index);
            copy.candidates[best_index].set_unique(i);
            try stack.append(copy);
        }
    }

    result.iters = iters;
    result.end_time = @intCast(i64, time.nanoTimestamp());
    return result;
}

const Description = struct {
    n_rows: usize,
    n_cols: usize,
    board: []const u8,
    row_constraints: []const u8,
    col_constraints: []const u8,
    solution: []const u8,
    state_index_to_board_index: []const usize,
};

fn parse_descriptions(allocator: Allocator, path: []const u8) ![]Description {
    const file = try fs.cwd().openFile(path, .{ .read = true });
    defer file.close();
    const stat = try file.stat();
    const contents = try file.reader().readAllAlloc(
        allocator,
        stat.size,
    );
    defer allocator.free(contents);

    var descriptions = ArrayList(Description).init(allocator);
    var rows = mem.split(u8, contents, "\n");

    // Remove comments
    while (rows.next()) |row| {
        if (row.len == 0 or row[0] != '#')
            break;
    }

    while(rows.index != null) {
        var desc = try parse_description(allocator, &rows);
        try descriptions.append(desc);
        _ = rows.next();
    }
    return descriptions.toOwnedSlice();
}

fn parse_description(allocator: Allocator, rows: *mem.SplitIterator(u8)) !Description {
    var n_rows: usize = undefined;
    var n_cols: usize = undefined;
    {
        var n: usize = undefined;
        {
            const untrimmed_row = rows.next().?;
            const row = std.mem.trimRight(u8, untrimmed_row, &std.ascii.spaces);
            var info = mem.split(u8, row, " ");
            var j: u8 = 0;
            while (info.next()) |c| {
                defer j += 1;
                switch (j) {
                    0 => n = try std.fmt.parseInt(usize, c, 10),
                    1 => n_rows = try std.fmt.parseInt(u8, c, 10),
                    2 => n_cols = try std.fmt.parseInt(u8, c, 10),
                    else => {}
                }
            }
            assert(j == 3);
        }
    }

    // debug.print("n {d} rows {d} cols {d}\n", .{n, n_rows, n_cols});

    const board_size = n_rows * n_cols;
    var board = try ArrayList(u8).initCapacity(allocator, board_size);
    var row_constraints = try ArrayList(u8).initCapacity(allocator, board_size);
    var col_constraints = try ArrayList(u8).initCapacity(allocator, board_size);
    var solution = try ArrayList(u8).initCapacity(allocator, board_size);
    defer solution.deinit();

    var section: usize = 0;
    var i: usize = 1;
    while (rows.next()) |untrimmed_row| {
        const row = std.mem.trimRight(u8, untrimmed_row, &std.ascii.spaces);
        if (row.len == 0) continue;

        // std.log.info("i {d} section {d}\n", .{i, section});
        var list = switch (section) {
            0 => &board,
            1 => &row_constraints,
            2 => &col_constraints,
            3 => &solution,
            else => unreachable
        };
        var inner_row = mem.split(u8, row, " ");
        while (inner_row.next()) |d| {
            const parsed = try std.fmt.parseInt(u8, d, 10);
            try list.append(parsed);
        }
        if (i % n_rows == 0) {
            section += 1;
            if (section > 3) break;
        }

        i += 1;
    }

    var state_index_to_board_index = std.ArrayList(usize).init(allocator);
    {
        for (board.items) |b, j| {
            if (b != 0) continue;
            try state_index_to_board_index.append(j);
        }
    }

    return Description{
        .n_rows = n_rows,
        .n_cols = n_cols,
        .board = board.toOwnedSlice(),
        .row_constraints = row_constraints.toOwnedSlice(),
        .col_constraints = col_constraints.toOwnedSlice(),
        .solution = solution.toOwnedSlice(),
        .state_index_to_board_index = state_index_to_board_index.toOwnedSlice(),
    };
}

fn createKakuros(allocator: Allocator, descriptions: []const Description) ![]Kakuro {
    var kakuros = try allocator.alloc(Kakuro, descriptions.len);

    for (descriptions) |desc, i| {
        const n_cells = blk: {
            var n: usize = 0;
            for (desc.board) |c| {
                if (c != WALL) n += 1;
            }
            break :blk n;
        };

        const aux_data = AuxData {
            .board = desc.board,
            .n_rows = desc.n_rows,
            .n_cols = desc.n_cols,
            .n_cells = n_cells,
            .precomputed_lines = try compute_lines(
                allocator,
                desc,
                n_cells,
            ),
        };
        const state = try State.init(allocator, aux_data);
        kakuros[i]  = Kakuro{
            .state = state,
            .aux_data = aux_data,
        };
    }

    return kakuros;
}

const Searcher = struct {
    const logger = std.log.scoped(.searcher);
    const MoveOrderEnum = enum {
        linear,
        random,
    };

    const Stats = struct {
        total_time: i64 = 0,
        total_iters: usize = 0,
    };

    const Self = @This();

    kakuro: Kakuro,
    stack: *ArrayList(State),
    max_iters_total: usize,
    max_iters_per_search: usize,
    max_retries: usize,
    move_order_enum: MoveOrderEnum,
    run_context: *RunContext,

    fn do_search(self: *Self, allocator: Allocator) !?SearchResult {
        const root = self.kakuro.state;
        const aux_data = self.kakuro.aux_data;
        var stack = self.stack;

        var stats = Stats{};
        var max_iters = self.max_iters_per_search;
        while (true) {
            const opts = SearchOpts {
                .max_iters = max_iters,
            };

            var i: usize = 0;
            while (i < self.max_retries) : (i += 1) {
                var arena = ArenaAllocator.init(allocator);
                var arena_allocator = arena.allocator();
                // defer arena.deinit();

                var cloned = try root.clone(arena_allocator, null);
                var result = try search(arena_allocator, stack, cloned, aux_data, opts, self.run_context);
                self.stack.resize(0) catch unreachable;
                stats.total_iters += result.iters;
                stats.total_time += result.elapsed();

                if(result.solution) |solution| {
                    _ = solution;
                    logger.info("SOLVED >>> iters {d:^2} iters/s {d:^10.2} time {s}", .{result.iters, result.iters_per_second(), fmt.fmtDurationSigned(result.elapsed())});
                }
                else {
                    logger.info("{d}/{d} ### FAILED ### >>> iters {d} iters/s {d} time {s}", .{i + 1, self.max_retries, result.iters, result.iters_per_second(), fmt.fmtDurationSigned(result.elapsed())});
                }
                return result;
            } else {
                debug.print("TOTAL FAILURE\n", .{});
                const old_max_iters = max_iters;
                max_iters *= 2;
                if (max_iters > self.max_iters_total)
                    break;
                logger.info("max iters {d} -> {d}\n", .{old_max_iters, max_iters});
                return null;
            }
        }
        unreachable;
    }
};

/// Kakuros that have so far resisted a solution. They are not actually unsolvable,
/// just not solvable by this Kakuro solver (so far!).
const unsolvable = blk: {
    var arr = [_]bool { false } ** 960;
    arr[187] = true;
    arr[237] = true;
    arr[240] = true;
    arr[251] = true;
    arr[253] = true;
    arr[258] = true;
    arr[259] = true;
    arr[260] = true;
    arr[941] = true;
    break :blk arr;
};

const Runner = struct {
    const logger = std.log.scoped(.runner);

    kakuros: []const Kakuro,
    allocator: Allocator,
    stack: *ArrayList(State),
    skips: ?[]const bool,
    solution: ?State,
    run_context: *RunContext,
    search_results: []?SearchResult,

    const Self = @This();

    fn init(allocator: Allocator, kakuros: []Kakuro, run_context: *RunContext) !Self {
        var stack = try allocator.create(ArrayList(State));
        stack.* = ArrayList(State).init(allocator);
        return Self {
            .kakuros = kakuros,
            .allocator = allocator,
            .stack = stack,
            .skips = &unsolvable,
            .solution = null,
            .run_context = run_context,
            .search_results = try allocator.alloc(?SearchResult, kakuros.len),
        };
    }

    fn runOne(self: *Self, index: usize) !void {
        logger.info("#{d} Running...", .{index});
        const search_result = try create_searcher(self, index - 1).do_search(self.allocator);
        if(search_result) |sr| {
            if (sr.solution) |s| {
                self.solution = s.*;
            }
        }
        self.search_results[index - 1] = search_result;
    }

    fn runRange(self: *Self, start: usize, end: usize) !void {
        var index: usize = start;
        while (index < end) : (index += 1) {
            if (self.skips) |skip| {
                if (skip[index]) continue;
            }
            try self.runOne(index);
        }
    }

    fn runAll(self: *Self) !void {
        try self.runRange(1, self.kakuros.len + 1);
    }

    fn create_searcher(self: *Self, index: usize) Searcher {
        return Searcher {
            .kakuro = self.kakuros[index],
            .stack = self.stack,
            .max_iters_total = 15_000_000,
            .max_iters_per_search = 4_000_000,
            .max_retries = 50,
            .move_order_enum = .linear,
            .run_context = self.run_context,
        };
    }

    fn report(self: Self) !void {
        var w = std.io.getStdOut().writer();
        try w.print("index,iters,time_ns\n", .{});
        for (self.search_results) |result, i| {
            if (result) |r| {
                try w.print("{d},{d},{d}\n", .{i + 1, r.iters, r.elapsed()});
            } else {
                try w.print("{d},{d},{d}\n", .{i + 1, -1, -1});
            }
        }
    }
};

fn display_board(state: State, aux_data: AuxData) void {
    const board = aux_data.board;
    const candidates = state.candidates;
    var cindex: usize = 0;
    for (board) |c, i| {
        if (c == WALL) {
            debug.print("  -", .{});
        } else {
            if (candidates[cindex].is_filled()) {
                debug.print("{d:3}", .{candidates[cindex].get_unique()});
            } else {
                debug.print("  X", .{});
            }
            cindex += 1;
        }
        if ((i + 1) % aux_data.n_cols == 0)
            debug.print("\n", .{});
    }
}

fn display_candidates(state: State, aux_data: AuxData) void {
    const buf_size: u8 = 9 * 2 + 2;
    const board = aux_data.board;
    var candidate_strs = testing.allocator.alloc([buf_size]u8, board.len) catch unreachable;
    defer testing.allocator.free(candidate_strs);

    var max_len: usize = 0;

    var cindex: usize = 0;
    var i: usize = 0;
    while (i < board.len) : (i += 1) {
        var store = [_]u8{' '} ** buf_size;
        var buf = io.fixedBufferStream(&store);
        var w = buf.writer();

        if (board[i] == WALL) {
            w.print("X", .{}) catch unreachable;
        } else {
            const cnds = state.candidates[cindex];
            w.writeByte('[') catch unreachable;
            {
                var j: u8 = 0;
                var n_written: u8 = 1;
                while (j < max_candidates) : (j += 1) {
                    if (cnds.is_candidate(j + 1)) {
                        w.print("{d}", .{j + 1}) catch unreachable;
                        if (n_written < cnds.count())
                            w.writeByte(' ') catch unreachable;
                        n_written += 1;
                    }
                }
            }
            w.writeByte(']') catch unreachable;
            cindex += 1;
        }

        const len = buf.getWritten().len;
        if (len > max_len)
            max_len = len;

        candidate_strs[i] = store;
    }

    var j: usize = 1;
    for (candidate_strs) |str| {
        debug.print("{s} ", .{str[0..max_len]});
        if (j % aux_data.n_cols == 0)
            debug.print("\n", .{});
        j += 1;
    }
}

fn printInfo(comptime T: type) void {
    @compileLog(T, @sizeOf(T), @alignOf(T));
}

fn printSizes() void {
    printInfo(Candidates);
    printInfo(State);
    printInfo(Line);
    printInfo(LinePair);
}


// ================================== GUI STUFF =================================
// ==============================================================================

// The first rewind is the current state,
// so this gives us current + 10 rewinds.
const max_rewinds: usize = 11;

const Queue = struct {
    len: usize,
    head: ?*Node,
    tail: ?*Node,

    const Self = @This();

    fn init() Self {
        return .{
            .len = 0,
            .head = null,
            .tail = null,
        };
    }

    fn add(self: *Self, allocator: Allocator, value: *State) !void {
        var tail = try allocator.create(Node);
        tail.* = .{
            .prev = self.tail,
            .next = null,
            .value = value,
        };

        if (self.tail) |t| {
            t.next = tail;
        }
        self.tail = tail;
        self.len = @minimum(self.len + 1, max_rewinds);
        if (self.len == 1) {
            self.head = tail;
        }
        if (self.len > max_rewinds) {
            var head = self.head.?;
            self.head = head.next;
            self.head.?.prev = null;
            head.deinit(allocator);
        }
    }

    fn get(self: Self, index: usize) ?*State {
        if (index > self.len - 1) return null;

        var i: usize = 0;
        var node = self.tail;
        while (i < index) : (i += 1) {
            if (node) |n| {
                node = n.prev;
            }
        }
        return node.?.value;
    }
};

const Node = struct {
    prev: ?*Node,
    next: ?*Node,
    value: *State,

    const Self = @This();

    fn deinit(self: *Self, allocator: Allocator) void {
        self.value.deinit(allocator);
        allocator.destroy(self.value);
        allocator.destroy(self);
    }
};

const Propagation = struct {
    index: usize,
    old: Candidates,
    new: Candidates,
};

const RunContext = if (build_options.mode == .gui) struct {
    running: Atomic(bool),
    rewinds: Queue,
    rewind_index: usize,
    state: ?*const State,
    consistent: Atomic(bool),
    iters: Atomic(usize),
    sleep_time_multiplier: usize,
    paused: u8,
    pauseMutex: std.Thread.Mutex,
    pauseCond: std.Thread.Condition,
    propagations: std.AutoArrayHashMap(usize, Propagation),
    propagations_mutex: std.Thread.Mutex,
    should_draw_propagation: u8,
    sleep_time_ns: usize,

    const Self = @This();

    fn markPropagation(self: *Self, index: usize, old: Candidates, new: Candidates) !void {
        self.propagations_mutex.lock();
        defer self.propagations_mutex.unlock();

        try self.propagations.put(index, .{
            .index = index,
            .old = old,
            .new = new,
        });
    }

    fn getPropagations(self: *Self, allocator: Allocator) !std.AutoArrayHashMap(usize, Propagation) {
        _ = allocator;
        self.propagations_mutex.lock();
        defer self.propagations_mutex.unlock();

        return try self.propagations.clone();
    }

    fn clearPropagations(self: *Self) void {
        self.propagations_mutex.lock();
        defer self.propagations_mutex.unlock();

        self.propagations.clearRetainingCapacity();
    }
} else void;

const cellSize: usize = 55;

const SolutionDrawMode = enum {
    none,
    solution,
    diff,
};

const screenWidth = 1200;
const screenHeight = 950;

fn numberKeyWasPressed() ?rl.KeyboardKey {
    const keys = [_]rl.KeyboardKey {
        rl.KeyboardKey.KEY_ZERO,
        rl.KeyboardKey.KEY_ONE,
        rl.KeyboardKey.KEY_TWO,
        rl.KeyboardKey.KEY_THREE,
        rl.KeyboardKey.KEY_FOUR,
        rl.KeyboardKey.KEY_FIVE,
        rl.KeyboardKey.KEY_SIX,
        rl.KeyboardKey.KEY_SEVEN,
        rl.KeyboardKey.KEY_EIGHT,
        rl.KeyboardKey.KEY_NINE,
    };

    for (keys) |k| {
        if (rl.IsKeyPressed(k))
            return k;
    }
    return null;
}

fn createRunContext(
    allocator: Allocator,
    kakuros: []Kakuro,
    drawIndex: usize,
    run_context: *RunContext,
    runner: *Runner,
    tid: *std.Thread,
    should_reset_camera: *bool,
) !void {
    run_context.* = RunContext{
        .running = Atomic(bool).init(true),
        .rewinds = Queue.init(),
        .rewind_index = 0,
        .consistent = Atomic(bool).init(true),
        .state = null,
        .iters = Atomic(usize).init(0),
        .sleep_time_multiplier = 1,
        .paused = 0,
        .pauseMutex = .{},
        .pauseCond = .{},
        .propagations = std.AutoArrayHashMap(usize, Propagation).init(allocator),
        .propagations_mutex = .{},
        .should_draw_propagation = 0,
        .sleep_time_ns = 500_000_000,
    };
    runner.* = try Runner.init(allocator, kakuros, run_context);
    tid.* = try std.Thread.spawn(.{}, Runner.runOne, .{runner, drawIndex});
    should_reset_camera.* = true;
}

fn recreateRunContext(
    allocator: Allocator,
    kakuros: []Kakuro,
    drawIndex: usize,
    run_context: *RunContext,
    runner: *Runner,
    tid: *std.Thread,
    should_reset_camera: *bool,
) !void {
    @atomicStore(u8, &run_context.paused, 0, .SeqCst);
    run_context.pauseCond.signal();
    run_context.running.store(false, .SeqCst);
    std.Thread.join(tid.*);

    try createRunContext(allocator, kakuros, drawIndex, run_context, runner, tid, should_reset_camera);
}


// TODO: replace haphazard thread safety with something that actually works
fn runGui(allocator: Allocator, descriptions: []const Description) !void {
    // Initialization
    //--------------------------------------------------------------------------------------

    rl.InitWindow(screenWidth, screenHeight, "zig-kakuro");
    rl.SetTargetFPS(60);

    const useCamera = true;
    var drawIndex: usize = 1;
    var shouldDrawCandidates = false;
    var shouldDrawCandidateIndexes = false;
    var shouldDrawIndexOverlay = false;
    var shouldDrawHelpOverlay = false;
    var shouldDrawDebugOverlay = true;
    var indexOverlayInputStartTime: f64 = 0;
    var keyBuffer = std.BoundedArray(u8, 3).init(0) catch unreachable;
    var solutionDrawMode: SolutionDrawMode = .none;

    const kakuros = try createKakuros(allocator, descriptions);
    var iters = try allocator.create(usize);
    iters.* = 0;
    var run_context: RunContext = undefined;
    var runner: Runner = undefined;
    var tid: std.Thread = undefined;
    var should_reset_camera = false;
    try createRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);

    var camera = rl.Camera2D{
        .target = .{ .x = 0, .y = 0 },
        .offset = .{ .x = 0, .y = 0 },
        .rotation = 0.0,
        .zoom = 0.8,
    };
    _ = camera;

    var prev_mouse_position = rl.GetMousePosition();

    while (!rl.WindowShouldClose()) {
        should_reset_camera = false;

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_M)) {
            std.log.info("PRESSED 'M'", .{});
            shouldDrawHelpOverlay = !shouldDrawHelpOverlay;
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_D)) {
            std.log.info("PRESSED 'D'", .{});
            shouldDrawDebugOverlay = !shouldDrawDebugOverlay;
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_Z)) {
            std.log.info("PRESSED 'Z'", .{});
            should_reset_camera = true;
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_U)) {
            if (drawIndex > 0) drawIndex -= 1;
            std.log.info("PRESSED 'U', index {d}", .{drawIndex});
            try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_I)) {
            if (drawIndex < descriptions.len) drawIndex += 1;
            std.log.info("PRESSED 'I', index {d}", .{drawIndex});

            try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_H)) {
            run_context.sleep_time_multiplier = @maximum(run_context.sleep_time_multiplier / 2, 1);
            std.log.info("PRESSED 'H', speed {d}", .{run_context.sleep_time_multiplier});
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_L)) {
            run_context.sleep_time_multiplier *= 2;
            std.log.info("PRESSED 'L', speed {d}", .{run_context.sleep_time_multiplier});
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_C)) {
            shouldDrawCandidates = !shouldDrawCandidates;
            std.log.info("PRESSED 'C', should draw candidates {d}", .{shouldDrawCandidates});
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_X)) {
            shouldDrawCandidateIndexes = !shouldDrawCandidateIndexes;
            std.log.info("PRESSED 'X', should draw candidate indexes {d}", .{shouldDrawCandidateIndexes});
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_O)) {
            _ = @atomicRmw(u8, &run_context.should_draw_propagation, .Xor, 1, .SeqCst);
            std.log.info("PRESSED 'O', should draw propagation {d}", .{run_context.should_draw_propagation});
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_S)) {
            solutionDrawMode = switch (solutionDrawMode) {
                .none => .solution,
                .solution => .diff,
                .diff => .none,
            };
            std.log.info("PRESSED 'S', should draw solution {s}", .{solutionDrawMode});
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_P)) {
            std.log.info("PRESSED 'P', paused {d}", .{run_context.paused});
            run_context.rewind_index = 0;
            const old = @atomicRmw(u8, &run_context.paused, .Xor, 1, .SeqCst);
            if (old == 1) {
                run_context.pauseCond.signal();
            }
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_R)) {
            std.log.info("PRESSED 'R'", .{});
            try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_LEFT)) {
            std.log.info("PRESSED 'LEFT'", .{});
            const rewind_index = @minimum(
                @minimum(run_context.rewind_index + 1, max_rewinds - 1),
                run_context.iters.load(.SeqCst),
            );
            run_context.rewind_index = rewind_index;
            run_context.state = run_context.rewinds.get(run_context.rewind_index);
        }

        if (rl.IsKeyPressed(rl.KeyboardKey.KEY_RIGHT)) {
            std.log.info("PRESSED 'RIGHT'", .{});
            run_context.rewind_index -|= 1;
            run_context.state = run_context.rewinds.get(run_context.rewind_index);
        }

        if (numberKeyWasPressed()) |key| {
            const number = @intCast(u8, @enumToInt(key));
            std.log.info("PRESSED '{d}'", .{number});
            if (!shouldDrawIndexOverlay) {
                shouldDrawIndexOverlay = true;
                indexOverlayInputStartTime = rl.GetTime();
            }
            try keyBuffer.append(number);
        }

        if (shouldDrawIndexOverlay and (keyBuffer.len == 3 or rl.GetTime() - indexOverlayInputStartTime > 0.7)) {
            defer {
                keyBuffer = std.BoundedArray(u8, 3).init(0) catch unreachable;
                shouldDrawIndexOverlay = false;
            }

            const index = std.fmt.parseInt(usize, keyBuffer.constSlice() , 10) catch unreachable;
            if (index > 0 and index <= kakuros.len) {
                drawIndex = index;
                try recreateRunContext(allocator, kakuros, drawIndex, &run_context, &runner, &tid, &should_reset_camera);
                should_reset_camera = true;
            }
        }

        const mouse_position = rl.GetMousePosition();
        defer prev_mouse_position = mouse_position;

        camera.zoom += rl.GetMouseWheelMove() * 0.05;
        if (rl.IsMouseButtonDown(rl.MouseButton.MOUSE_LEFT_BUTTON)) {
            const delta_x = mouse_position.x - prev_mouse_position.x;
            const delta_y = mouse_position.y - prev_mouse_position.y;
            camera.target = rl.Vector2{
                .x = camera.target.x - delta_x,
                .y = camera.target.y - delta_y,

            };
        }

        // Draw
        //----------------------------------------------------------------------------------
        var desc = descriptions[drawIndex - 1];
        if (should_reset_camera) {
            resetCamera(desc, &camera);
        }

        rl.BeginDrawing();
        rl.ClearBackground(rl.WHITE);
        if (useCamera)
            rl.BeginMode2D(camera);

        const board = desc.board;
        const row_constraints = desc.row_constraints;
        const col_constraints = desc.col_constraints;
        const n_cols = desc.n_cols;
        const n_rows = desc.n_rows;

        for (board) |s, index| {
            // std.log.info("index {d} cell {d}", .{index, s});
            const row = index / n_cols;
            const col = index % n_cols;
            if (s == 0) {
                drawEmptyCell(col + 1, row + 1);
            } else {
                drawOutOfBounds(col + 1, row + 1);
            }
        }


        var i: usize = 0;
        while (i < n_cols) : (i += 1) {
            drawOutOfBounds(i, 0);
        }

        i = 0;
        while (i < n_rows) : (i += 1) {
            drawOutOfBounds(0, i);
        }

        var prev: usize = 0;
        for (row_constraints) |constraint, index| {
            const row = index / n_cols;
            const col = index % n_cols;
            if (prev != constraint) {
                if (constraint != 0) {
                    const colConstraint = col_constraints[index];
                    drawNumberedBox(col, row + 1, constraint, colConstraint != 0);
                }
            }
            prev = constraint;
        }

        prev = 0;
        var c: usize = 0;
        while (c < n_cols) : (c += 1) {
            var r: usize = 0;
            while (r < n_rows) : (r += 1) {
                const index = r * n_cols + c;
                const constraint = col_constraints[index];
                if (prev != constraint) {
                    if (constraint != 0) {
                        const rowConstraint = row_constraints[index];
                        const shouldFill = r == 0 or rowConstraint == 0;
                        drawNumberedBoxInvert(c + 1, r, constraint, shouldFill);
                    }
                }
                prev = constraint;
            }
        }

        switch (solutionDrawMode) {
            .none => if (run_context.state) |state| {
                drawState(desc, state);
                if (state.move_index) |index| {
                    const board_index = desc.state_index_to_board_index[index];
                    const row = board_index / desc.n_cols;
                    const col = board_index % desc.n_cols;
                    drawCell(col + 1, row + 1, rl.Fade(rl.BLUE, 0.5));
                }
            },
            .solution => drawSolution(desc),
            .diff => if (run_context.state) |state| {
                drawState(desc, state);
                drawSolutionDiff(desc, state);
                if (state.move_index) |index| {
                    const board_index = desc.state_index_to_board_index[index];
                    const row = board_index / desc.n_cols;
                    const col = board_index % desc.n_cols;
                    drawCell(col + 1, row + 1, rl.Fade(rl.BLUE, 0.5));
                }
            },
        }

        var propagations = try runner.run_context.getPropagations(allocator);
        defer propagations.deinit();

        if (shouldDrawCandidates) {
            if (run_context.state) |state| {
                drawCandidates(desc, state, propagations);
            }
        }

        if (shouldDrawCandidateIndexes) {
            if (run_context.state) |state| {
                drawCandidateIndexes(desc, state);
            }
        }

        drawPropagations(desc, propagations);

        if (useCamera)
            rl.EndMode2D();

        if (shouldDrawDebugOverlay) {
            drawDebugOverlay(allocator, &run_context, drawIndex, solutionDrawMode, shouldDrawCandidates, kakuros.len);
        }

        if (shouldDrawIndexOverlay) {
            drawIndexOverlay(keyBuffer.constSlice());
        }

        if (shouldDrawHelpOverlay) {
            drawHelpOverlay();
        } else {
            const help = "Press 'M' for help";
            const font_size = 44;
            rl.DrawText(help[0..], screenWidth - computeTextWidth(help[0..], font_size) - 20, screenHeight - 70, font_size, rl.RED);
        }

        rl.EndDrawing();
        //----------------------------------------------------------------------------------
    }

    // De-Initialization
    //--------------------------------------------------------------------------------------
    rl.CloseWindow(); // Close window and OpenGL context
    //--------------------------------------------------------------------------------------
}

fn resetCamera(desc: Description, camera: *rl.Camera2D) void {
    const cell_size_float = @intToFloat(f32, cellSize);
    const board_width = @intToFloat(f32, desc.n_cols) * cell_size_float;
    const board_height = @intToFloat(f32, desc.n_rows) * cell_size_float;
    const width_scale_factor = screenWidth / board_width;
    const height_scale_factor = screenHeight / board_height;
    camera.zoom = @minimum(@minimum(width_scale_factor, height_scale_factor), 1.0);
    camera.target = .{ .x = 0, .y = 0 };
    camera.offset = .{ .x = 0, .y = 0 };
}

fn drawPropagations(desc: Description, propagations: std.AutoArrayHashMap(usize, Propagation)) void {
    for (propagations.keys()) |p| {
        const board_index = desc.state_index_to_board_index[p];
        const row = board_index / desc.n_cols;
        const col = board_index % desc.n_cols;
        drawCell(col + 1, row + 1, rl.Fade(rl.YELLOW, 0.3));
    }
}

// TODO: do this without allocation
fn drawDebugOverlay(
    allocator: Allocator,
    run_context: *const RunContext,
    drawIndex: usize,
    solutionDrawMode: SolutionDrawMode,
    shouldDrawCandidates: bool,
    n_kakuros: usize,
) void {
    const mouse_position = rl.GetMousePosition();
    const Printer = struct {
        allocator: Allocator,
        const Self = @This();
        fn printLine(self: Self, comptime str: []const u8, args: anytype) []const u8 {
            return std.fmt.allocPrintZ(self.allocator, str, args) catch unreachable;
        }
    };
    const printer = Printer{ .allocator = allocator };
    const format = printer.printLine;
    const strs = [_][]const u8 {
        format("index: {d}/{d}", .{drawIndex, n_kakuros}),
        format("iters: {d}", .{run_context.iters.load(.SeqCst) - run_context.rewind_index}),
        format("speed: {d}x", .{run_context.sleep_time_multiplier}),
        format("rewind: {d}", .{run_context.rewind_index}),
        format("solution: {s}", .{@tagName(solutionDrawMode)}),
        format("candidates: {}", .{shouldDrawCandidates}),
        format("consistent: {s}", .{run_context.consistent.load(.SeqCst)}),
        format("paused: {d}", .{run_context.paused != 0}),
        format("x {d} y {d}", .{@floatToInt(isize, mouse_position.x), @floatToInt(isize, mouse_position.y)}),
    };

    const font_size = 36;
    const offset = font_size + 4;
    const padding = 20;
    const width = 300 + padding * 2;
    const x_pos = screenWidth - width;
    const rec = rl.Rectangle{
        .x = @intToFloat(f32, x_pos),
        .y = 0,
        .width = @intToFloat(f32, width),
        .height = @intToFloat(f32, offset * strs.len + 5),
    };
    rl.DrawRectangleRec(rec, rl.Fade(rl.BLACK, 0.8));

    var y_pos: c_int = 0;
    for (strs) |str| {
        rl.DrawText(str.ptr, x_pos + padding, y_pos, font_size, rl.RED);
        y_pos += offset;
        allocator.free(str);
    }
}

fn drawHelpOverlay() void {
    const rec = rl.Rectangle {
        .x = 0,
        .y = 0,
        .width = screenWidth,
        .height = screenHeight,
    };
    rl.DrawRectangleRec(rec, rl.Fade(rl.BLACK, 0.8));

    const fontSize: u32 = 44;
    const padding: c_int = 20;
    var y_pos: c_int = padding;

    const strs = [_][]const u8 {
        "M - Show this menu",
        "[NUMBER] - Goto Kakuro",
        "P - Pause",
        "R - Reset",
        "I - Solve next board",
        "U - Solve previous board",
        "L - Increase solver speed",
        "H - Decrease solver speed",
        "O - Show propagation",
        "C - Show/hide candidates",
        "X - Show/hide indices",
        "S - Show/hide/diff solution",
        "D - Show/hide debug overlay",
        "Z - Reset camera",
        "LEFT - Rewind",
        "RIGHT - Go forward",
    };

    for (strs) |str| {
        rl.DrawText(str.ptr, padding, y_pos, fontSize, rl.WHITE);
        y_pos += fontSize + padding / 2;
    }
}

fn drawIndexOverlay(asciiNumbers: []const u8) void {
    const rec = rl.Rectangle {
        .x = 0,
        .y = 0,
        .width = screenWidth,
        .height = screenHeight,
    };
    rl.DrawRectangleRec(rec, rl.Fade(rl.BLACK, 0.8));

    var buf: [30:0]u8 = undefined;
    const str = std.fmt.bufPrintZ(&buf, "{s}", .{asciiNumbers}) catch unreachable;
    const fontSize: u32 = 188;
    const textSize = computeTextWidth(str, fontSize);
    rl.DrawText(str.ptr, (screenWidth / 2) - @divFloor(textSize, 2), @divFloor(screenHeight, 2) - (fontSize - 2), fontSize, rl.WHITE);
}

fn computeTextWidth(text: []const u8, fontSize: c_int) c_int {
    return rl.MeasureText(text.ptr, fontSize);
}

fn drawState(desc: Description, state: *const State) void {
    var stateIndex: usize = 0;
    for (desc.board) |s, i| {
        if (s != 0) continue;

        const candidate = state.candidates[stateIndex];
        if (candidate.count() == 1) {
            const row = i / desc.n_cols;
            const col = i % desc.n_cols;
            const val = candidate.get_unique();
            drawNumber(col + 1, row + 1, val);
        }
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawSolution(desc: Description) void {
    for (desc.solution) |s, i| {
        if (s == 0) continue;

        const row = i / desc.n_cols;
        const col = i % desc.n_cols;
        drawNumber(col + 1, row + 1, s);
    }
}

fn drawSolutionDiff(desc: Description, state: *const State) void {
    var stateIndex: usize = 0;
    for (desc.solution) |s, i| {
        if (s == 0) continue;

        const candidate = state.candidates[stateIndex];
        if (candidate.count() == 1) {
            const row = i / desc.n_cols;
            const col = i % desc.n_cols;
            const val = candidate.get_unique();
            const rec = rectangleForPoint(col + 1, row + 1);
            if (val == s) {
                rl.DrawRectangleRec(rec, rl.Fade(rl.GREEN, 0.3));
            } else {
                rl.DrawRectangleRec(rec, rl.Fade(rl.RED, 0.3));
            }
        }
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawCandidates(desc: Description, state: *const State, propagations: ?std.AutoArrayHashMap(usize, Propagation)) void {
    var stateIndex: usize = 0;
    for (desc.board) |s, i| {
        if (s != 0) continue;

        const candidate = state.candidates[stateIndex];
        if (candidate.count() == 1) {
            stateIndex += 1;
            if (stateIndex == state.candidates.len) break;
            continue;
        }
        const row = i / desc.n_cols;
        const col = i % desc.n_cols;
        const old = blk: {
            if (propagations) |ps| {
                break :blk if (ps.get(stateIndex)) |p| p.old else null;
            } else break :blk null;
        };
        drawCandidatesForCell(col + 1, row + 1, candidate, old);
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawCandidatesForCell(x: usize, y: usize, candidates: Candidates, old: ?Candidates) void {
    _ = old;
    const fontSize = 20;
    const unit = cellSize / 3;
    var i: u8 = 1;
    const fx = @intToFloat(f32, x * cellSize);
    const fy = @intToFloat(f32, y * cellSize);
    const rec = rl.Rectangle {
        .x = fx,
        .y = fy,
        .width = cellSize,
        .height = cellSize
    };
    const count = candidates.count();
    if (count == 0) {
        rl.DrawRectangleRec(rec, rl.Fade(rl.RED, 0.3));
    }
    var buf: [2:0]u8 = undefined;
    while (i <= max_candidates) : (i += 1) {
        if (candidates.is_candidate(i)) {
            const text = std.fmt.bufPrintZ(&buf, "{d}", .{i}) catch unreachable;
            const nudgeX = ((i - 1) % 3) * unit + 2;
            const nudgeY = ((i - 1) / 3) * unit + 1;
            const innerX = @intCast(c_int, x * cellSize + nudgeX);
            const innerY = @intCast(c_int, y * cellSize + nudgeY);
            rl.DrawText(text.ptr,  innerX, innerY, fontSize, rl.GRAY);
        } else if (!candidates.is_candidate(i) and old != null and old.?.is_candidate(i)) {
            const text = std.fmt.bufPrintZ(&buf, "{d}", .{i}) catch unreachable;
            const nudgeX = ((i - 1) % 3) * unit + 2;
            const nudgeY = ((i - 1) / 3) * unit + 1;
            const innerX = @intCast(c_int, x * cellSize + nudgeX);
            const innerY = @intCast(c_int, y * cellSize + nudgeY);
            rl.DrawText(text.ptr,  innerX, innerY, fontSize, rl.GRAY);
            const innerFx = @intToFloat(f32, x * cellSize + nudgeX);
            const innerFy = @intToFloat(f32, y * cellSize + nudgeY);
            rl.DrawLineEx(
                rl.Vector2 { .x = innerFx, .y = innerFy },
                rl.Vector2 { .x = innerFx + 10, .y = innerFy + 16 },
                3,
                rl.RED,
            );
        }
    }
}

fn drawCandidateIndexes(desc: Description, state: *const State) void {
    var stateIndex: usize = 0;
    const unit = cellSize / 3;
    for (desc.board) |s, i| {
        if (s != 0) continue;

        const x = i % desc.n_cols + 1;
        const y = i / desc.n_cols + 1;

        var buf: [30:0]u8 = undefined;
        const text = std.fmt.bufPrintZ(&buf, "{d}", .{stateIndex}) catch unreachable;
        const nudgeX = 2 * unit - 5;
        const nudgeY = 1;
        const innerX = @intCast(c_int, x * cellSize + nudgeX);
        const innerY = @intCast(c_int, y * cellSize + nudgeY);

        rl.DrawText(text.ptr, innerX, innerY, 12, rl.GRAY);
        stateIndex += 1;
        if (stateIndex == state.candidates.len) break;
    }
}

fn drawNumber(x: usize, y: usize, number: usize) void {
    const fontSize = 40;
    var buf: [2:0]u8 = undefined;
    const text = std.fmt.bufPrintZ(&buf, "{d}", .{number}) catch unreachable;
    const textSize = rl.MeasureTextEx(rl.GetFontDefault(), text.ptr, fontSize, 0);
    const paddingX = @divFloor(@intCast(c_int, cellSize) - @divFloor(@floatToInt(c_int, textSize.x), 2), 2);
    const paddingY = @divFloor(@intCast(c_int, cellSize) - @divFloor(@floatToInt(c_int, textSize.y), 2), 4);
    rl.DrawText(text.ptr, @intCast(c_int, x * cellSize) + paddingX, @intCast(c_int, y * cellSize) + paddingY, fontSize, rl.BLACK);
}

fn drawEmptyCell(x: usize, y: usize) void {
    drawCell(x, y, rl.WHITE);
}

fn drawOutOfBounds(x: usize, y: usize) void {
    drawCell(x, y, rl.BLACK);
}

fn drawCell(x: usize, y: usize, color: rl.Color) void {
    const rec = rectangleForPoint(x, y);
    rl.DrawRectangleRec(rec, color);
    rl.DrawRectangleLinesEx(rec, 1, rl.BLACK);
}

fn rectangleForPoint(x: usize, y: usize) rl.Rectangle {
    return rl.Rectangle {
        .x = @intToFloat(f32, x * cellSize),
        .y = @intToFloat(f32, y * cellSize),
        .width = cellSize,
        .height = cellSize,
    };
}

fn drawNumberedBox(x: usize, y: usize, number: usize, fillLower: bool) void {
    const max = @intToFloat(f32, cellSize);
    const posX = @intToFloat(f32, x * cellSize);
    const posY = @intToFloat(f32, y * cellSize);
    rl.DrawTriangle(
        .{ .x = max + posX, .y = max + posY },
        .{ .x = max + posX, .y = 0   + posY },
        .{ .x = 0   + posX, .y = 0   + posY },
        rl.WHITE,
    );

    // const nudge = 5;
    const centerX = @floatToInt(c_int, (max + max + 0 + posX * 3) / 3);
    const centerY = @floatToInt(c_int, (max + 0 + 0 + posY * 3) / 3);
    var buf: [30:0]u8 = undefined;
    const text = std.fmt.bufPrintZ(&buf, "{d}", .{number}) catch unreachable;
    rl.DrawText(text.ptr, centerX - 5, centerY - 13, 20, rl.BLACK);

    if (fillLower) {
        rl.DrawTriangle(
            .{ .x = max + posX, .y = max + posY },
            .{ .x = 0   + posX, .y = 0   + posY },
            .{ .x = 0   + posX, .y = max + posY },
            rl.BLACK,
        );
    }

    const rec = rectangleForPoint(x, y);
    rl.DrawRectangleLinesEx(rec, 1, rl.BLACK);
}

fn drawNumberedBoxInvert(x: usize, y: usize, number: usize, fillUpper: bool) void {
    const max = @intToFloat(f32, cellSize);
    const posX = @intToFloat(f32, x * cellSize);
    const posY = @intToFloat(f32, y * cellSize);
    if (fillUpper) {
        rl.DrawTriangle(
            .{ .x = max + posX, .y = max + posY },
            .{ .x = max + posX, .y = 0   + posY },
            .{ .x = 0   + posX, .y = 0   + posY },
            rl.BLACK,
        );

    } else {
        rl.DrawLineEx(
            rl.Vector2 { .x = posX, .y = posY },
            rl.Vector2 { .x = posX + max, .y = posY + max },
            3,
            rl.BLACK,
        );
    }

    rl.DrawTriangle(
        .{ .x = max + posX, .y = max + posY },
        .{ .x = 0   + posX, .y = 0   + posY },
        .{ .x = 0   + posX, .y = max + posY },
        rl.WHITE,
    );

    const centerY = @floatToInt(c_int, (max + max + 0 + posY * 3) / 3);
    var buf: [30:0]u8 = undefined;
    const text = std.fmt.bufPrintZ(&buf, "{d}", .{number}) catch unreachable;
    rl.DrawText(text.ptr, @intCast(c_int, x * cellSize) + 5, centerY - 5, 20, rl.BLACK);

    const rec = rectangleForPoint(x, y);
    rl.DrawRectangleLinesEx(rec, 1, rl.BLACK);
}


// ==================================== MAIN ====================================
// ==============================================================================
//

fn solve(allocator: Allocator, descriptions: []const Description) !void {
    const kakuros = try createKakuros(allocator, descriptions);
    var runner = try Runner.init(allocator, kakuros, {});
    try runner.runAll();
    // try runner.report();
}

pub fn main() !void {
    std.log.info("mode {}", .{build_options.mode});

    const argv = std.os.argv;
    var boards_path: []const u8 = "boards.txt";
    if (argv.len > 1) {
        if (std.mem.eql(u8, std.mem.span(argv[1]), "--boards")) {
            boards_path = std.mem.span(argv[2]);
        }
    }

    const allocator = std.heap.c_allocator;
    const descriptions = try parse_descriptions(allocator, boards_path);

    switch (build_options.mode) {
        .gui => try runGui(allocator, descriptions),
        .solve => try solve(allocator, descriptions),
    }
}
