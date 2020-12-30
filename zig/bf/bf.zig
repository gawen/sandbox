//! Brainf**k interpreter, as an exercise to learn Zig.

const std = @import("std");
const Allocator = std.mem.Allocator;
const print = std.debug.print;

var ga = std.heap.GeneralPurposeAllocator(.{}){};
const MAX_SIZE = 64 * 1024 * 1024;

const BFError = error{ NoFilenameGiven, UnfinishedBlock };

pub fn main() !void {
    const allocator = &ga.allocator;

    const code = try readSource(allocator, MAX_SIZE);
    defer allocator.free(code);

    var interp = Interpreter.init(allocator, code);
    defer interp.deinit();

    while (try interp.step()) {}
}

/// Read entire source code from the first argument given to the process.
fn readSource(allocator: *Allocator, max_size: usize) ![]u8 {
    var arg_it = try std.process.argsWithAllocator(allocator);
    defer arg_it.deinit();

    // get bf filename to open
    _ = arg_it.skip(); // skip program name
    const arg_or_err = arg_it.next(allocator) orelse return BFError.NoFilenameGiven;
    const arg = arg_or_err catch |err| return err;

    // read content
    const file = try std.fs.cwd().openFile(arg, .{});
    return try file.reader().readAllAlloc(allocator, max_size);
}

/// Interpreter class.
pub const Interpreter = struct {
    allocator: *Allocator,

    /// source code
    code: []u8,

    /// instruction pointer
    ip: usize,

    /// data pointer
    ptr: usize,

    /// data memory
    mem: std.ArrayList(u8),

    /// block stack
    stack: std.ArrayList(usize),

    pub fn init(allocator: *Allocator, code: []u8) Interpreter {
        return Interpreter{
            .allocator = allocator,
            .code = code,
            .ip = 0,
            .ptr = 0,
            .mem = std.ArrayList(u8).init(allocator),
            .stack = std.ArrayList(usize).init(allocator),
        };
    }

    pub fn deinit(self: Interpreter) void {
        self.mem.deinit();
        self.stack.deinit();
    }

    /// Execute one instruction.
    pub fn step(self: *Interpreter) !bool {
        const inst = self.readInst() orelse return false;

        //std.debug.print("ip: {}, ptr: {}, memory: {}, stack: {}, inst: {}\n", .{ self.ip, self.ptr, self.mem.items.len, self.stack.items.len, inst });

        switch (inst) {
            Instruction.NEXT => {
                self.ptr += 1;
            },
            Instruction.PREV => {
                self.ptr -= 1;
            },
            Instruction.INC => {
                const dPtr = try self.dataPtr();
                _ = @addWithOverflow(u8, dPtr.*, 1, dPtr);
            },
            Instruction.DEC => {
                const dPtr = try self.dataPtr();
                _ = @subWithOverflow(u8, dPtr.*, 1, dPtr);
            },
            Instruction.BEGIN => {
                const dPtr = try self.dataPtr();
                if (dPtr.* != 0) {
                    try self.stack.append(self.ip - 1);
                } else {
                    try self.breakLoop();
                }
            },
            Instruction.END => {
                self.ip = self.stack.pop();
            },
            Instruction.OUT => {
                const dPtr = try self.dataPtr();
                try std.io.getStdOut().writer().writeByte(dPtr.*);
            },
            Instruction.INP => {
                const dPtr = try self.dataPtr();
                dPtr.* = try std.io.getStdIn().reader().readByte();
            },
        }

        return true;
    }

    /// Read one instruction and move IP to the next.
    fn readInst(self: *Interpreter) ?Instruction {
        while (self.ip < self.code.len) {
            if (Instruction.fromChar(self.code[self.ip])) |inst| {
                self.ip += 1;
                return inst;
            }

            self.ip += 1;
        }

        return null;
    }

    /// Returns the pointer to the current data.
    fn dataPtr(self: *Interpreter) !*u8 {
        // ensure memory has enough capacity.
        const capacity = self.ptr + 1;
        try self.mem.ensureCapacity(capacity);

        // initialize all new cells to 0
        while (self.mem.items.len < capacity) {
            const i = self.mem.items.len;
            self.mem.items.len += 1;
            self.mem.items[i] = 0;
        }

        return &self.mem.items[self.ptr];
    }

    /// Break the current loop.
    fn breakLoop(self: *Interpreter) !void {
        var level: usize = 1;
        while (level > 0) {
            const inst = self.readInst() orelse return BFError.UnfinishedBlock;
            switch (inst) {
                Instruction.BEGIN => {
                    level += 1;
                },
                Instruction.END => {
                    level -= 1;
                },
                else => {},
            }
        }
    }
};

/// List of BF instructions.
const Instruction = enum {
    NEXT,
    PREV,
    INC,
    DEC,
    OUT,
    INP,
    BEGIN,
    END,

    pub fn fromChar(char: u8) ?Instruction {
        return switch (char) {
            '>' => Instruction.NEXT,
            '<' => Instruction.PREV,
            '+' => Instruction.INC,
            '-' => Instruction.DEC,
            '.' => Instruction.OUT,
            ',' => Instruction.INP,
            '[' => Instruction.BEGIN,
            ']' => Instruction.END,
            else => null,
        };
    }
};
