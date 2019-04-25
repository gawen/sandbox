use std::char;

struct Memory {
    m: Vec<u8>,
}

impl Memory {
    fn new() -> Memory {
        Memory { m: Vec::new() }
    }

    fn set(&mut self, addr: usize, value: u8) {
        if self.m.len() <= addr {
            self.m.resize(addr + 1, 0);
        }
        assert!(self.m.len() >= addr);

        self.m[addr] = value;
    }

    fn get(&self, addr: usize) -> u8 {
        if self.m.len() > addr { self.m[addr] } else { 0 }
    }
}

struct BFVM {
    ip: usize,
    dp: usize,
    cs: Vec<usize>,
    code: String,
    mem: Memory,
}

enum StepState {
    Executed,
    BadInstruction { inst: char },
    Finished,
    StackUnderflow,
    UnclosedLoop,
    UnprintableCharacter { c: u8 },
}

impl BFVM {
    fn new(code: String) -> BFVM {
        BFVM {
            ip: 0,
            dp: 0,
            cs: Vec::new(),
            code: code,
            mem: Memory::new(),
        }
    }

    fn read_op(&mut self) -> Option<char> {
        let x = self.code.chars().nth(self.ip);
        self.ip = self.ip + 1;
        x
    }

    fn step(&mut self) -> StepState {
        let c = match self.read_op() {
            Some(c) => c,
            None => return StepState::Finished,
        };

        match c {
            '>' => self.dp = self.dp + 1,
            '<' => {
                if self.dp > 0 {
                    self.dp = self.dp - 1
                }
            }
            '+' => {
                let v = self.mem.get(self.dp);
                self.mem.set(self.dp, v.wrapping_add(1));
            }
            '-' => {
                let v = self.mem.get(self.dp);
                self.mem.set(self.dp, v.wrapping_sub(1));
            }

            '.' => {
                let v = self.mem.get(self.dp);
                match char::from_u32(v as u32) {
                    Some(x) => println!("{}", x), // XXX TODO
                    None => return StepState::UnprintableCharacter { c: v },
                }
            }

            ',' => panic!("TODO"),
            '[' => {
                let v = self.mem.get(self.dp);

                if v == 0 {
                    // look for next ']'
                    let mut level = 1;
                    while level > 0 {
                        let c = match self.read_op() {
                            Some(c) => c,
                            None => return StepState::UnclosedLoop,
                        };

                        match c {
                            '[' => level = level + 1,
                            ']' => level = level - 1,
                            _ => {}
                        };
                    }
                } else {
                    self.cs.push(self.ip - 1);
                }
            }

            ']' => {
                match self.cs.pop() {
                    Some(ip) => {
                        if self.mem.get(self.dp) != 0 {
                            self.ip = ip;
                        }
                    }
                    None => return StepState::StackUnderflow,
                }
            }

            _ => return StepState::BadInstruction { inst: c },
        }

        return StepState::Executed;
    }

    fn run(&mut self) -> StepState {
        loop {
            let x = self.step();

            match x {
                StepState::Finished => return x,
                _ => {}
            }
        }
    }
}

fn main() {
    let mut vm = BFVM::new(
        "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.".to_string(),
    );

    vm.run();
}
