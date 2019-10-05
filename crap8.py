# Crap-8, The shitty Chip-8 interpreter.
# Lindsay Gaff <lindsaygaff@gmail.com>

# To the extent possible under law, the person who associated CC0 with
# Crap-8 has waived all copyright and related or neighboring rights
# to Crap-8.

# You should have received a copy of the CC0 legalcode along with this
# work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.

import sys
import logging
import pygame
import random
from datetime import datetime
import argparse

aparser = argparse.ArgumentParser(description="The shitty Chip-8 emulator")
aparser.add_argument('program',
    help="A compiled Chip-8 program to load")
aparser.add_argument('--breakpoint',
    help="A hexadecimal program address at which to pause execution",
    metavar="X",
    nargs="+",
    default=[],
    type=lambda x: int(x, 0))
aparser.add_argument('--debug',
    help="Enable verbose debug logging",
    action="store_true")

logging.basicConfig(level=logging.INFO, filename="crap8-log.txt")

## CONSTANTS ##

# Clock speeds used by Chip-8
TIMER_HZ = 60
CYCLE_HZ = 500

TOTAL_RAM = 4096
LOAD_POS = 0x200

# RAM Display constants
RAM_X = 32 # bytes to display per row
RAM_Y = int(TOTAL_RAM/RAM_X) # total rows to display
RAM_RES = 8

# Chip-8 Video display constants
VIDEO_X = 64
VIDEO_Y = 32
VIDEO_RES = 8
# Pixel colors for display
PIXEL_ON = (255,255,255)
PIXEL_OFF = (64,64,64)

# Resolution of fonts used for register display
REG_FONT_RES = 18
REG_FONT_PAD = 10


# Chip-8 ROM Font map
FONT_LOAD = 0x50
FONT_MAP = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, # 0
    0x20, 0x60, 0x20, 0x20, 0x70, # 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, # 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, # 3
    0x90, 0x90, 0xF0, 0x10, 0x10, # 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, # 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, # 6
    0xF0, 0x10, 0x20, 0x40, 0x40, # 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, # 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, # 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, # A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, # B
    0xF0, 0x80, 0x80, 0x80, 0xF0, # C
    0xE0, 0x90, 0x90, 0x90, 0xE0, # D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, # E
    0xF0, 0x80, 0xF0, 0x80, 0x80  # F
]

# The key map is a little jumbled since the
# Chip-8 has a slightly skewed layout, where
# internal key values are identical to their
# face value in hex.
# I've mapped their equivalents to a grid beginning at key 2 and
# proceeding 4 keys across each row and all the way down

# +-----+-----+-----+-----+
# | 1/1 | 2/2 | 3/3 | C/4 |
# +-----+-----+-----+-----+
# | 4/Q | 5/W | 6/E | D/R |
# +-----+-----+-----+-----+
# | 7/A | 8/S | 9/D | E/F |
# +-----+-----+-----+-----+
# | A/Z | 0/X | B/C | F/V |
# +-----+-----+-----+-----+

KEY_MAP = [
    pygame.K_x, pygame.K_1, pygame.K_2, pygame.K_3,
    pygame.K_q, pygame.K_w, pygame.K_e, pygame.K_a,
    pygame.K_s, pygame.K_d, pygame.K_z, pygame.K_c,
    pygame.K_4, pygame.K_r, pygame.K_f, pygame.K_v
]

# Machine state variables
main_mem = []
vmem = []
reg_I = 0
reg_V = []
reg_PC = 0
stack = [0,0,0,0,0,0,0,0]
reg_SP = 0

# programmable timers used by Chip-8
# Both operate at TIMER_HZ hz.
s_timer = 0
d_timer = 0
# Ticks deducted per wall-second from timers
timer_tps = 1000000 / TIMER_HZ





def main(argv):
    global running
    global d_timer, s_timer
    global reg_PC, reg_V, reg_I
    global main_mem
    global vmem

    running = False
    logging.info("Crap8 - The shitty Chip-8 interpreter")
    args = aparser.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logging.info("Initialise display engine")
    pygame.init()
    pygame.font.init()

    
    

    # RAM display
    # TODO: re-enable this.
    logging.debug(f"Main memory display {RAM_X} by {RAM_Y}, {RAM_X*RAM_RES} x {RAM_Y*RAM_RES} pixels")    

    # Register display
    reg_font = pygame.font.SysFont('Consolas', REG_FONT_RES)
    reg_disp_xpos = 0
    reg_disp_ypos = (VIDEO_Y * VIDEO_RES)
    reg_disp_w = (VIDEO_X * VIDEO_RES)
    reg_disp_h = (5 * REG_FONT_RES) + REG_FONT_PAD
    logging.debug(f"Register display {reg_disp_w} x {reg_disp_h} pixels")

    logging.debug(f"Video memory display {VIDEO_X} by {VIDEO_Y}, {VIDEO_X*VIDEO_RES} x {VIDEO_Y * VIDEO_RES} pixels")

    # Total pygame display size
    screen_x = (VIDEO_X * VIDEO_RES) #  + (RAM_X * RAM_RES)
    screen_y = (VIDEO_Y * VIDEO_RES) + reg_disp_h
    logging.debug(f"Screen x resolution {screen_x}px")
    logging.debug(f"Screen y resolution {screen_y}px")

    
    pygame.display.set_caption("CRAP-8 DISPLAY")
    logging.info(f"Display mode {screen_x} x {screen_y}")
    screen = pygame.display.set_mode([screen_x, screen_y])
    
    main_mem = c_alloc(TOTAL_RAM)
    main_mem[FONT_LOAD:FONT_LOAD+len(FONT_MAP)] = FONT_MAP.copy()
    logging.debug(f"Main memory {TOTAL_RAM:d} bytes initalised")
    logging.debug(f"Fonts loaded to {FONT_LOAD:04x}")
    
    # vmem is a 2-d array of tuples. The first value is the pixel state
    # Second value is whether the pixel has changed in this machine cycle
    # So we can redraw it if necessary
    vmem = ins_cls()
    logging.debug(f"Video memory {VIDEO_X} bytes initialised")

    reg_I = 0
    reg_V = c_alloc(16)
    logging.debug(f"Registers V0:VF, I initialized")

    reg_PC = LOAD_POS
    logging.debug(f"Register PC initialised to 0x{reg_PC:04x}")


    logging.info(f"Loading program {argv[0]} at 0x{reg_PC:04x}")

    with open(args.program, 'rb') as p:
        program = bytearray(p.read())
        logging.info(f"Program length {len(program)} bytes.")
        if len(program) > len(main_mem[reg_PC:]):
            raise Exception("Program is too large.")
        main_mem[reg_PC:reg_PC + len(program)] = program

    logging.info("Emulation starting")

    running = True
    step = False
    do_cycle = True
    
    while running:
        screen.fill((255,255,255), (reg_disp_xpos, reg_disp_ypos, reg_disp_w, reg_disp_h))    
        cycle_start = datetime.now()
        # display_ram(screen)
        display_video(screen)
        display_regs(screen, reg_font)
        pygame.display.flip()
        if reg_PC in args.breakpoint:
            step = True
        if reg_PC < 0 or reg_PC >= len(main_mem): 
            raise Exception("Memory access error")
        # Fetch
        instruction = main_mem[reg_PC] << 8 | main_mem[reg_PC+1]
        if do_cycle: ## Decode & Execute
            if instruction == 0x00E0: # CLS - clear the screen
                vmem = ins_cls()
            elif instruction == 0x00EE: # RET - return from subroutine
                ins_ret()
            elif instruction >> 12 == 0x1: # 0x1nnn JP nnn - Jump to instruction nnn
                nnn = instruction & 0x0FFF
                ins_jmp(nnn)
            elif instruction >> 12 == 0x2: # 0x2nnn CALL - Call subroutine at nnn
                nnn = instruction & 0x0FFF
                ins_call(nnn)
            elif instruction >> 12 == 0x3: # 0x3xkk SE Vx, byte - Skip if Vx == kk
                logging.debug(f"{instruction:04x} {instruction >> 8:04x} {instruction >> 8 & 0x0F:04x}")
                x = instruction >> 8 & 0x0F
                kk = instruction & 0x00FF
                logging.debug(f"{instruction:04x} {x:04x} {kk:04x}")
                ins_skipim(x, kk)
            elif instruction >> 12 == 0x4: # 0x4xkk - SNE Vx, kk
                x = instruction >> 8 & 0x0F
                kk = instruction & 0x00FF
                ins_skipim(x, kk, eq=False)
            elif instruction >> 12 == 0x5:
                unimplemented(instruction)
            elif instruction >> 12 == 0x6: # 0x6xnn LD Vx, nn
                Vx = instruction >> 8 & 0x0F
                nn = instruction & 0x00FF # Oof!
                ins_load(Vx, nn)
            elif instruction >> 12 == 0x7: # 0x7xkk - ADD Vx, kk
                x = instruction >> 8 & 0x0F
                kk = instruction & 0x00FF
                ins_add(x, kk)
            elif instruction >> 12 == 0x8: # 0x8000 instructions handle ALU ops
                x = instruction >> 8 & 0x0F
                y = instruction >> 4 & 0x0F
                op = instruction & 0x000F # Ooof!
                ins_alu(x, op, y)
            elif instruction >> 12 == 0x9: # SNE Vx Vy
                if instruction & 0x01:
                    unimplemented(instruction)
                else:
                    logging.debug(f"SNE {instruction:04x} x {instruction >> 8 & 0x0F:1x} y {instruction >> 4 & 0x0F:1x}")
                    x = instruction >> 8 & 0x0F
                    y = instruction >> 4 & 0x0F
                    ins_skipreg(x, y, eq=False)

            elif instruction >> 12 == 0xA: # 0xAnnn LD I, nnn
                nnn = instruction & 0x0FFF
                ins_loadi(nnn)
            elif instruction >> 12 == 0xB:
                unimplemented(instruction)
            elif instruction >> 12 == 0xC: # 0xCxkk - RND Vx, kk
                x = instruction >> 8 & 0x0F
                kk = instruction & 0x00FF
                ins_rnd(x, kk)
            elif instruction >> 12 == 0xD: # 0xDxyn DRW Vx, Vy, nn
                Vx = instruction >> 8 & 0x0F
                Vy = instruction >> 4 & 0x0F
                nn = instruction & 0x000F
                ins_draw(Vx, Vy, nn)
            elif instruction >> 12 == 0xE: # Key ops
                x = instruction >> 8 & 0x0F
                code = instruction & 0x00FF
                ins_skipkey(x, code)
            elif instruction >> 12 == 0xF: # Other IO
                ins_io(instruction)
            else:
                logging.error(f"Undefined opcode {instruction:04x} at address {reg_PC:04x}")
                exit()

            # Bounds-check the pc, roll it over if out of bounds
            
            reg_PC += 2 % 2**16
        if step:
            do_cycle = False
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running  = False
                if event.key == pygame.K_SPACE:
                    do_cycle = True
                if event.key == pygame.K_p:
                    do_cycle = True
                    step = False
            if event.type == pygame.QUIT:
                running = False
        cycle_end = datetime.now()
        cycle_time = (cycle_end - cycle_start).microseconds
        d_timer = int(timer_update(d_timer, cycle_time))
        s_timer = int(timer_update(s_timer, cycle_time))
    pygame.display.quit()
    logging.info("Emulation halted. Press any key to quit.")

def c_alloc(n):
    """Allocate n byte array as memory"""
    return [0 for i in range(n)]

def bin_print(c):
    """Print a byte's ascii value, or '.' if out of range"""
    if c < 32:
        return '.'
    elif (c == '\r') or (c == '\t'):
        return ' '
    return chr(c)

def dump_ram(map, start=0, count=-1, bytes_per_row=8):
    """pretty-print the contents of memory to stdout"""
    rows = int(len(map[:count]) / bytes_per_row)
    pc = start
    row = 0
    while(row < rows):
        print(f"0x{pc:04X} | ", 
            " ".join([f'0x{map[pc+x]:02x}' for x in range(bytes_per_row)]),
            " | ",
            "".join([bin_print(map[pc+c]) for c in range(bytes_per_row)]))
        pc += bytes_per_row
        row += 1

def dump_registers(V, pc, I):
    """pretty-print the contents of the register file"""
    # Simple debug format, 8 rows of 2 registers
    # Followed by the PC and the index register

    print ("\n".join([f'{f"V{x}":<3}: {V[x]:02X} {f"V{x+1}":<3}: {V[x+1]:02X}' for x in range(0, 16, 2)]))
    print (f"PC: {pc:02X}")
    print (f"I:  {I:04X}")
        
def display_video(screen):
    global vmem
    for y in range (VIDEO_Y):
        for x in range (VIDEO_X):
            # logging.debug(f"{x}, {y})")
            px = vmem[y][x]
            if px[1]:
                if px[0]:
                    color = PIXEL_ON
                else:
                    color = PIXEL_OFF
                pygame.draw.rect(screen, color, (x*VIDEO_RES,y*VIDEO_RES,VIDEO_RES,VIDEO_RES))
                vmem[y][x] = (px[0], False)

def display_ram(screen):
    """Update the ram display with contents of memory"""
    for y in range(RAM_Y):
        for x in range(RAM_X):
            cell = y * RAM_X + x
        
            byte = main_mem[cell]
            r = (byte >> 5 & 0x07) << 5
            g = (byte >> 2 & 0x07) << 5
            b = (byte & 0x03) << 5
            pygame.draw.rect(screen, (r,g,b), (VIDEO_X*VIDEO_RES+x*RAM_RES,y*RAM_RES,RAM_RES,RAM_RES))

def display_regs(screen, font):
    line_off = font.size("V")[1]
    disp = ""
    # screen.fill((255,255,255))
    for x in range(0, 16, 4):
        disp = ""
        for r in range(4):
            disp += f"V{x+r:1X}: 0x{reg_V[x+r]:04x} "
        ts = font.render(disp, False, (0,0,0))
        
        screen.blit(ts, (0, (VIDEO_Y * VIDEO_RES) + line_off * (x / 4)))

    disp = f"PC: 0x{reg_PC:04x} I:  0x{reg_I:04x} DT: 0x{d_timer:04x} ST: 0x{s_timer:04x}"
    ts = font.render(disp, False, (0,0,0))
    screen.blit(ts, (0,(VIDEO_Y * VIDEO_RES) + line_off * 4))

def timer_update(timer, ms):
    """Remove ms from timer according to TIMER_HZ
       Treats timer as a floating point value. Since chip-8 timers are expected to be
       integers, bear this fact in mind if you need to display it"""
    
    nt = timer - ms/(1000000 / TIMER_HZ)
    if nt > 0:
        return nt
    else:
        return 0

def ins_cls():
    """Handle instruction 0x00e0 CLS - Clear the screen"""
    return [[(0, True) for x in range(VIDEO_X)] for y in range(VIDEO_Y)]

def ins_ret():
    global reg_PC
    global reg_SP
    logging.info(f"{reg_PC:04x} | OP 0x00EE - RET")
    logging.info(f"")
    reg_PC = stack[reg_SP]
    reg_SP -= 1
    if reg_SP < 0:
        logging.error("Stack underflow.")

    
def unimplemented(instruction):
    """Handle unimplemented instructions. Likely unnecessary in final release"""
    global reg_PC, running
    logging.info(f"{reg_PC:04x} | OP 0x{instruction:04x} - Unimplemented")
    running = False

def ins_jmp(n):
    """Handle JP instruction"""
    global reg_PC
    logging.info(f"{reg_PC:04x} | OP 0x1{n:03x} - JP {n:03x}")
    reg_PC = n - 2 # TODO: This is a shitty hack. Fix it.

def ins_call(n):
    """Handle CALL instruction"""
    global stack
    global reg_PC
    global reg_SP
    logging.info(f"{reg_PC:04x} | OP 0x2{n:03x} - CALL {n:03x}")
    logging.info("")
    reg_SP += 1
    if reg_SP >= len(stack):
        logging.error("Stack overflow")
        exit()
    else:
        stack[reg_SP] = reg_PC
        reg_PC = n - 2 # TODO: This is a shitty hack. Fix it.



def ins_skipim(x, kk, eq=True):
    global reg_PC
    if eq:
        logging.info(f"{reg_PC:04x} | OP 0x3{x:01x}{kk:02x} - SE V{x:1x}, {kk:02x} ({reg_V[x]:02x})")
        if reg_V[x] == kk:
            reg_PC += 2
    else:
        logging.info(f"{reg_PC:04x} | OP 0x4{x:01x}{kk:02x} - SNE V{x:1x}, {kk:02x} ({reg_V[x]:02x})")
        if reg_V[x] != kk:
            reg_PC += 2

def ins_load(reg, n):
    global reg_V
    logging.info(f"{reg_PC:04x} | OP 0x6{reg:1x}{n:02x} - LD V{reg:1x}, {n:02x}")
    reg_V[reg] = n

def ins_add(x, kk):
    logging.info(f"{reg_PC:04x} | OP 0x7{x:01x}{kk:02x} - ADD V{x:1x}, {kk:02x} ({reg_V[x]} + {kk})")
    reg_V[x] += kk
    reg_V[x] &= 0xFF

def ins_alu(x, op, y):
    global running
    global reg_V
    reg_V[0xF] = 0
    if op == 0x0:
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - LD V{x:1x}, V{y:1x} (V{x:1x} <- {reg_V[y]:02x})")
        reg_V[x] = reg_V[y]
    elif op == 0x1:
        # OR Vx, Vy
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - OR V{x:1x}, V{y:1x} ({reg_V[x]:02x} | {reg_V[y]:02x} = {reg_V[x] | reg_V[y]:04x})")
        reg_V[x] = reg_V[x] | reg_V[y]
    elif op == 0x2:
        # AND Vx, Vy
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - AND V{x:1x}, V{y:1x} ({reg_V[x]:02x} & {reg_V[y]:02x} = {reg_V[x] & reg_V[y]:04x})")
        reg_V[x] = reg_V[x] & reg_V[y]
    elif op == 0x3:
        # XOR Vx, Vy
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - XOR V{x:1x}, V{y:1x} ({reg_V[x]:02x} ^ {reg_V[y]:02x} = {reg_V[x] ^ reg_V[y]:04x})")
        reg_V[x] = reg_V[x] ^ reg_V[y]
    elif op == 0x4:
        # ADD Vx, Vy, set carry flag if > 255. Truncate to 8 bits.
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - ADD V{x:1x}, V{y:1x} ({reg_V[x]:02x} + {reg_V[y]:02x} = {reg_V[x] + reg_V[y]:04x})")
        result = reg_V[x] + reg_V[y]
        if result > 0xFF:
            reg_V[0xF] = 1
        reg_V[x] = result & 0xFF
    elif op == 0x5:
        # SUB Vx, Vy. Set the flag if Vx > Vy. Truncate to 8 bits.
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - SUB V{x:1x}, V{y:1x} ({reg_V[x]:02x} - {reg_V[y]:02x} = {reg_V[x] - reg_V[y]:04x})")
        if reg_V[x] > reg_V[y]:
            reg_V[0xF] = 1
        reg_V[x] = (reg_V[x] - reg_V[y]) & 0xFF
    elif op == 0x6:
        # SHR Vx. If lsb of Vx is 1, set VF.
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - SHR V{x:1x}, V{y:1x} ({reg_V[x]:02x} >> {reg_V[y]:02x} = {reg_V[x] >> reg_V[y]:04x}))")
        reg_V[0xF] = reg_V[x] % 2
        reg_V[x] = reg_V[x] >> 1 
    elif op == 0x7:
        # SUBN Vx, Vy
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - SUBN V{x:1x}, V{y:1x} ({reg_V[y]:02x} - {reg_V[x]:02x} = {reg_V[y] - reg_V[x]:04x}))")
        if reg_V[y] > reg_V[x]:
            reg_V[0xF] = 1
        reg_V[x] = (reg_V[y] - reg_V[x]) & 0xFF
    elif op == 0xE:
        # SHL Vx, Vy
        logging.info(f"{reg_PC:04x} | OP 0x8{x:01x}{y:1x}{op:1x} - SHL V{x:1x}, V{y:1x} ({reg_V[x]:02x} << {reg_V[y]:02x} = {reg_V[x] << reg_V[y]:04x}))")
        reg_V[0xF] = (reg_V[x] >> 8) & 0x1
        reg_V[x] = (reg_V[x] << 1) & 0xFF
    else:
        logging.error(f"Invalid instruction at {reg_PC:04x}: {op:04x}")
        running = False
    

def ins_skipreg(x, y, eq=True):
    global reg_PC
    if eq:
        logging.info(f"{reg_PC:04x} | OP 0x0{x:01x}{y:1x}0 - SE V{x:1x}, V{y:1x} ({reg_V[x]:02x}, {reg_V[y]:02x})")
        if reg_V[x] == reg_V[y]:
            reg_PC += 2
    else:
        logging.info(f"{reg_PC:04x} | OP 0x9{x:01x}{y:1x}0 - SNE V{x:1x}, V{y:1x} ({reg_V[x]:02x}, {reg_V[y]:02x})")
        if reg_V[x] != reg_V[y]:
            reg_PC += 2    

def ins_loadi(n):
    global reg_I
    logging.info(f"{reg_PC:04x} | OP 0xA{n:03x} - LD I, {n:03x}")
    reg_I = n

def ins_rnd(x, kk):
    global reg_V
    logging.info(f"{reg_PC:04x} | OP 0xC{x:01x}{kk:02x} - RND V{x:1x}, {kk:02x}")
    randalthor = random.randint(0, 255) & kk
    reg_V[x] = randalthor


def ins_draw(x, y, n):
    global reg_V
    global vmem
    logging.info(f"{reg_PC:04x} | OP 0xD{x:1x}{y:1x}{n:1x} - DRW V{x:1x}, V{y:1x}, {n:1x} ({reg_V[x]:02x} {reg_V[y]:02x})")
    # Draw sprite at location (x, y) into vmem
    # Each byte in memory at [I] represents one line of the sprite.
    # Pixels are binary and the value should be xor'd with the incoming pixel
    # for collision detection.
    reg_V[0xF] = 0
    logging.debug(f"Memory loc {reg_I:04x}, {n} bytes:")
    logging.debug(len(main_mem))
    for cell in range(n):
        logging.debug(f"{main_mem[reg_I + cell]:08b} | {main_mem[reg_I + cell]:02x}")
    for row in range(n):
        ypos = (reg_V[y] + row) % VIDEO_Y
        for col in range(8):
            xpos = (reg_V[x] + col) % VIDEO_X

            px = main_mem[reg_I + row] >> (7 - col) & 0x1
            logging.debug(f"pixel ({xpos}, {ypos}) in {px} out {vmem[ypos][xpos][0]} :: {vmem[ypos][xpos][0] ^ px}")
            if px + vmem[ypos][xpos][0] == 2:
                logging.debug("set collision")
                reg_V[0xF] = 1
            
            vmem[ypos][xpos] = (vmem[ypos][xpos][0] ^ px, True)


def ins_skipkey(x, code):
    global reg_PC
    global running
    if code == 0x9E:
        logging.info(f"{reg_PC:04x} | OP 0xE{x:1x}9E - SKP V{x:1x} ({reg_V[x]:02x})")
        if pygame.key.get_pressed()[KEY_MAP[reg_V[x]]]:
            reg_PC += 2
    elif code == 0xA1:
        logging.info(f"{reg_PC:04x} | OP 0xE{x:1x}A1 - SKNP V{x:1x} ({reg_V[x]:02x})")
        logging.debug(f"Key {KEY_MAP[reg_V[x]]}")
        if not pygame.key.get_pressed()[KEY_MAP[reg_V[x]]]:
            reg_PC += 2
    else:
        logging.error(f"Invalid instruction at {reg_PC:04x}: {code:04x}")
        running = False

def ins_io(op):
    global reg_V
    global reg_PC
    global running
    global d_timer
    global s_timer
    global main_mem
    global reg_I
    code = op & 0x00FF
    x = op >> 8 & 0x0F
    if code == 0x07:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD V{x:1x}, DT")
        reg_V[x] = d_timer
    elif code == 0x0A:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD V{x:1x}, K")
        # Halt execution and wait for a keypress. Store the key in Vx
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            logging.debug(f"Event found {event.type}")
            if (event.type == pygame.KEYDOWN) and (event.key in KEY_MAP):
                logging.debug(f"Store event {event.key} to V{x:1x}")
                reg_V[x] = KEY_MAP.index(event.key)
                break
    elif code == 0x15:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD DT, V{x:1x}")
        d_timer = reg_V[x]
    elif code == 0x18:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD ST, V{x:1x}")
        s_timer = reg_V[x]
    elif code == 0x1E:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - ADD I, V{x:1x}")
        reg_I += reg_V[x] 
        reg_I &= 0xFFFF
    elif code == 0x29:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD F, V{x:1x}")
        if reg_V[x] < 0 or reg_V[x] > len(FONT_MAP):
            logging.error("Invalid font designator at {reg_PC:04x}: {reg_V[x]:04x}")
            running = False
        else:
            reg_I = FONT_LOAD + 5 * reg_V[x]
    elif code == 0x33:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD B, V{x:1x}")
        hundreds = int(reg_V[x] / 100)
        tens = int((reg_V[x] - 100 * hundreds) / 10)
        digits = reg_V[x] % 10
        main_mem[reg_I] = hundreds
        main_mem[reg_I+1] = tens
        main_mem[reg_I+2] = digits

    elif code == 0x55:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD [I], V{x:1x}")
        for n in range(x):
            main_mem[reg_I+n] = reg_V[n]
    elif code == 0x65:
        logging.info(f"{reg_PC:04x} | OP 0xF{x:1x}{code:02x} - LD V{x:1x}, [I]")
        logging.debug(f"I: {reg_I:04x}")
        for n in range(x+1):
            logging.debug(f"V{n:1x} -> {main_mem[reg_I+n]:02x}")
            reg_V[n] = main_mem[reg_I+n]
    else:
        logging.error(f"Invalid instruction at {reg_PC:04x}: {op:04x}")
        running = False
    
        
    



if __name__ == "__main__":
    main(sys.argv[1:])