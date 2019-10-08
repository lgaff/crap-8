# crap-8
## The shitty Python Chip-8 emulator.

This is a Chip-8 emulator created using Python and Pygame for demonstration purposes.

## Usage

Run the emulator by calling the script:

`python crap8.py /path/to/CHIP8_BINARY`

Some additional parameters are possible as well, see `--help` for more details.

## Controls

Keys are mapped to a 4 x 4 grid on a standard QWERTY keyboard to simulate the layout of a Chip-8 machine.
Starting from the `1` key and moving across, this maps the Chip-8 key against it's corresponding keyboard key:

```text
+-----+-----+-----+-----+
| 1/1 | 2/2 | 3/3 | C/4 |
+-----+-----+-----+-----+
| 4/Q | 5/W | 6/E | D/R |
+-----+-----+-----+-----+
| 7/A | 8/S | 9/D | E/F |
+-----+-----+-----+-----+
| A/Z | 0/X | B/C | F/V |
+-----+-----+-----+-----+
```

Additionally, when in single-step mode, the **space** key executes one machine cycle. The **p** key un-steps the emulation, returning it to full speed.

The **Esc** key ends the emulation.

## Display

The Display provides - in addition to the Chip-8 display at its top left - a visualisation of the Chip-8 memory and register contents.

To the left of the display will be the memory contents, one square per byte of memory for a total of 4096 bytes. The memory is colored to... *prettify*... the display. The choice of coloring is arbitrary except that the mappings between color and byte are 1:1.

The Register display shows register contents, including both Chip-8 timers.

## Limitations

- Crap-8 does not currently emulate sound. The sound timer exists, but nothing currently happens when it reaches zero.
- Due to the nature of Python, I have yet to find a way to accurately emulate a fixed clock speed. This is nominally meant to be 500hz, however Python's sleep function does not have the necessary resolution to lock the cycle speed to 2ms on most platforms.
- To counter this, some inefficiency was worked back into the design to try and simulate as best as possible a fixed clock speed. The means of doing so is dependent on system IO write speed, and is not likely to be accurately recreated on better systems than the one I have available to me.
