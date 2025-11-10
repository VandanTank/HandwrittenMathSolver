"""
Author: VANDAN TANK
"""

import tkinter as tk
from PIL import ImageGrab
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, sys.path[0] + '\\src')
import model

CANVAS_WIDTH = 1200
CANVAS_HEIGHT = 200
LINE_WIDTH = 20
DEBOUNCE_MS = 300  # delay in ms for updating predictions


class HandwritingCalculator:
    """GUI for real-time handwritten math recognition using Tkinter."""

    def __init__(self):
        """Initialize GUI, model, and background thread executor."""
        self.translator = model.HandwritingTranslator()
        self.root = tk.Tk()
        self.root.title('Handwritten Calculator')

        # Drawing canvas
        self.c = tk.Canvas(self.root, bg='white', width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.c.grid(row=0, column=0)

        # Display label for recognized equation and result
        self.v_calculation = tk.StringVar()
        f1 = tk.Frame(self.root, width=CANVAS_WIDTH)
        f1.grid(row=1, column=0, sticky='w')
        self.label = tk.Label(f1, textvariable=self.v_calculation, font=('Courier', 30))
        self.label.grid(row=0, column=0)
        self.v_calculation.set('Right-click to clear')

        self.last_display = ''
        self._debounce_after_id = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._pred_lock = threading.Lock()

        self.setup()

    def setup(self):
        """Set up event bindings and start Tkinter main loop."""
        self.x_pos = None
        self.y_pos = None
        self.c.bind('<B1-Motion>', self._on_paint_motion)
        self.c.bind('<ButtonRelease-1>', self.finish_draw)
        self.c.bind('<ButtonRelease-3>', self.reset)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.reset(None)
        self.root.mainloop()

    def paint(self, event):
        """Draw on the canvas while mouse is dragged."""
        if self.x_pos and self.y_pos:
            self.c.create_line(
                self.x_pos, self.y_pos, event.x, event.y,
                width=LINE_WIDTH, smooth=tk.TRUE, capstyle=tk.ROUND
            )
        self.x_pos = event.x
        self.y_pos = event.y

    def _on_paint_motion(self, event):
        """Handle stroke drawing and schedule delayed translation."""
        self.paint(event)
        self._schedule_translate_debounced()

    def reset(self, event):
        """Clear the canvas and reset display."""
        self.c.delete('all')
        self.v_calculation.set('Right-click to clear')
        self.last_display = ''
        if self._debounce_after_id:
            try:
                self.root.after_cancel(self._debounce_after_id)
            except Exception:
                pass
            self._debounce_after_id = None

    def finish_draw(self, event):
        """Trigger an immediate prediction when drawing finishes."""
        self.x_pos, self.y_pos = None, None
        if self._debounce_after_id:
            try:
                self.root.after_cancel(self._debounce_after_id)
            except Exception:
                pass
            self._debounce_after_id = None
        self._run_translate_in_background()

    def _schedule_translate_debounced(self):
        """Schedule prediction after short inactivity."""
        if self._debounce_after_id:
            try:
                self.root.after_cancel(self._debounce_after_id)
            except Exception:
                pass
        self._debounce_after_id = self.root.after(DEBOUNCE_MS, self._run_translate_in_background)

    def _run_translate_in_background(self):
        """Capture current canvas and submit recognition in a background thread."""
        x1 = self.c.winfo_rootx() + self.c.winfo_x()
        y1 = self.c.winfo_rooty() + self.c.winfo_y()
        x2 = x1 + CANVAS_WIDTH
        y2 = y1 + CANVAS_HEIGHT
        image = ImageGrab.grab((x1, y1, x2, y2)).convert('L')

        def task(img):
            with self._pred_lock:
                return self.translator.translate(img)

        try:
            future = self._executor.submit(task, image)
            future.add_done_callback(self._on_translate_done)
            self._pending_future = future
        except RuntimeError:
            pass

    def _on_translate_done(self, fut):
        """Handle translation result once prediction completes."""
        try:
            result = fut.result()
        except Exception:
            return

        def do_update():
            if result != getattr(self, 'last_display', ''):
                self.last_display = result
                self.v_calculation.set(result)

        try:
            self.root.after(0, do_update)
        except Exception:
            pass

    def _on_close(self):
        """Gracefully shut down threads and close window."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    HandwritingCalculator()
