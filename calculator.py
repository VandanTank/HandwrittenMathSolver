"""
Author: VANDAN TANK (modified)
Live recognition: debounce + background inference (keeps UI responsive)
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

# debounce delay in milliseconds (adjust to taste: 200-500 ms)
DEBOUNCE_MS = 300


class HandwritingCalculator(object):
    """GUI app for handwritten math expression recognition using tkinter."""

    def __init__(self):
        """Initialize GUI and translator model, plus thread executor."""
        self.translator = model.HandwritingTranslator()
        self.root = tk.Tk()
        self.root.title('Handwritten Calculator')

        # drawing canvas
        self.c = tk.Canvas(self.root, bg='white', width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
        self.c.grid(row=0, column=0)

        # display label
        self.v_calculation = tk.StringVar()
        f1 = tk.Frame(self.root, width=CANVAS_WIDTH)
        f1.grid(row=1, column=0, sticky='w')
        self.label = tk.Label(f1, textvariable=self.v_calculation, font=('Courier', 30))
        self.label.grid(row=0, column=0)
        self.v_calculation.set('Right-click for clear')

        # last displayed text cache (prevents re-writing same string)
        self.last_display = ''

        # debounce scheduling
        self._debounce_after_id = None

        # thread pool for background inference
        # single worker is usually fine; more can be used but ensure model thread-safety.
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._pred_lock = threading.Lock()  # serialize access to model if necessary

        # setup bindings and start mainloop
        self.setup()

    def setup(self):
        """Set initial state and bind mouse events."""
        self.x_pos = None
        self.y_pos = None
        self.c.bind('<B1-Motion>', self._on_paint_motion)
        self.c.bind('<ButtonRelease-1>', self.finish_draw)
        self.c.bind('<ButtonRelease-3>', self.reset)
        # ensure we clean up threads on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.reset(None)
        self.root.mainloop()

    def paint(self, event):
        """Draw on canvas (called by motion handler)."""
        if self.x_pos and self.y_pos:
            self.c.create_line(
                self.x_pos, self.y_pos, event.x, event.y,
                width=LINE_WIDTH, smooth=tk.TRUE, capstyle=tk.ROUND
            )
        self.x_pos = event.x
        self.y_pos = event.y

    # --- new: paint motion handler that also schedules debounce translate
    def _on_paint_motion(self, event):
        # draw the stroke
        self.paint(event)
        # schedule debounce translation (will cancel previous scheduled call)
        self._schedule_translate_debounced()

    def reset(self, event):
        """Clear canvas and reset label/cache."""
        self.c.delete('all')
        self.v_calculation.set('Right-click for clear')
        self.last_display = ''
        # cancel any pending scheduled translate
        if self._debounce_after_id:
            try:
                self.root.after_cancel(self._debounce_after_id)
            except Exception:
                pass
            self._debounce_after_id = None

    def finish_draw(self, event):
        """User released mouse: do an immediate translate (no extra delay)."""
        # clear last positions
        self.x_pos, self.y_pos = None, None
        # cancel scheduled debounce (we'll run immediate)
        if self._debounce_after_id:
            try:
                self.root.after_cancel(self._debounce_after_id)
            except Exception:
                pass
            self._debounce_after_id = None
        # run an immediate translate in background
        self._run_translate_in_background()

    def _schedule_translate_debounced(self):
        """Schedule translate to run DEBOUNCE_MS after last stroke event."""
        # cancel previous schedule
        if self._debounce_after_id:
            try:
                self.root.after_cancel(self._debounce_after_id)
            except Exception:
                pass
        # schedule new call
        self._debounce_after_id = self.root.after(DEBOUNCE_MS, self._run_translate_in_background)

    # --- background inference
    def _run_translate_in_background(self):
        """Grab canvas image and submit translation to executor."""
        # capture the current drawn image region
        x1 = self.c.winfo_rootx() + self.c.winfo_x()
        y1 = self.c.winfo_rooty() + self.c.winfo_y()
        x2 = x1 + CANVAS_WIDTH
        y2 = y1 + CANVAS_HEIGHT
        image = ImageGrab.grab((x1, y1, x2, y2)).convert('L')

        # if a previous future is pending, we can choose to let it complete or cancel (we'll let it complete)
        # submit a new translation task
        # we wrap model call in a safe function that optionally locks the model if needed
        def task(img):
            # acquire lock if model isn't thread-safe; small cost but safer
            with self._pred_lock:
                return self.translator.translate(img)

        # submit to executor
        try:
            future = self._executor.submit(task, image)
            # set callback to handle result when ready (in background thread)
            future.add_done_callback(self._on_translate_done)
            self._pending_future = future
        except RuntimeError:
            # executor already shutdown — ignore
            pass

    def _on_translate_done(self, fut):
        """Callback executed in worker thread when translate finishes.
        We must update Tk UI in the main thread, so use root.after().
        """
        try:
            result = fut.result()
        except Exception as e:
            # swallow and optionally print (don't block UI)
            # print("Translate error:", e)
            return

        # schedule update on Tk main thread
        def do_update():
            if result != getattr(self, 'last_display', ''):
                self.last_display = result
                self.v_calculation.set(result)

        try:
            self.root.after(0, do_update)
        except Exception:
            # window closed or scheduling failed
            pass

    def _on_close(self):
        """Clean shutdown: stop executor, then destroy root window."""
        try:
            # stop accepting new tasks and wait briefly for running tasks to finish
            self._executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass


if __name__ == '__main__':
    HandwritingCalculator()
