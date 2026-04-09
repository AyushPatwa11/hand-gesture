import tkinter as tk
from tkinter import ttk
import threading
import shared_state


def _update_status(label):
    running = shared_state.get("running")
    bloom = shared_state.get("bloom")
    sound = shared_state.get("sound")
    label.config(text=f"Running: {running} | Bloom: {bloom} | Sound: {sound}")
    label.after(500, _update_status, label)


def run_gui():
    root = tk.Tk()
    root.title("IronMan Controls")

    frm = ttk.Frame(root, padding=12)
    frm.grid()

    status = ttk.Label(frm, text="Starting...")
    status.grid(column=0, row=0, columnspan=2, pady=(0, 8))

    def toggle_bloom():
        shared_state.toggle("bloom")
        _update_status(status)

    def toggle_sound():
        shared_state.toggle("sound")
        _update_status(status)

    def do_save():
        shared_state.request_save()

    def do_clear():
        shared_state.request_clear()

    def toggle_running():
        cur = shared_state.get("running")
        shared_state.set("running", not cur)
        _update_status(status)

    bloom_btn = ttk.Button(frm, text="Toggle Bloom", command=toggle_bloom)
    bloom_btn.grid(column=0, row=1, sticky="ew", padx=4, pady=4)

    sound_btn = ttk.Button(frm, text="Toggle Sound", command=toggle_sound)
    sound_btn.grid(column=1, row=1, sticky="ew", padx=4, pady=4)

    save_btn = ttk.Button(frm, text="Save Image", command=do_save)
    save_btn.grid(column=0, row=2, sticky="ew", padx=4, pady=4)

    clear_btn = ttk.Button(frm, text="Clear Canvas", command=do_clear)
    clear_btn.grid(column=1, row=2, sticky="ew", padx=4, pady=4)

    run_btn = ttk.Button(frm, text="Start/Stop Capture", command=toggle_running)
    run_btn.grid(column=0, row=3, columnspan=2, sticky="ew", padx=4, pady=8)

    quit_btn = ttk.Button(frm, text="Exit", command=root.destroy)
    quit_btn.grid(column=0, row=4, columnspan=2, sticky="ew", padx=4, pady=4)

    _update_status(status)
    root.mainloop()


if __name__ == "__main__":
    run_gui()
