import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

class TransformApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Householder and Givens Transformations")

        self.n = tk.IntVar(value=2)
        self.transform = tk.StringVar(value="Householder")
        self.method = tk.StringVar(value="Method2")

        self.figure = None
        self.canvas = None

        opt_frame = ttk.LabelFrame(self, text="Options")
        opt_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(opt_frame, text="Dimension n:").grid(row=0, column=0, sticky=tk.W)
        n_menu = ttk.OptionMenu(opt_frame, self.n, self.n.get(), 2, 3, command=lambda _: self.update_inputs())
        n_menu.grid(row=0, column=1, sticky=tk.W)

        ttk.Label(opt_frame, text="Transformation:").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(opt_frame, text="Householder", variable=self.transform,
                        value="Householder", command=self.update_inputs).grid(row=1, column=1, sticky=tk.W)
        ttk.Radiobutton(opt_frame, text="Givens", variable=self.transform,
                        value="Givens", command=self.update_inputs).grid(row=1, column=2, sticky=tk.W)

        self.method_frame = ttk.Frame(opt_frame)
        self.method_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W)
        ttk.Label(self.method_frame, text="v⊥ method:").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(self.method_frame, text="Method1", variable=self.method,
                        value="Method1").grid(row=0, column=1, sticky=tk.W)
        ttk.Radiobutton(self.method_frame, text="Method2", variable=self.method,
                        value="Method2").grid(row=0, column=2, sticky=tk.W)

        self.input_frame = ttk.LabelFrame(self, text="Inputs")
        self.input_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.update_inputs()

        self.compute_btn = ttk.Button(self, text="Compute and Plot", command=self.compute_and_plot)
        self.compute_btn.pack(side=tk.TOP, pady=5)

    def update_inputs(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        n = self.n.get()
        trans = self.transform.get()
        row = 0
        
        ttk.Label(self.input_frame, text="Vector x:").grid(row=row, column=0, sticky=tk.W)
        self.x_entries = []
        for i in range(n):
            e = ttk.Entry(self.input_frame, width=10)
            e.grid(row=row, column=1 + i, padx=2)
            self.x_entries.append(e)
            
        row += 1
        if trans == "Householder":
            ttk.Label(self.input_frame, text="k (1 ≤ k ≤ n):").grid(row=row, column=0, sticky=tk.W)
            self.k_spin = tk.Spinbox(self.input_frame, from_=1, to=n, width=5)
            self.k_spin.grid(row=row, column=1, sticky=tk.W)
            self.method_frame.grid()
            
        else:
            ttk.Label(self.input_frame, text="i (1 ≤ i < k ≤ n):").grid(row=row, column=0, sticky=tk.W)
            self.i_spin = tk.Spinbox(self.input_frame, from_=1, to=n-1, width=5, command=lambda: self.update_k_spin())
            self.i_spin.grid(row=row, column=1, sticky=tk.W)
            ttk.Label(self.input_frame, text="k (i < k ≤ n):").grid(row=row, column=2, sticky=tk.W)
            self.k_spin = tk.Spinbox(self.input_frame, from_=2, to=n, width=5)
            self.k_spin.grid(row=row, column=3, sticky=tk.W)
            # Hide method selection
            self.method_frame.grid_remove()

    def update_k_spin(self):
        try:
            i_val = int(self.i_spin.get())
            n = self.n.get()
            self.k_spin.config(from_=i_val+1, to=n)
        except ValueError:
            pass

    def compute_and_plot(self):
        
        try:
            x = np.array([float(e.get()) for e in self.x_entries])
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values for vector x.")
            return
        n = self.n.get()
        
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.figure = Figure(figsize=(5, 5))
        if self.transform.get() == "Householder":
            try:
                k = int(self.k_spin.get()) - 1 
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid k.")
                return
            v, v_perp, x_reflected = self.compute_householder(x, k)
            self.plot_householder(v, v_perp, x_reflected)
        else:
            try:
                i = int(self.i_spin.get()) - 1
                k = int(self.k_spin.get()) - 1
            except ValueError:
                messagebox.showerror("Input Error", "Please enter valid i and k.")
                return
            y, c, s = self.compute_givens(x, i, k)
            self.plot_givens(x, y)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def compute_householder(self, x, k):
     
        norm_x = np.linalg.norm(x)
        e_k = np.zeros_like(x)
        e_k[k] = 1
        v = x - norm_x * e_k
        
        if self.method.get() == "Method1":
            
            if len(x) == 2:
                v_perp = np.array([-v[1], v[0]])
            else:
                
                ref = np.array([1, 0, 0])
                if np.allclose(v / np.linalg.norm(v), ref / np.linalg.norm(ref)):
                    ref = np.array([0, 1, 0])
                v_perp = np.cross(v, ref)
        else:
           
            v_perp = x - (np.dot(x, v) / np.dot(v, v)) * v
        
        x_reflected = x - 2 * (np.dot(v, x) / np.dot(v, v)) * v
        return v, v_perp, x_reflected

    def compute_givens(self, x, i, k):
        
        r = np.hypot(x[i], x[k])
        if r == 0:
            c, s = 1.0, 0.0
        else:
            c = x[i] / r
            s = -x[k] / r
        y = x.copy()
        y[i] = c * x[i] - s * x[k]
        y[k] = s * x[i] + c * x[k]
        return y, c, s

    def plot_householder(self, v, v_perp, x_reflected):
        n = len(v)
        L = np.linalg.norm(v) * 1.5
        
        
        norm_x = np.linalg.norm(v + np.linalg.norm(v) * np.eye(1, n, k=int(self.k_spin.get())-1).flatten())
        x = v + norm_x * np.eye(1, n, k=int(self.k_spin.get())-1).flatten()

        if n == 2:
            ax = self.figure.add_subplot(111)
            t = np.linspace(-L, L, 100)
            
            ax.plot(v[0] * t / np.linalg.norm(v), v[1] * t / np.linalg.norm(v), 
                    'g-', label='Sp{v} (Reflection direction)')
            
            ax.plot(v_perp[0] * t / np.linalg.norm(v_perp), v_perp[1] * t / np.linalg.norm(v_perp), 
                    'r--', label='Sp{v⊥} (Reflection hyperplane)')
            
            ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1,
                    color='blue', width=0.005, label='Original x')
            ax.quiver(0, 0, x_reflected[0], x_reflected[1], angles='xy', scale_units='xy', scale=1,
                    color='orange', width=0.005, label='Reflected x')
            
            ax.scatter([0,], [0,], color='black')
            ax.axis('equal')
            ax.legend()
            ax.set_title('Householder Reflection in 2D')
            
        else:
            ax = self.figure.add_subplot(111, projection='3d')
            t = np.linspace(-L, L, 50)
            
            ax.plot(v[0] * t / np.linalg.norm(v), v[1] * t / np.linalg.norm(v), v[2] * t / np.linalg.norm(v),
                    'g-', label='Sp{v} (Reflection direction)')
            
            ax.plot(v_perp[0] * t / np.linalg.norm(v_perp), v_perp[1] * t / np.linalg.norm(v_perp),
                    v_perp[2] * t / np.linalg.norm(v_perp), 'r--', label='Sp{v⊥} (Reflection plane)')
            
            ax.quiver(0, 0, 0, x[0], x[1], x[2], color='blue', label='Original x', length=1, normalize=True)
            ax.quiver(0, 0, 0, x_reflected[0], x_reflected[1], x_reflected[2], 
                    color='orange', label='Reflected x', length=1, normalize=True)
            
            ax.set_box_aspect([1,1,1])
            ax.legend()
            ax.set_title('Householder Reflection in 3D')

    def plot_givens(self, x, y):
        n = len(x)
        if n == 2:
            ax = self.figure.add_subplot(111)
            ax.quiver(0, 0, x[0], x[1], angles='xy', scale_units='xy', scale=1, 
                    color='blue', label='x', width=0.005)
            ax.quiver(0, 0, y[0], y[1], angles='xy', scale_units='xy', scale=1, 
                    color='red', label='y', width=0.005)
            lim = np.max(np.abs(np.concatenate([x, y]))) * 1.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            ax.set_title('The Givens Transformation 2D')
            ax.axis('equal')
        else:
            ax = self.figure.add_subplot(111, projection='3d')
            
            max_val = np.max(np.abs(np.concatenate([x, y]))) * 1.5
            lims = (-max_val, max_val)
            
            ax.quiver(0, 0, 0, x[0], x[1], x[2], 
                    color='blue', label='x', arrow_length_ratio=0.1, linewidth=2)
            ax.quiver(0, 0, 0, y[0], y[1], y[2], 
                    color='red', label='y', arrow_length_ratio=0.1, linewidth=2)
            
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_zlim(lims)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            ticks = np.linspace(-max_val, max_val, 5)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_zticks(ticks)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_box_aspect([1, 1, 1])
            
            ax.legend()
            ax.set_title('The Givens Transformation 3D')
            
            ax.view_init(elev=20, azim=30)

if __name__ == '__main__':
    app = TransformApp()
    app.mainloop()
