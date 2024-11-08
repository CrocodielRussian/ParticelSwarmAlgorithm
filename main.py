import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk as NavigationToolbar2TkAgg
from matplotlib.figure import Figure


class ParticelSwarmAlgorithm:
    def __init__(self, population_size, num_iterations, num_position_axis, speed_rate, inertia_weight_start, inertia_weight_end, max_value, min_value, best_global_value_factor, best_own_value_factor):
        self.population_size = population_size
        self.num_position_axis = num_position_axis
        self.num_iterations = num_iterations
        self.inertia_weight_start = inertia_weight_start
        self.inertia_weight_end = inertia_weight_end
        self.max_value = max_value
        self.min_value = min_value
        self.speed_rate = speed_rate
        self.best_global_value_factor = best_global_value_factor
        self.best_own_value_factor = best_own_value_factor
        self.particles, self.velocities = self.create_population()
        self.personal_best_positions = []


    def get_particles(self):
        return self.particles

    def get_velocities(self):
        return self.velocities

    def eval_func(self, x, y):
        return 8 * x ** 2 + 4 * x * y + 5 * y ** 2

    def create_population(self):
        particles = np.random.uniform(low=self.min_value, high=self.max_value, size=(self.population_size, self.num_position_axis))
        velocities = np.random.uniform(low=self.speed_rate, high=self.speed_rate, size=(self.population_size, self.num_position_axis))

        return particles, velocities
    
    def get_personal_best_positions(self):
        return self.personal_best_positions

    def run(self):
        self.personal_best_positions = self.particles
        personal_best_scores = np.array([self.eval_func(p[0], p[1]) for p in self.particles])
        global_best_position =  self.personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        # Итерации
        for i in range(self.num_iterations):
            # Обновление замедления скорости
            inertia_weight = self.inertia_weight_start - (self.inertia_weight_start - self.inertia_weight_end) * i / self.num_iterations

            # Обновление скорости и позиции частиц
            r1, r2 = np.random.rand(2)
            self.velocities = inertia_weight * self.velocities + self.best_own_value_factor * r1 * ( self.personal_best_positions - self.particles) + self.best_global_value_factor * r2 * (global_best_position - self.particles)
            self.particles = self.particles + self.velocities

            # Обновление личных лучших результатов
            for j in range(self.population_size):
                score = self.eval_func(self.particles[j][0], self.particles[j][1])
                if score < personal_best_scores[j]:
                    self.personal_best_positions[j] = self.particles[j]
                    personal_best_scores[j] = score

            # Обновление глобального лучшего результата
            best_index = np.argmin(personal_best_scores)
            if personal_best_scores[best_index] < global_best_score:
                global_best_position = self.personal_best_positions[best_index]
                global_best_score = personal_best_scores[best_index]

        return global_best_position, global_best_score



class UI:
    def __init__(self, master):
        self.master = master
        self.master.title("Роевой алгоритм")

        self.population_size = tk.IntVar(value=100)
        self.speed_rate = tk.DoubleVar(value=0.2)
        self.iterations_entry = tk.IntVar(value=10)
        self.best_own_value = tk.DoubleVar(value=1.5)
        self.best_global_value = tk.DoubleVar(value=1.5)
        self.iteration_count = tk.IntVar(value=10)
        self.max_value = tk.IntVar(value=10)
        self.min_value = tk.IntVar(value=-10)
        self.inertia_weight_start = tk.DoubleVar(value=0.2)
        self.inertia_weight_end = tk.DoubleVar(value=0.4)
        self.iteration_count_done = 0

        self.create_widgets()

    def create_widgets(self):
        # Фрейм для ввода параметров
        parameter_frame = ttk.LabelFrame(self.master, text="Параметры роевого алгоритма")
        parameter_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(parameter_frame, text="Функция: 8 * x ** 2 + 4 * x * y + 5 * y ** 2").grid(row=0, column=0, sticky="w")

        ttk.Label(parameter_frame, text="Размер популяции:").grid(row=1, column=0, sticky="w")
        population_entry = ttk.Entry(parameter_frame, textvariable=self.population_size)
        population_entry.grid(row=1, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Коэфф. текущей скорости:").grid(row=2, column=0, sticky="w")
        speed_rate_entry = ttk.Entry(parameter_frame, textvariable=self.speed_rate)
        speed_rate_entry.grid(row=2, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Коэфф. собственного лучшего значения:").grid(row=3, column=0, sticky="w")
        best_own_value_entry = ttk.Entry(parameter_frame, textvariable=self.best_own_value)
        best_own_value_entry.grid(row=3, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Коэфф. глобального лучшего значения:").grid(row=4, column=0, sticky="w")
        best_global_value_entry = ttk.Entry(parameter_frame, textvariable=self.best_global_value)
        best_global_value_entry.grid(row=4, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Начальный коэфф. скорости:").grid(row=5, column=0, sticky="w")
        inertia_weight_start_entry = ttk.Entry(parameter_frame, textvariable=self.inertia_weight_start)
        inertia_weight_start_entry.grid(row=5, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Конечный коэфф. скорости:").grid(row=6, column=0, sticky="w")
        inertia_weight_end_entry = ttk.Entry(parameter_frame, textvariable=self.inertia_weight_end)
        inertia_weight_end_entry.grid(row=6, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Минимальное значение:").grid(row=7, column=0, sticky="w")
        inertia_weight_start_entry = ttk.Entry(parameter_frame, textvariable=self.min_value)
        inertia_weight_start_entry.grid(row=7, column=1, sticky="w")

        ttk.Label(parameter_frame, text="Максимальное значение:").grid(row=8, column=0, sticky="w")
        inertia_weight_end_entry = ttk.Entry(parameter_frame, textvariable=self.max_value)
        inertia_weight_end_entry.grid(row=8, column=1, sticky="w")
        
        ttk.Label(parameter_frame, text="Количество итераций:").grid(row=9, column=0, sticky="w")
        iteration_count_entry = ttk.Entry(parameter_frame, textvariable=self.iteration_count)
        iteration_count_entry.grid(row=9, column=1, sticky="w")


        # Фрейм для кнопок
        button_frame = ttk.Frame(self.master)
        button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        start_button = ttk.Button(button_frame, text="Старт", command=self.start_algorithm)
        start_button.grid(row=0, column=0, padx=5)


        stat_frame = ttk.LabelFrame(self.master, text="Результаты")
        stat_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")


        self.place = tk.Canvas(self.master)
        self.f = Figure(figsize=(6,4), dpi=100)
        self.graph_plot = self.f.add_subplot(111)

        self.graph_canvas = FigureCanvasTkAgg(self.f, self.place)
        self.widget = self.graph_canvas.get_tk_widget()
        self.widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2TkAgg(self.graph_canvas, self.place)
        self.toolbar.update()
        self.graph_canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.place.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

    
        # Фрейм для вывода лучшего результата
        best_result_frame = ttk.LabelFrame(self.master, text="Лучший результат")
        best_result_frame.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(best_result_frame, text="Координаты точки:").grid(row=0, column=0, sticky="w")
        self.best_result_label = ttk.Label(best_result_frame, text="")
        self.best_result_label.grid(row=0, column=1, sticky="w")

        ttk.Label(best_result_frame, text="Значение функции:").grid(row=1, column=0, sticky="w")
        self.best_fitness_label = ttk.Label(best_result_frame, text="")
        self.best_fitness_label.grid(row=1, column=1, sticky="w")

        iteration_frame = ttk.Frame(self.master)
        iteration_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(iteration_frame, text="Количество итераций:").grid(row=0, column=0, sticky="w")
        self.iteration_label = ttk.Label(iteration_frame, text=self.iteration_count_done)
        self.iteration_label.grid(row=0, column=1, sticky="w")


    def set_graph(self, population):
        if self.widget:
            self.widget.destroy()

        if self.toolbar:
            self.toolbar.destroy()
        x_coords = [particle[0] for particle in population]
        y_coords = [particle[1] for particle in population]


        self.place = tk.Canvas(self.master)
        self.f = Figure(figsize=(6,4), dpi=100)
        self.graph_plot = self.f.add_subplot(111)
        self.graph_plot.plot(x_coords, y_coords, '.g', markersize=5)

        self.graph_canvas = FigureCanvasTkAgg(self.f, self.place)
        self.widget = self.graph_canvas.get_tk_widget()
        self.widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2TkAgg(self.graph_canvas, self.place)
        self.toolbar.update()
        self.graph_canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.place.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")



    def start_algorithm(self):
        try:
            population_size = self.population_size.get()
            speed_rate = self.speed_rate.get()
            inertia_weight_start = self.inertia_weight_start.get()
            inertia_weight_end = self.inertia_weight_end.get()
            num_iterations = self.iteration_count.get()
            best_own_value = self.best_own_value.get()
            best_global_value = self.best_global_value.get()
            min_value_of_searching = self.min_value.get()
            max_value_of_searching = self.max_value.get()

        except ValueError:
            messagebox.showerror("Ошибка", "В некоторые поля не были переданы аргументы")
            return

        self.iteration_count_done += self.iteration_count.get()

        pa = ParticelSwarmAlgorithm(population_size = population_size, num_iterations = num_iterations, num_position_axis = 2, speed_rate = speed_rate, inertia_weight_start=inertia_weight_start, inertia_weight_end=inertia_weight_end, min_value=min_value_of_searching, max_value=max_value_of_searching, best_global_value_factor=best_global_value, best_own_value_factor=best_own_value)
        
        global_best_position, global_best_score = pa.run()

        personal_best_positions = pa.get_personal_best_positions()
        self.set_graph(personal_best_positions)

        self.update_best_result_label(global_best_position, global_best_score)
        self.update_count_iterations(self.iteration_count_done)

    def update_best_result_label(self, best_individual, best_fitness):
        print(round(best_individual[0].item(), 10), round(best_individual[1].item(), 10))
        print(round(best_fitness, 15))
        self.best_result_label.config(text=f"Координаты точки: {round(best_individual[0].item(), 10), round(best_individual[1].item(), 10)}")
        self.best_fitness_label.config(text=f"Значение функции: {round(best_fitness, 15)}")

    def update_count_iterations(self, iteration_count):
        self.iteration_label.config(text=f"{iteration_count}")


if __name__ == "__main__":
    root = tk.Tk()
    app = UI(root)
    root.mainloop()

