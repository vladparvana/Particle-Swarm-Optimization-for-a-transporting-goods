import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
import heapq
import testing
from typing import List, Tuple

"""
PROIECT IA - 2024
PARVANA VLAD-STEFAN
RADION CIPRIAN
"""
class CourierSystem:
    """
    Clasa ce implementeaza sistemul de transport a unei firme de curierat utilizand
    A* pentru gasirea drumurilor si Partiecle Swarm Optimization pentru determinarea ordinii optime
    a vizitarii oraselor + Random Key Encoding
    """
    def __init__(self, root):
        """
        Inițializeaza interfața grafica și datele de baza ale sistemului.
        """
        self.root = root
        self.root.title("Sistem Optimizare Transport")


        # Coordonatele orașelor principale (longitudine, latitudine)
        self.cities = {
            'București': (44.4268, 26.1025),
            'Cluj-Napoca': (46.7712, 23.6236),
            'Timișoara': (45.7489, 21.2087),
            'Iași': (47.1585, 27.6014),
            'Constanța': (44.1598, 28.6348),
            'Brașov': (45.6579, 25.6012),
            'Craiova': (44.3302, 23.7949),
            'Galați': (45.4353, 28.0080),
            'Oradea': (47.0465, 21.9189),
            'Sibiu': (45.7929, 24.1252)
        }

        # Definim drumurile inițiale intr-o singura directie
        initial_roads = {
            'București': {
                'Brașov': 170,
                'Constanța': 225,
                'Craiova': 230
            },
            'Cluj-Napoca': {
                'Oradea': 152,
                'Sibiu': 175
            },
            'Timișoara': {
                'Oradea': 170,
                'Sibiu': 290
            },
            'Iași': {
                'Galați': 260,

            },
            'Constanța': {
                'București': 225,
                'Galați': 190
            },
            'Brașov': {
                'București': 170,
                'Sibiu': 145
            },
            'Craiova': {
                'București': 230,
                'Sibiu': 235
            },
            'Galați': {
                'Iași': 260,
                'Constanța': 190
            },
            'Oradea': {
                'Cluj-Napoca': 152,
                'Timișoara': 170
            },
            'Sibiu': {
                'Cluj-Napoca': 175,
                'Timișoara': 290,
                'Brașov': 145,
                'Craiova': 235
            }
        }

        # Initializam dictionarul roads care va stoca drumurile în ambele direcții
        self.roads = {}
        for city in self.cities:
            self.roads[city] = {}

        # Adăugam fiecare drum O SINGURA DATA pentru fiecare pereche de orașe
        for city1, connections in initial_roads.items():
            for city2, distance in connections.items():
                # Verificam dacă drumul exista deja în cealalta directie
                if city2 in self.roads and city1 in self.roads[city2]:
                    continue
                # Adaugam drumul in ambele directii
                self.roads[city1][city2] = distance
                if city2 not in self.roads:
                    self.roads[city2] = {}
                self.roads[city2][city1] = distance

        self.headquarters = set()  # Set pentru stocarea sediilor
        self.setup_gui()  # Configurarea interfetei grafice

    def setup_gui(self):
        """
        Configureaza si afișeaza elementele interfetei grafice, cum ar fi harta, zona de comanda,
        butoanele si listele.
        """
        # Frame pentru harta
        self.map_frame = tk.Frame(self.root, bg='green', width=600, height=400)
        self.map_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas pentru harta
        self.map_canvas = tk.Canvas(self.map_frame, bg='white')
        self.map_canvas.pack(fill=tk.BOTH, expand=True)

        # Inccrcarea și afisarea harții
        self.load_map()

        # Frame pentru comanda
        self.order_frame = tk.Frame(self.root, bg='blue', width=400)
        self.order_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

        # Buton pentru adaugare sediu
        self.add_hq_button = tk.Button(self.map_frame, text="Adaugă Sediu", command=self.enable_hq_placement)
        self.add_hq_button.pack(pady=5)

        # Combobox pentru selectare sediu plecare
        self.hq_label = tk.Label(self.order_frame, text="Sediu Plecare:")
        self.hq_label.pack(pady=5)
        self.hq_select = ttk.Combobox(self.order_frame, state='readonly')
        self.hq_select.pack(pady=5)
        self.hq_select.bind('<<ComboboxSelected>>', self.update_destination_cities)

        # Listbox pentru orașe destinatie
        self.dest_label = tk.Label(self.order_frame, text="Orașe Destinație:")
        self.dest_label.pack(pady=5)
        self.dest_listbox = tk.Listbox(self.order_frame, selectmode=tk.MULTIPLE)
        self.dest_listbox.pack(pady=5)

        # Buton pentru calculare ruta
        self.calc_button = tk.Button(self.order_frame, text="Calculează Ruta", command=self.calculate_route)
        self.calc_button.pack(pady=10)

        # Label pentru afisare rezultate
        self.result_label = tk.Label(self.order_frame, text="", wraplength=350)
        self.result_label.pack(pady=10)

    def update_destination_cities(self, event=None):
        """
        Actualizează lista de orașe destinație, incluzând sediile, cu excepția sediului de plecare.
        """
        selected_hq = self.hq_select.get()
        self.dest_listbox.delete(0, tk.END)  # Ștergem toate elementele din listbox

        # Adaugam toate orasele și sediile, cu excepția sediului de plecare
        all_destinations = set(self.cities.keys()) | self.headquarters
        all_destinations.remove(selected_hq)

        for city in sorted(all_destinations):
            # Marcam sediile cu un prefix special pentru a le evidenția
            if city in self.headquarters:
                self.dest_listbox.insert(tk.END, f"[SEDIU] {city}")
            else:
                self.dest_listbox.insert(tk.END, city)

    def load_map(self):
        """
        Încărcă și afișează harta (un dreptunghi alb simplificat) și poziționează orașele pe hartă.
        """
        self.map_canvas.create_rectangle(50, 50, 550, 350, fill='white')

        # Plasam orasele pe harta
        for city, coords in self.cities.items():
            x = 50 + (coords[1] - 20) * 40
            y = 50 + (49 - coords[0]) * 40
            self.map_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill='black')
            self.map_canvas.create_text(x, y - 10, text=city, font=('Arial', 8))

    def enable_hq_placement(self):
        """
        Activeaza modul de plasare a sediilor, asociind functia `place_headquarters` cu evenimentul de click pe harta
        """
        self.map_canvas.bind('<Button-1>', self.place_headquarters)

    def place_headquarters(self, event):
        """
        Plasează un sediu pe harta, determinând cel mai apropiat oras de coordonatele click-ului
        """
        # Gasim cel mai apropiat oras de click
        click_x, click_y = event.x, event.y
        closest_city = min(self.cities.items(),
                           key=lambda x: math.sqrt((50 + (x[1][1] - 20) * 40 - click_x) ** 2 +
                                                   (50 + (49 - x[1][0]) * 40 - click_y) ** 2))

        if closest_city[0] not in self.headquarters:
            self.headquarters.add(closest_city[0])
            x = 50 + (closest_city[1][1] - 20) * 40
            y = 50 + (49 - closest_city[1][0]) * 40
            self.map_canvas.create_rectangle(x - 5, y - 5, x + 5, y + 5, fill='red')

            # Actualizam combobox-ul cu sediile
            self.hq_select['values'] = tuple(sorted(self.headquarters))
            if len(self.headquarters) == 1:
                self.hq_select.set(closest_city[0])

            # Actualizam lista de destinatii
            self.update_destination_cities()

    def get_clean_city_name(self, city: str) -> str:
        """
        Elimina prefixul [SEDIU] dacă exista, pentru a putea accesa datele din dictionarul de orase
        """
        return city.replace("[SEDIU] ", "") if city.startswith("[SEDIU]") else city

    def calculate_route(self):
        """
        Calculeaza și afișeaza ruta optima, inclusiv orașele intermediare, folosind PSO pentru ordinea orașelor
        și A* pentru determinarea rutelor între ele.
        """
        start_city = self.hq_select.get()
        if not start_city:
            messagebox.showerror("Eroare", "Selectati un sediu de plecare!")
            return

        selected_indices = self.dest_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Eroare", "Selectati cel puțin un oras destinatie!")
            return

        # Obținem orasele destinatie
        destinations = [self.dest_listbox.get(idx) for idx in selected_indices]
        cities = [start_city] + destinations

        # Calculam ruta optima folosind PSO
        optimal_route = self.pso_tsp(cities, start_city)

        # Calculam distanța totala și construim path-ul complet folosind A*
        total_distance = 0
        detailed_route = []
        segment_distances = []

        for i in range(len(optimal_route)):
            city1 = optimal_route[i]
            city2 = optimal_route[(i + 1) % len(optimal_route)]

            # Folosim numele curate pentru calculul rutei
            clean_city1 = self.get_clean_city_name(city1)
            clean_city2 = self.get_clean_city_name(city2)

            path, dist = self.a_star(clean_city1, clean_city2)
            if path is None:
                messagebox.showerror("Eroare", f"Nu s-a putut gasi o ruta intre {city1} și {city2}")
                return

            if i < len(optimal_route) - 1:
                detailed_route.extend(path[:-1])
            else:
                detailed_route.extend(path)

            segment_distances.append((city1, city2, dist))
            total_distance += dist

        # Construim textul rezultatului
        result_text = "Ruta detaliată:\n"

        for i in range(len(segment_distances)):
            city1, city2, dist = segment_distances[i]
            clean_city1 = self.get_clean_city_name(city1)
            clean_city2 = self.get_clean_city_name(city2)
            path, _ = self.a_star(clean_city1, clean_city2)

            result_text += f"\nSegment {i + 1}: {city1} -> {city2} ({dist:.1f} km)\n"
            result_text += f"Orase traversate: {' -> '.join(path)}\n"

        result_text += f"\nDistanta totala: {total_distance:.1f} km"
        self.result_label.config(text=result_text)

        # Desenam ruta pe harta
        self.map_canvas.delete("route")
        for i in range(len(detailed_route) - 1):
            city1, city2 = detailed_route[i], detailed_route[i + 1]
            x1 = 50 + (self.cities[city1][1] - 20) * 40
            y1 = 50 + (49 - self.cities[city1][0]) * 40
            x2 = 50 + (self.cities[city2][1] - 20) * 40
            y2 = 50 + (49 - self.cities[city2][0]) * 40
            self.map_canvas.create_line(x1, y1, x2, y2, fill='red', width=2, tags="route")

    def get_neighbors(self, city: str) -> List[str]:
        """
        Returneaza lista de orase conectate direct cu orasul dat.
        """
        clean_city = self.get_clean_city_name(city)
        return list(self.roads[clean_city].keys())

    def calculate_geographical_distance(self, city1: str, city2: str) -> float:
        """
        Calculează distanța geografica în linie dreaptă între două orașe
        """

        clean_city1 = self.get_clean_city_name(city1)
        clean_city2 = self.get_clean_city_name(city2)

        lat1, lon1 = self.cities[clean_city1]
        lat2, lon2 = self.cities[clean_city2]
        return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111

    def get_road_distance(self, city1: str, city2: str) -> float:
        """
        Returneaza distanta reala pe drum intre doua orașe conectate direct
        """
        clean_city1 = self.get_clean_city_name(city1)
        clean_city2 = self.get_clean_city_name(city2)
        try:
            return self.roads[clean_city1][clean_city2]
        except KeyError:
            return float('inf')

    def a_star(self, start: str, goal: str) -> Tuple[List[str], float]:
        """
        Implementare a algoritmului A* pentru gasirea celui mai scurt drum intre doua orașe,
        folosind distanta geografica ca euristică și distanta rutiera ca cost real
        """
        frontier = [(0, start, [start])]  # (f_score, oras, path) - Coada de prioritate
        explored = set()  # Set pentru orasele deja vizitate
        g_score = {start: 0}  # Costul real pană la fiecare oraa

        while frontier:
            f, current, path = heapq.heappop(frontier) # Obtinem nodul cu cel mai mic f_score

            if current == goal:
                # Imparțim distanta totala la 2 pentru a corecta dublarea
                return path, g_score[current] / 2 # returnăm path-ul și costul

            if current in explored:
                continue # Daca nodul a fost vizitat, trecem la următorul

            explored.add(current)

            for neighbor in self.get_neighbors(current):
                if neighbor in explored:
                    continue # Daca vecinul a fost vizitat, trecem la următorul

                road_distance = self.get_road_distance(current, neighbor) # Calculam costul de la nodul curent la vecin
                tentative_g = g_score[current] + road_distance # Costul estimat pana la vecin

                if neighbor not in g_score or tentative_g < g_score[neighbor]: # Daca gasim un drum mai bun catre vecin
                    g_score[neighbor] = tentative_g # Actualizam costul real
                    h_score = self.calculate_geographical_distance(neighbor, goal) # Calculam euristica
                    f_score = tentative_g + h_score # Calculam f_score
                    heapq.heappush(frontier, (f_score, neighbor, path + [neighbor])) # Adaugam nodul în coada de prioritate

        return None, float('inf') # Daca nu am gasit niciun drum, returnam None și infinit

    def pso_tsp(self, cities: List[str], start_city: str) -> List[str]:
        """
        Implementarea algoritmului Particle Swarm Optimization (PSO) pentru rezolvarea
        problemei Traveling Salesman Problem (TSP), folosind Random Key Encoding
        """
        clean_cities = [self.get_clean_city_name(city) for city in cities]
        clean_start_city = self.get_clean_city_name(start_city)

        num_cities = len(clean_cities)
        if num_cities <= 1:
            return cities

        # Parametri PSO
        num_particles = 30 # Numarul de particule
        num_iterations = 50 # Numarul de iterații
        w = 0.7  # inertia
        c1 = 2.0  # factorul cognitiv
        c2 = 2.0  # factorul social

        particles = np.random.uniform(low=0.0, high=1.0, size=(num_particles, num_cities)) # Initializam particulele cu chei aleatorii
        velocities = np.zeros((num_particles, num_cities)) # Initializam vitezele particulelor cu 0

        pbest = particles.copy() # Retinem cele mai bune poziții ale particulelor
        pbest_fitness = np.full(num_particles, float('inf')) # Initializam fitness-ul celor mai bune poziții cu infinit

        gbest = particles[0].copy()  # Retinem cea mai buna pozitie globala
        gbest_fitness = float('inf') # Initializam fitness-ul celei mai bune poziții globale cu infinit

        def random_key_to_route(keys: np.ndarray) -> List[str]:
            """
            Converteste cheile aleatorii intr-o ruta valida
            """
            city_keys = list(zip(keys, cities)) # Asociem cheile cu orasele
            sorted_pairs = sorted(city_keys, key=lambda x: x[0]) # Sortam perechile după cheie
            route = [pair[1] for pair in sorted_pairs] # Obtinem ruta sortata după chei

            if start_city in route:
                start_idx = route.index(start_city)
                route = route[start_idx:] + route[:start_idx]  # Asiguram că ruta începe cu sediul

            return route

        def calculate_fitness(route: List[str]) -> float:
            """
            Calculeaza lungimea totala a rutei
            """
            if self.get_clean_city_name(route[0]) != clean_start_city: # Verificam dacă ruta începe cu sediul
                return float('inf')

            total_distance = 0
            for i in range(len(route)):
                city1 = self.get_clean_city_name(route[i])
                city2 = self.get_clean_city_name(route[(i + 1) % len(route)])
                path, dist = self.a_star(city1, city2) # Calculam costul drumului între orașe
                if path is None:
                    return float('inf')
                total_distance += dist # Adaugam costul la distanța totala
            return total_distance # Returnam distanța totală

            # Procesul de optimizare PSO

        for iteration in range(num_iterations):
            for i in range(num_particles):
                particles[i] = particles[i] + velocities[i] # Actualizam pozitiile particulelor
                particles[i] = np.clip(particles[i], 0, 1)  # Restrangem cheile între 0 și 1

                current_route = random_key_to_route(particles[i]) # Obtinem ruta corespunzatoare poziției
                current_fitness = calculate_fitness(current_route) # Calculam fitness-ul rutei

                if current_fitness < pbest_fitness[i]: # Daca fitness-ul curent e mai bun decat cel mai bun al particulei
                    pbest[i] = particles[i].copy()
                    pbest_fitness[i] = current_fitness

                    if current_fitness < gbest_fitness: # Daca fitness-ul curent e mai bun decât cel mai bun global
                        gbest = particles[i].copy()
                        gbest_fitness = current_fitness

            for i in range(num_particles):
                r1, r2 = np.random.rand(2) # Generam 2 numere aleatorii pentru componentele cognitiva și sociala
                cognitive_velocity = c1 * r1 * (pbest[i] - particles[i]) # Calculam componenta cognitiva
                social_velocity = c2 * r2 * (gbest - particles[i]) # Calculam componenta socială
                velocities[i] = w * velocities[i] + cognitive_velocity + social_velocity # Actualizam vitezele particulelor

        return random_key_to_route(gbest) # Returnam ruta corespunzatoare celei mai bune poziții globale


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        root = tk.Tk()
        app = CourierSystem(root)
        root.withdraw() # Ascundem fereastra principala în timpul testelor
        testing.run_benchmark(app) # Rulam testele de benchmark
    else:
        root = tk.Tk()
        app = CourierSystem(root)
        root.mainloop() # Afisam fereastra principala