import time
import numpy as np
import pandas as pd



def run_benchmark(app):
    # Seturi de orase și sediu plecare pentru teste
    test_scenarios = [
        {"start_city": "București", "dest_cities": ["Cluj-Napoca", "Timișoara", "Iași"]},
        {"start_city": "Cluj-Napoca", "dest_cities": ["Timișoara", "Sibiu", "Brașov", "Constanța"]},
        {"start_city": "Timișoara", "dest_cities": ["București", "Cluj-Napoca", "Iași", "Galați", "Craiova"]},
        {"start_city": "Brașov", "dest_cities": ["București", "Sibiu", "Cluj-Napoca"]},
        {"start_city": "Sibiu", "dest_cities": ["Cluj-Napoca", "Timișoara", "Brașov", "Craiova", "București"]}
    ]

    num_trials = 10  # Numarul de rulari pentru fiecare scenariu
    results = []

    for scenario in test_scenarios:
        start_city = scenario["start_city"]
        dest_cities = scenario["dest_cities"]
        cities = [start_city] + dest_cities

        a_star_times = []
        a_star_distances = []
        pso_times = []
        pso_distances = []
        geo_distance_calls = 0
        road_distance_calls = 0

        print(f"Benchmark pentru scenariul: {start_city} -> {', '.join(dest_cities)}")

        for trial in range(num_trials):
            print(f"Trial {trial + 1}/{num_trials}")

            # A* Benchmark
            start_time = time.time()
            app.calculate_geographical_distance_original = app.calculate_geographical_distance
            app.get_road_distance_original = app.get_road_distance

            def mocked_calculate_geographical_distance(*args):
                nonlocal geo_distance_calls
                geo_distance_calls += 1
                return app.calculate_geographical_distance_original(*args)

            def mocked_get_road_distance(*args):
                nonlocal road_distance_calls
                road_distance_calls += 1
                return app.get_road_distance_original(*args)

            app.calculate_geographical_distance = mocked_calculate_geographical_distance
            app.get_road_distance = mocked_get_road_distance

            _, a_star_dist = app.a_star(start_city, dest_cities[0])  # Rulam A* doar o data
            end_time = time.time()
            a_star_times.append(end_time - start_time)
            a_star_distances.append(a_star_dist)

            # PSO Benchmark
            start_time = time.time()
            app.calculate_geographical_distance = app.calculate_geographical_distance_original
            app.get_road_distance = app.get_road_distance_original

            optimal_route = app.pso_tsp(cities, start_city)
            total_distance = 0
            for i in range(len(optimal_route)):
                city1 = optimal_route[i]
                city2 = optimal_route[(i + 1) % len(optimal_route)]
                _, dist = app.a_star(city1, city2)
                total_distance += dist

            end_time = time.time()
            pso_times.append(end_time - start_time)
            pso_distances.append(total_distance)

            app.calculate_geographical_distance = app.calculate_geographical_distance_original
            app.get_road_distance = app.get_road_distance_original


        avg_a_star_time = np.mean(a_star_times)
        avg_a_star_dist = np.mean(a_star_distances)
        avg_pso_time = np.mean(pso_times)
        avg_pso_dist = np.mean(pso_distances)

        print(f"  A* Time: {avg_a_star_time:.4f} sec")
        print(f"  A* Dist: {avg_a_star_dist:.1f} km")
        print(f"  PSO Time: {avg_pso_time:.4f} sec")
        print(f"  PSO Dist: {avg_pso_dist:.1f} km")
        print(f"  Geo distance calls: {geo_distance_calls}")
        print(f"  Road distance calls: {road_distance_calls}")

        results.append({
            "start_city": start_city,
            "dest_cities": ", ".join(dest_cities),
            "avg_a_star_time": avg_a_star_time,
            "avg_a_star_dist": avg_a_star_dist,
            "avg_pso_time": avg_pso_time,
            "avg_pso_dist": avg_pso_dist,
            "geo_distance_calls": geo_distance_calls,
            "road_distance_calls": road_distance_calls,
        })

    # Transformam rezultatele într-un tabel cu pandas
    df = pd.DataFrame(results)
    print("\n--- Benchmark Results ---")
    print(df)