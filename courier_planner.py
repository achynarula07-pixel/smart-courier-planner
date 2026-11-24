"""
Smart Courier Planner - Capstone Assignment
Author : Achyutam Narula | BCA (AI & Data Science) | Roll No. 2501060193

Pipeline (one delivery trip):
  Phase 2 : Merge Sort      -> order parcels by deadline, then priority
            0/1 Knapsack DP -> choose the best parcels that fit in the van
  Phase 3 : Dijkstra        -> shortest road distance between all stops
            Nearest-Neighbour TSP (approximation) -> order of the stops
  Phase 4 : Runtime tests on growing input sizes + correctness checks

Run:  python courier_planner.py            (full demo + performance tests)
"""
import heapq
import itertools
import os
import random
import sys
import time
from dataclasses import dataclass

INF = float("inf")


# ----------------------------------------------------------------------
# DATA DESIGN
# ----------------------------------------------------------------------
@dataclass
class Parcel:
    pid: str        # parcel id, e.g. "P01"
    node: int       # delivery point (graph node id)
    weight: int     # kg
    priority: int   # 1 (low) .. 5 (high)
    deadline: int   # hours from now

    @property
    def value(self):
        """Knapsack value = priority weight + urgency bonus (earlier deadline => bigger)."""
        return self.priority * 10 + max(0, 24 - self.deadline)


class Graph:
    """Undirected weighted graph stored as an adjacency list {u: [(v, km), ...]}."""

    def __init__(self, n):
        self.n = n
        self.adj = {i: [] for i in range(n)}

    def add_edge(self, u, v, w):
        self.adj[u].append((v, w))
        self.adj[v].append((u, w))


# ----------------------------------------------------------------------
# PHASE 2a - SORTING
# ----------------------------------------------------------------------
def parcel_key(p):
    """Earlier deadline first; for equal deadlines the higher priority first."""
    return (p.deadline, -p.priority)


def merge_sort(arr, key=lambda x: x):
    """Stable O(n log n) merge sort (guaranteed worst case)."""
    if len(arr) <= 1:
        return arr[:]
    mid = len(arr) // 2
    left, right = merge_sort(arr[:mid], key), merge_sort(arr[mid:], key)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if key(left[i]) <= key(right[j]):
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


def quick_sort(arr, key=lambda x: x):
    """Randomised quick sort - O(n log n) average, O(n^2) worst. Used for comparison only."""
    if len(arr) <= 1:
        return arr[:]
    pivot = key(arr[random.randrange(len(arr))])
    less = [x for x in arr if key(x) < pivot]
    equal = [x for x in arr if key(x) == pivot]
    more = [x for x in arr if key(x) > pivot]
    return quick_sort(less, key) + equal + quick_sort(more, key)


# ----------------------------------------------------------------------
# PHASE 2b - 0/1 KNAPSACK (DYNAMIC PROGRAMMING)
# ----------------------------------------------------------------------
def knapsack_dp(parcels, capacity):
    """
    dp[i][w] = best total value using the first i parcels with weight limit w.
    Returns (best_value, chosen_parcels, dp_table).  Time/space O(n * capacity).
    """
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        p = parcels[i - 1]
        row_prev, row = dp[i - 1], dp[i]
        for w in range(capacity + 1):
            row[w] = row_prev[w]                                   # skip parcel
            if p.weight <= w and row_prev[w - p.weight] + p.value > row[w]:
                row[w] = row_prev[w - p.weight] + p.value          # take parcel
    chosen, w = [], capacity                                       # backtrack
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            chosen.append(parcels[i - 1])
            w -= parcels[i - 1].weight
    chosen.reverse()
    return dp[n][capacity], chosen, dp


def knapsack_greedy(parcels, capacity):
    """Value/weight greedy heuristic - fast approximation, used to compare with DP."""
    total, load, chosen = 0, 0, []
    for p in sorted(parcels, key=lambda p: p.value / p.weight, reverse=True):
        if load + p.weight <= capacity:
            chosen.append(p); load += p.weight; total += p.value
    return total, chosen


def knapsack_bruteforce(parcels, capacity):
    """Exhaustive check (only for tiny inputs) to prove the DP answer is optimal."""
    best = 0
    for mask in range(1 << len(parcels)):
        wt = val = 0
        for i, p in enumerate(parcels):
            if mask >> i & 1:
                wt += p.weight; val += p.value
        if wt <= capacity and val > best:
            best = val
    return best


# ----------------------------------------------------------------------
# PHASE 3 - GRAPH ALGORITHMS
# ----------------------------------------------------------------------
def dijkstra(g, src):
    """Binary-heap Dijkstra. Returns (dist, prev). O((V+E) log V)."""
    dist = [INF] * g.n
    prev = [None] * g.n
    dist[src] = 0
    heap = [(0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v, w in g.adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    return dist, prev


def path_from(prev, src, dst):
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


def floyd_warshall(g):
    """Used only as an independent check of Dijkstra on small graphs."""
    d = [[INF] * g.n for _ in range(g.n)]
    for i in range(g.n):
        d[i][i] = 0
        for j, w in g.adj[i]:
            d[i][j] = min(d[i][j], w)
    for k in range(g.n):
        for i in range(g.n):
            for j in range(g.n):
                if d[i][k] + d[k][j] < d[i][j]:
                    d[i][j] = d[i][k] + d[k][j]
    return d


def shortest_paths_between_stops(g, depot, stops):
    """Run Dijkstra from the depot and every stop -> distance/prev tables."""
    nodes = [depot] + [s for s in stops if s != depot]
    table = {u: dijkstra(g, u) for u in nodes}
    return nodes, table


def nearest_neighbour_route(depot, stops, table):
    """
    TSP approximation: from the current stop always go to the closest unvisited
    stop, finally return to the depot.  O(k^2) on k stops.
    Returns (order, total_km) where order starts and ends at the depot.
    """
    unvisited = set(stops) - {depot}
    order, cur, total = [depot], depot, 0
    while unvisited:
        nxt = min(unvisited, key=lambda s: table[cur][0][s])
        total += table[cur][0][nxt]
        order.append(nxt); unvisited.remove(nxt); cur = nxt
    total += table[cur][0][depot]
    order.append(depot)
    return order, total


def bruteforce_tsp(depot, stops, table):
    """Optimal tour by trying every permutation (only for <= 8 stops)."""
    stops = [s for s in set(stops) if s != depot]
    best = INF
    for perm in itertools.permutations(stops):
        seq = [depot, *perm, depot]
        length = sum(table[a][0][b] for a, b in zip(seq, seq[1:]))
        best = min(best, length)
    return best


def expand_route(order, table):
    """Replace each hop of the TSP order by the actual road path found by Dijkstra."""
    full = [order[0]]
    for a, b in zip(order, order[1:]):
        full += path_from(table[a][1], a, b)[1:]
    return full


# ----------------------------------------------------------------------
# DATA GENERATORS (for performance tests)
# ----------------------------------------------------------------------
def random_graph(n_nodes, seed):
    rnd = random.Random(seed)
    g = Graph(n_nodes)
    order = list(range(n_nodes)); rnd.shuffle(order)
    for a, b in zip(order, order[1:]):                     # chain => connected
        g.add_edge(a, b, rnd.randint(1, 20))
    for _ in range(n_nodes * 2):                           # extra roads
        a, b = rnd.sample(range(n_nodes), 2)
        g.add_edge(a, b, rnd.randint(1, 20))
    return g


def random_parcels(n, n_nodes, seed):
    rnd = random.Random(seed)
    return [Parcel(f"P{i:04d}", rnd.randint(1, n_nodes - 1), rnd.randint(1, 15),
                   rnd.randint(1, 5), rnd.randint(1, 24)) for i in range(n)]


# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------
def plan_trip(parcels, graph, capacity, depot=0):
    """Full pipeline; returns a dict with every intermediate result."""
    ordered = merge_sort(parcels, parcel_key)
    best, chosen, dp = knapsack_dp(ordered, capacity)
    stops = sorted({p.node for p in chosen})
    nodes, table = shortest_paths_between_stops(graph, depot, stops)
    order, km = nearest_neighbour_route(depot, stops, table)
    return dict(ordered=ordered, best=best, chosen=chosen, dp=dp, stops=stops,
                table=table, order=order, km=km)


def timed(fn, *a):
    t = time.perf_counter()
    r = fn(*a)
    return r, (time.perf_counter() - t) * 1000      # milliseconds


# ----------------------------------------------------------------------
# DEMO (fixed small dataset - used for screenshots / explanation)
# ----------------------------------------------------------------------
NAMES = ["Depot", "Rajiv Chowk", "Sector 17", "Model Town", "Green Park",
         "Civil Lines", "Mall Road", "Airport Rd"]

DEMO_EDGES = [(0, 1, 4), (0, 2, 7), (1, 2, 3), (1, 3, 6), (2, 4, 5), (3, 4, 2),
              (3, 5, 8), (4, 6, 4), (5, 6, 3), (5, 7, 9), (6, 7, 6), (0, 7, 15)]

DEMO_PARCELS = [
    Parcel("P01", 3, 5, 3, 6),  Parcel("P02", 1, 3, 5, 2),
    Parcel("P03", 6, 8, 2, 10), Parcel("P04", 5, 4, 4, 5),
    Parcel("P05", 7, 7, 1, 20), Parcel("P06", 4, 6, 5, 3),
    Parcel("P07", 2, 2, 3, 8),  Parcel("P08", 6, 9, 4, 12),
    Parcel("P09", 1, 5, 2, 15), Parcel("P10", 5, 3, 3, 6),
]
VAN_CAPACITY = 25  # kg


def demo():
    print("=" * 70)
    print("PHASE 1 - PROBLEM SETUP")
    print("=" * 70)
    g = Graph(len(NAMES))
    for u, v, w in DEMO_EDGES:
        g.add_edge(u, v, w)
    print(f"Van capacity: {VAN_CAPACITY} kg | Depot: {NAMES[0]} | "
          f"Parcels: {len(DEMO_PARCELS)} | Graph: {g.n} nodes, {len(DEMO_EDGES)} roads")
    print("\nAdjacency list:")
    for u in range(g.n):
        print(f"  {u} {NAMES[u]:<12} -> " +
              ", ".join(f"{NAMES[v]}({w}km)" for v, w in g.adj[u]))
    print("\nInput parcels (unsorted):")
    print(f"  {'ID':<4} {'Dest':<12} {'kg':>3} {'Pri':>3} {'DL(h)':>5} {'Value':>5}")
    for p in DEMO_PARCELS:
        print(f"  {p.pid:<4} {NAMES[p.node]:<12} {p.weight:>3} {p.priority:>3} "
              f"{p.deadline:>5} {p.value:>5}")

    r = plan_trip(DEMO_PARCELS, g, VAN_CAPACITY)

    print("\n" + "=" * 70)
    print("PHASE 2 - SORTING + KNAPSACK DP")
    print("=" * 70)
    print("Parcels after Merge Sort (deadline asc, priority desc):")
    for p in r["ordered"]:
        print(f"  {p.pid} deadline={p.deadline:>2}h priority={p.priority} "
              f"weight={p.weight}kg value={p.value}")
    print(f"\nDP table dp[i][w] (rows = parcels in sorted order, columns = capacity 0..{VAN_CAPACITY}):")
    print("      " + "".join(f"{w:>4}" for w in range(VAN_CAPACITY + 1)))
    for i, row in enumerate(r["dp"]):
        label = "none" if i == 0 else r["ordered"][i - 1].pid
        print(f"{label:<5} " + "".join(f"{x:>4}" for x in row))
    load = sum(p.weight for p in r["chosen"])
    print(f"\nBest total value = {r['best']}   Load = {load}/{VAN_CAPACITY} kg")
    print("Selected parcels:", ", ".join(p.pid for p in r["chosen"]))
    print("Left for next trip:", ", ".join(p.pid for p in r["ordered"] if p not in r["chosen"]))
    gv, gc = knapsack_greedy(r["ordered"], VAN_CAPACITY)
    print(f"Greedy (value/weight) heuristic value = {gv}  "
          f"-> DP is {r['best'] - gv} better" if r["best"] > gv else
          f"Greedy (value/weight) heuristic value = {gv} (same as DP here)")

    print("\n" + "=" * 70)
    print("PHASE 3 - DIJKSTRA + NEAREST-NEIGHBOUR TSP")
    print("=" * 70)
    print("Delivery stops:", ", ".join(NAMES[s] for s in r["stops"]))
    print("\nDijkstra shortest paths from the Depot:")
    dist, prev = r["table"][0]
    for s in r["stops"]:
        print(f"  Depot -> {NAMES[s]:<12} {dist[s]:>3} km  via "
              + " > ".join(NAMES[x] for x in path_from(prev, 0, s)))
    print("\nNearest-neighbour route (stop order):")
    print("  " + " -> ".join(NAMES[x] for x in r["order"]))
    leg_total = 0
    for a, b in zip(r["order"], r["order"][1:]):
        d = r["table"][a][0][b]; leg_total += d
        print(f"    {NAMES[a]:<12} -> {NAMES[b]:<12} {d:>3} km  (cumulative {leg_total} km)")
    print("Actual road path:", " > ".join(NAMES[x] for x in expand_route(r["order"], r["table"])))
    print(f"TOTAL ROUTE LENGTH = {r['km']} km")
    opt = bruteforce_tsp(0, r["stops"], r["table"])
    print(f"Optimal (brute force) = {opt} km | Approximation ratio = {r['km'] / opt:.3f}")
    print("\nDelivery manifest (parcels grouped in route order):")
    seen = set()
    for stop in r["order"][1:-1]:
        if stop in seen:
            continue
        seen.add(stop)
        ps = [p.pid for p in r["chosen"] if p.node == stop]
        print(f"  {NAMES[stop]:<12}: {', '.join(ps)}")
    return g


# ----------------------------------------------------------------------
# PHASE 4 - CORRECTNESS + PERFORMANCE
# ----------------------------------------------------------------------
def correctness_checks(demo_graph):
    print("\n" + "=" * 70)
    print("PHASE 4a - CORRECTNESS EVIDENCE")
    print("=" * 70)
    for n in (10, 100, 1000):
        ps = random_parcels(n, 50, n)
        assert [p.pid for p in merge_sort(ps, parcel_key)] == \
               [p.pid for p in sorted(ps, key=parcel_key)]
        assert [parcel_key(p) for p in quick_sort(ps, parcel_key)] == \
               [parcel_key(p) for p in sorted(ps, key=parcel_key)]
    print("[PASS] merge sort and quick sort match Python's sorted() on n = 10, 100, 1000")

    for seed in range(5):
        ps = random_parcels(12, 10, seed)
        best, chosen, _ = knapsack_dp(ps, 30)
        assert best == knapsack_bruteforce(ps, 30)
        assert sum(p.weight for p in chosen) <= 30
    print("[PASS] knapsack DP equals brute-force optimum on 5 random 12-parcel cases")

    fw = floyd_warshall(demo_graph)
    for s in range(demo_graph.n):
        assert dijkstra(demo_graph, s)[0] == fw[s]
    print("[PASS] Dijkstra equals Floyd-Warshall for every source on the demo graph")

    ratios = []
    for seed in range(10):
        g = random_graph(15, seed)
        stops = random.Random(seed).sample(range(1, 15), 7)
        _, table = shortest_paths_between_stops(g, 0, stops)
        _, nn = nearest_neighbour_route(0, stops, table)
        ratios.append(nn / bruteforce_tsp(0, stops, table))
    print(f"[PASS] NN tour vs optimal on 10 random graphs (7 stops): "
          f"avg ratio {sum(ratios)/len(ratios):.3f}, worst {max(ratios):.3f}")


def performance_tests(plot=True):
    print("\n" + "=" * 70)
    print("PHASE 4b - RUNTIME TESTS (milliseconds)")
    print("=" * 70)
    sizes = [10, 50, 100, 250, 500, 1000]
    rows = []
    hdr = f"{'n':>5} {'nodes':>5} {'cap':>5} {'stops':>5} {'merge':>8} {'quick':>8} " \
          f"{'knap':>9} {'dijk':>9} {'NN':>8} {'km':>6}"
    print(hdr); print("-" * len(hdr))
    for n in sizes:
        nodes = n // 2 + 10
        ps = random_parcels(n, nodes, n)
        g = random_graph(nodes, n)
        cap = int(0.25 * sum(p.weight for p in ps))
        ordered, t_merge = timed(merge_sort, ps, parcel_key)
        _, t_quick = timed(quick_sort, ps, parcel_key)
        (best, chosen, _), t_knap = timed(knapsack_dp, ordered, cap)
        stops = sorted({p.node for p in chosen})
        (nodes_l, table), t_dij = timed(shortest_paths_between_stops, g, 0, stops)
        (order, km), t_nn = timed(nearest_neighbour_route, 0, stops, table)
        rows.append((n, t_merge, t_quick, t_knap, t_dij, t_nn))
        print(f"{n:>5} {nodes:>5} {cap:>5} {len(stops):>5} {t_merge:>8.2f} {t_quick:>8.2f} "
              f"{t_knap:>9.2f} {t_dij:>9.2f} {t_nn:>8.2f} {km:>6}")

    os.makedirs("results", exist_ok=True)
    with open("results/runtimes.csv", "w") as f:
        f.write("n,merge_ms,quick_ms,knapsack_ms,dijkstra_ms,nn_tsp_ms\n")
        for r in rows:
            f.write(",".join(f"{x:.3f}" if i else str(x) for i, x in enumerate(r)) + "\n")
    print("Saved results/runtimes.csv")

    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [r[0] for r in rows]
            plt.figure(figsize=(8, 5))
            for idx, lab in [(1, "Merge sort"), (2, "Quick sort"), (3, "Knapsack DP"),
                             (4, "Dijkstra (all stops)"), (5, "Nearest-neighbour TSP")]:
                plt.plot(xs, [max(r[idx], 1e-3) for r in rows], marker="o", label=lab)
            plt.yscale("log"); plt.xscale("log")
            plt.xlabel("Number of parcels (n)"); plt.ylabel("Runtime (ms, log scale)")
            plt.title("Runtime of major steps vs input size")
            plt.grid(True, which="both", alpha=0.3); plt.legend()
            plt.tight_layout(); plt.savefig("results/runtime_chart.png", dpi=150)
            print("Saved results/runtime_chart.png")
        except ImportError:
            print("matplotlib not installed - run: pip install matplotlib")

    print("\nOptimization notes:")
    print(" - Knapsack DP is the bottleneck (O(n*capacity)); use a 1-D table if only the value")
    print("   is needed, or scale weights / use branch-and-bound for very large vans.")
    print(" - Dijkstra is re-run for every stop; cache results or use A* / multi-source tricks.")
    print(" - NN tour can be improved with 2-opt local search (usually 5-15% shorter).")
    print(" - Sorting cost is negligible next to DP/graph steps; merge sort is kept for stability.")


if __name__ == "__main__":
    demo_graph = demo()
    correctness_checks(demo_graph)
    performance_tests(plot="--no-plot" not in sys.argv)
