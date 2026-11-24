# Smart Courier Planner

**Capstone Assignment** - BCA (AI & Data Science), 2nd Year, Semester 3, Section D  
**Author:** Achyutam Narula | **Roll No.:** 2501060193 | **Email:** 2501060193@krmu.edu.in

A Python prototype that takes a list of parcels and delivery points and produces
(1) an optimised packing plan for one van trip and (2) an ordered delivery route.

## Algorithms used

| Step | Algorithm | Complexity | Why |
|---|---|---|---|
| Parcel prioritisation | Merge Sort (Quick Sort for comparison) | O(n log n) | Stable, guaranteed worst case |
| Box / van filling | 0/1 Knapsack - Dynamic Programming | O(n x W) | Exact optimum, unlike greedy |
| Shortest paths | Dijkstra (binary heap) | O((V+E) log V) | Non-negative road distances |
| Route order | Nearest-Neighbour TSP (approximation) | O(k^2) | Exact TSP is NP-hard |

## Pipeline

```
Parcels --Merge Sort--> sorted list --Knapsack DP--> chosen parcels
   --> delivery stops --Dijkstra--> distance table --Nearest Neighbour--> route + km
```

## How to run

```bash
pip install -r requirements.txt     # only needed for the runtime chart
python courier_planner.py           # demo + correctness checks + performance tests
python courier_planner.py --no-plot # same, without matplotlib
```

Python 3.8+ required. Outputs are saved in `results/`:
`output_trace.txt`, `runtimes.csv`, `runtime_chart.png`.

## Data design

- **Parcel:** `pid, node (delivery point), weight (kg), priority (1-5), deadline (hours)`
- **Knapsack value:** `priority x 10 + (24 - deadline)` (urgent + important = more value)
- **Graph:** undirected, weighted adjacency list `{node: [(neighbour, km), ...]}`, node 0 = depot

## Files

- `courier_planner.py` - complete implementation (Phases 1-4)
- `results/` - output trace, runtime CSV and chart
- `Courier_Planner_Report.docx` - final report (Phase 5)
