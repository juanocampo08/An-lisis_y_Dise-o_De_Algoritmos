# ============================================================
#  Comparación Dijkstra vs Bellman-Ford — Red vial Medellín
#  Rúbrica: modelado, implementación, comparación, métricas
# ============================================================

import time

# ─── 1. GRAFO ───────────────────────────────────────────────

class GrafoVial:
    def __init__(self):
        self.calles = {}

    def agregar_nodo(self, nodo):
        if nodo not in self.calles:
            self.calles[nodo] = []

    def agregar_calle(self, origen, destino, dist, peligro, bidireccional=False):
        self.calles[origen].append({"para": destino, "dist": dist, "peligro": peligro})
        if bidireccional:
            self.calles[destino].append({"para": origen, "dist": dist, "peligro": peligro})

    def obtener_nodos(self):
        return list(self.calles.keys())

    def obtener_aristas(self):
        aristas = []
        for origen, vecinos in self.calles.items():
            for v in vecinos:
                aristas.append((origen, v["para"], v["dist"], v["peligro"]))
        return aristas


def construir_grafo():
    g = GrafoVial()
    nodos = [
        "El Poblado", "Laureles", "Envigado",
        "Centro", "Belén", "Aranjuez",
        "Robledo", "Itagüí", "Bello"
    ]
    for n in nodos:
        g.agregar_nodo(n)

    calles = [
        ("El Poblado", "Centro",    3200, 0.20, True),
        ("El Poblado", "Envigado",  1800, 0.10, True),
        ("Centro",     "Laureles",  2900, 0.35, True),
        ("Centro",     "Aranjuez",  4100, 0.25, True),
        ("Laureles",   "Belén",     2200, 0.15, True),
        ("Laureles",   "Robledo",   3000, 0.40, True),
        ("Aranjuez",   "Bello",     5500, 0.60, True),
        ("Robledo",    "Aranjuez",  3800, 0.45, True),
        ("Envigado",   "Itagüí",   2100, 0.30, True),
        ("Belén",      "Itagüí",   3400, 0.55, True),
        ("Centro",     "Robledo",  4500, 0.42, True),
        ("El Poblado", "Itagüí",   2600, 0.38, True),
    ]
    for origen, destino, dist, peligro, bid in calles:
        g.agregar_calle(origen, destino, dist, peligro, bid)
    return g


# ─── 2. FUNCIÓN DE COSTO ────────────────────────────────────

def costo_arista(dist, peligro, alpha, beta):
    return alpha * float(dist) + beta * float(peligro)


# ─── 3. DIJKSTRA (tu implementación) ────────────────────────

def dijkstra(grafo, inicio, fin, alpha, beta):
    nodos = grafo.obtener_nodos()
    costos = {nodo: float("inf") for nodo in nodos}
    anteriores = {}
    visitados = set()
    costos[inicio] = 0
    aristas_evaluadas = 0

    while True:
        # Seleccionar nodo no visitado con menor costo — O(V)
        nodo_actual = None
        menor_distancia = float("inf")
        for nodo in nodos:
            if nodo not in visitados and costos[nodo] < menor_distancia:
                menor_distancia = costos[nodo]
                nodo_actual = nodo

        if nodo_actual is None or nodo_actual == fin:
            break

        visitados.add(nodo_actual)

        for vecino_info in grafo.calles[nodo_actual]:
            vecino = vecino_info["para"]
            aristas_evaluadas += 1
            c = costo_arista(vecino_info["dist"], vecino_info["peligro"], alpha, beta)
            nuevo_costo = costos[nodo_actual] + c
            if nuevo_costo < costos[vecino]:
                costos[vecino] = nuevo_costo
                anteriores[vecino] = nodo_actual

    # Reconstruir ruta
    ruta = []
    nodo = fin
    while nodo in anteriores:
        ruta.append(nodo)
        nodo = anteriores[nodo]
    ruta.append(inicio)
    ruta.reverse()

    return {
        "ruta": ruta if ruta[0] == inicio and ruta[-1] == fin else [],
        "costo": costos[fin],
        "nodos_visitados": len(visitados),
        "aristas_evaluadas": aristas_evaluadas,
    }


# ─── 4. BELLMAN-FORD ─────────────────────────────────────────

def bellman_ford(grafo, inicio, fin, alpha, beta):
    nodos   = grafo.obtener_nodos()
    aristas = grafo.obtener_aristas()
    V = len(nodos)

    costos     = {nodo: float("inf") for nodo in nodos}
    anteriores = {}
    costos[inicio] = 0

    aristas_evaluadas  = 0
    iteraciones_reales = 0

    # Relajar |V|-1 veces
    for i in range(V - 1):
        iteraciones_reales += 1
        hubo_cambio = False

        for origen, destino, dist, peligro in aristas:
            aristas_evaluadas += 1
            if costos[origen] == float("inf"):
                continue
            c = costo_arista(dist, peligro, alpha, beta)
            if costos[origen] + c < costos[destino]:
                costos[destino] = costos[origen] + c
                anteriores[destino] = origen
                hubo_cambio = True

        # Early stopping: si no hubo cambios, ya convergió
        if not hubo_cambio:
            break

    # Detección de ciclos negativos
    ciclo_negativo = False
    for origen, destino, dist, peligro in aristas:
        c = costo_arista(dist, peligro, alpha, beta)
        if costos[origen] != float("inf") and costos[origen] + c < costos[destino]:
            ciclo_negativo = True
            break

    # Reconstruir ruta
    ruta = []
    nodo = fin
    while nodo in anteriores:
        ruta.append(nodo)
        nodo = anteriores[nodo]
    ruta.append(inicio)
    ruta.reverse()

    # Nodos que salieron de inf = nodos "alcanzados"
    nodos_visitados = sum(1 for c in costos.values() if c != float("inf"))

    return {
        "ruta": ruta if ruta[0] == inicio and ruta[-1] == fin else [],
        "costo": costos[fin],
        "nodos_visitados": nodos_visitados,
        "aristas_evaluadas": aristas_evaluadas,
        "iteraciones": iteraciones_reales,
        "ciclo_negativo": ciclo_negativo,
    }


# ─── 5. COMPARACIÓN ──────────────────────────────────────────

def comparar(grafo, inicio, fin, alpha, beta):
    t0 = time.perf_counter()
    rd = dijkstra(grafo, inicio, fin, alpha, beta)
    t1 = time.perf_counter()
    rb = bellman_ford(grafo, inicio, fin, alpha, beta)
    t2 = time.perf_counter()

    td = (t1 - t0) * 1000
    tb = (t2 - t1) * 1000

    return rd, rb, td, tb


def imprimir_comparacion(rd, rb, td, tb, inicio, fin, alpha, beta):
    sep = "─" * 58

    print(f"\n{sep}")
    print(f"  Origen: {inicio}   Destino: {fin}")
    print(f"  α = {alpha}   β = {beta}")
    print(f"  C(e) = {alpha}×distancia + {beta}×riesgo")
    print(sep)

    ancho = 24
    print(f"  {'Métrica':<22}  {'Dijkstra':>{ancho}}  {'Bellman-Ford':>{ancho}}")
    print(f"  {'─'*22}  {'─'*ancho}  {'─'*ancho}")

    def fmt_ruta(r):
        if not r:
            return "sin ruta"
        return " → ".join(n.split()[0] for n in r)

    def fmt_costo(c):
        return f"{c:,.1f}" if c != float("inf") else "∞"

    print(f"  {'Ruta':<22}  {fmt_ruta(rd['ruta']):>{ancho}}  {fmt_ruta(rb['ruta']):>{ancho}}")
    print(f"  {'Costo total':<22}  {fmt_costo(rd['costo']):>{ancho}}  {fmt_costo(rb['costo']):>{ancho}}")
    print(f"  {'Nodos procesados':<22}  {rd['nodos_visitados']:>{ancho}}  {rb['nodos_visitados']:>{ancho}}")
    print(f"  {'Aristas evaluadas':<22}  {rd['aristas_evaluadas']:>{ancho}}  {rb['aristas_evaluadas']:>{ancho}}")
    print(f"  {'Iteraciones':<22}  {'n/a (por nodo)':>{ancho}}  {rb['iteraciones']:>{ancho}}")
    print(f"  {'Ciclo negativo':<22}  {'no aplica':>{ancho}}  {'Sí ⚠️' if rb['ciclo_negativo'] else 'No':>{ancho}}")
    print(f"  {'Tiempo (ms)':<22}  {td:>{ancho}.4f}  {tb:>{ancho}.4f}")
    print(f"  {'Complejidad':<22}  {'O(V²) sin heap':>{ancho}}  {'O(V×E)':>{ancho}}")

    # Conclusión automática
    if rd["ruta"] and rb["ruta"]:
        if abs(rd["costo"] - rb["costo"]) < 0.01:
            print(f"\n  ✓ Ambos encontraron la misma ruta con costo idéntico.")
        else:
            print(f"\n  ⚠ Los costos difieren — revisar implementación.")
    elif not rd["ruta"] and not rb["ruta"]:
        print(f"\n  ✗ Ambos confirman: no existe ruta.")


# ─── 6. MAIN ────────────────────────────────────────────────

def main():
    grafo  = construir_grafo()
    inicio = "El Poblado"
    fin    = "Bello"

    print("=" * 58)
    print("  DIJKSTRA vs BELLMAN-FORD — Red vial Medellín")
    print("=" * 58)

    escenarios = [
        (1.0, 0.0, "Solo distancia (más rápido)"),
        (0.0, 1.0, "Solo riesgo   (más seguro)"),
        (1.0, 1.0, "Balance       (distancia + riesgo)"),
        (0.5, 2.0, "Prioridad seguridad"),
        (2.0, 0.5, "Prioridad rapidez"),
    ]

    for alpha, beta, descripcion in escenarios:
        print(f"\n  Escenario: {descripcion}")
        rd, rb, td, tb = comparar(grafo, inicio, fin, alpha, beta)
        imprimir_comparacion(rd, rb, td, tb, inicio, fin, alpha, beta)

    # Caso sin ruta (nodo aislado)
    grafo.agregar_nodo("Nodo_aislado")
    print(f"\n  Escenario: nodo sin conexiones (caso borde)")
    rd, rb, td, tb = comparar(grafo, inicio, "Nodo_aislado", 1.0, 1.0)
    imprimir_comparacion(rd, rb, td, tb, inicio, "Nodo_aislado", 1.0, 1.0)

    print("\n" + "=" * 58)
    print("  RESUMEN DE TRADE-OFFS")
    print("=" * 58)
    print("""
  Dijkstra
    + Más rápido en grafos sin pesos negativos
    + Para en cuanto llega al destino (early stop)
    - No detecta ciclos negativos
    - Requiere heap para ser O((V+E) log V)
      (esta versión es O(V²) sin heap)

  Bellman-Ford
    + Maneja pesos negativos (útil si β es negativo)
    + Detecta ciclos negativos — más robusto
    + Más simple de implementar correctamente
    - Siempre recorre todas las aristas × (V-1)
    - Más lento en grafos grandes sin ciclos negativos
  """)

if __name__ == "__main__":
    main()