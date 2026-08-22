# Integración de DTBC en main.py, conductance.py y ab_ac_proof.py

## Qué se modificó

**`conductance.py`**: agregué `run_dtbc_conductance(...)`, con la misma firma que
`run_cap_conductance`/`run_transparent_conductance` ya existentes, para que sea un
reemplazo directo en cualquier script que ya use esas funciones. Usa `dtbc.py`
(la implementación exacta, corroborada contra Akramov et al. 2026) para propagar el
paquete y mide T/R integrando la corriente exacta en cada frontera (no por absorción en
una región finita, como el CAP).

**`main.py`**: `main()` ahora acepta `boundary="none"` (default, comportamiento idéntico
al de siempre — pared dura implícita) o `boundary="dtbc"`. Corré con
`python main.py --boundary dtbc`. Además agregué `compare_boundaries()` /
`python main.py --compare`, que corre ambas fronteras una vez y guarda
`boundary_comparison.png` con la comparación directa (densidad final + historia de
probabilidad superpuestas) — ya generado, adjunto.

**`ab_ac_proof.py`**: `simulate_G(...)` ahora acepta `boundary="dtbc"` además de
`"cap"`/`"transparent"`, y `junction_correction` como parámetro explícito (antes estaba
fijo en `True`). Agregué dos funciones nuevas, `plot_ab_oscillations_compare_boundaries`
y `plot_ac_oscillations_compare_boundaries`, que superponen CAP (viejo) vs DTBC (nuevo)
vs la fórmula analítica en un mismo gráfico. Corré con:
```
python ab_ac_proof.py --plots ab-compare ac-compare --n-compare 15 --compare-time-ps 70
python ab_ac_proof.py --plots ab-compare ac-compare --n-compare 15 --compare-time-ps 70 --no-junction-correction
```

## Los gráficos generados (adjuntos)

Corrí las 4 combinaciones (AB/AC × junction_correction True/False), 15 puntos cada una,
70 ps por punto (~28 min de cómputo total, DTBC es O(tiempo²) por punto):

1. `boundary_comparison.png` (de `main.py --compare`): con la frontera vieja (pared dura)
   nada puede salir jamás del dominio — P(t)/P(0)=1 siempre, por construcción. Con DTBC,
   la probabilidad decae genuinamente a medida que escapa, sin reflexión espuria.

2. `ab_oscillations_compare_boundaries_jcTrue.png` / `ac_oscillations_compare_boundaries_jcTrue.png`
   (configuración default del repo, `junction_correction=True`): CAP y DTBC dan
   prácticamente el mismo resultado (ambos aplastados muy por debajo de lo analítico) —
   confirma otra vez que `junction_correction` domina sobre cualquier mejora de frontera.

3. `ab_oscillations_compare_boundaries_jcFalse.png` / `ac_oscillations_compare_boundaries_jcFalse.png`
   (`junction_correction=False`): CAP y DTBC coinciden de cerca entre sí (con
   diferencias pequeñas pero reales, más visibles en la curva AC hacia φ_so~1.4-2.0),
   ambos todavía por debajo de lo analítico — consistente con el hallazgo de la sesión
   anterior de que el promediado espectral del paquete de ondas (no la frontera en sí)
   es la causa dominante de la brecha remanente en esta configuración.

## Nota honesta

Estos gráficos no muestran a DTBC "arreglando" las oscilaciones AB/AC — y eso es
exactamente lo que ya habíamos encontrado: el problema dominante ya no es la frontera
(eso lo resolvimos), sino `junction_correction` (factor >100x) y el promediado espectral
del paquete (factor ~2x). DTBC sigue siendo la mejora correcta y ya validada para el
problema que sí resuelve (reflexión en el truncamiento de los leads); ahora está
integrado y disponible como opción en los tres archivos para que sigas iterando sobre
los otros dos problemas usando esta base ya confiable.

## Costo computacional — para tenerlo en cuenta

DTBC es O(pasos_de_tiempo²) por punto (una convolución de memoria genuina, igual que en
el propio artículo). Un barrido de 15 puntos a 70 ps tardó ~7 minutos; duplicar los
puntos o el tiempo total escala mal (cuadráticamente en el tiempo). Si querés correr
barridos más finos, considera bajar `total_time_ps` (usando la vida media de las
resonancias del anillo, ver `quasibound.py`, como guía de cuánto es "suficiente") antes
que aumentar `n_points` sin límite.
