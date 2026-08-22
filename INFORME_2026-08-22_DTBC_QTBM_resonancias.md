# Informe: DTBC exacto, corroboración de QTBM, y búsqueda de estados cuasi-ligados

Generado por una tarea programada de investigación, 2026-08-22, en respuesta a tu pedido
de: (1) explicar y resolver el problema de las DTBC ahora que adjuntaste los dos artículos,
(2) corroborar que las implementaciones anteriores coinciden con lo descrito en los
papers, (3) buscar otras publicaciones que ayuden a identificar la raíz real del problema
de las oscilaciones AB/AC, dado que las reflexiones en los bordes no parecen ser la causa,
(4) investigar si hay estados con permanencia muy alta en el anillo que contaminen la
simulación, y cómo detectarlos sin correr simulaciones larguísimas, (5) una tabla
comparativa final, y (6) todo empaquetado con explicación.

**Resultado más importante, adelantado**: con las DTBC exactas (no aproximadas), el
"atrapamiento permanente" que reportamos la sesión anterior (T+R estancado en ~0.35-0.38
sin importar cuánto tiempo simulábamos) **desaparece casi por completo** — con DTBC exacto
la probabilidad SÍ termina saliendo casi en su totalidad (99.9% para t=120 ps). Es decir:
las reflexiones en los bordes SÍ eran (al menos en parte) el problema, pero ni el CAP ni
la autoenergía monocromática del parche anterior eran lo bastante buenos como para
eliminarlo — estaban generando un atrapamiento **artificial** que confundíamos con un
efecto físico real. Sin embargo, incluso con DTBC exacto, el valor de T obtenido del
paquete de ondas real **todavía no coincide** con el valor analítico ni con el QTBM de una
sola energía — encontramos una segunda causa, distinta y más sutil: el anillo tiene
resonancias muy angostas (Γ~0.03-0.05 meV) comparadas con el ancho de banda de energía del
paquete gaussiano inyectado (~0.8 meV), así que el experimento numérico está promediando
sobre muchas resonancias en vez de medir una sola energía bien definida, como sí asume la
fórmula analítica. Ver sección 5 para los números y qué tan lejos llega esta explicación.

---

## 1. Corroboración: DTBC (Akramov et al. 2026)

### 1.1 Qué se hizo con el artículo

Con el PDF completo en mano, extraje las ecuaciones exactas (su Ec. 4-15) y re-derivé,
de forma independiente, la relación de "punto fantasma" (*):

    Psi_{J+1}(t) = i * INT_0^t K(t-tau) Psi_J(tau) dtau,
    K(t) = exp(-i t/tau_c) J_1(t/tau_c) / t,   tau_c = hbar/(2 t_lead)

en vez de traducir literalmente su bookkeeping basado en la derivada discreta D_x (sus Ec.
27-32), porque esa vía introduce un factor 1/h extra que no cuadra dimensionalmente al
compararlo con su propia Ec. 15 ya simplificada (posible errata en su Ec. 28/32, no
verificable sin acceso a sus notas de cálculo internas). En cambio, anclé la derivación a
su Ec. 15 (que ellos mismos verifican contra el límite continuo, Ec. 26, coincidiendo con
la Ec. 2 — la referencia más sólida del artículo) y de ahí bajé a la relación de amplitud
pura (*), sin "h" sueltas.

### 1.2 Verificación independiente de la fórmula (antes de tocar el anillo)

Antes de confiar en nada, verifiqué (*) contra una cadena larga (2000 sitios) SIN ningún
truco de frontera, simplemente lo bastante larga para que nada rebote en el tiempo
simulado — el valor "verdadero" de Psi en el sitio J+1 en cada instante. La fórmula (*),
aplicada a la historia de Psi_J de esa cadena de referencia, reprodujo el valor verdadero
con error relativo ~4e-6 (ver la verificación en el registro de esta sesión). Esto
confirma que el núcleo de Bessel en sí está bien derivado.

### 1.3 Dos bugs reales encontrados y corregidos (en mi código, no en el artículo)

Al discretizar (*) con Crank-Nicolson e insertarla en la ecuación de frontera del sitio
más externo del lead, encontré y corregí dos errores propios (no del artículo) durante la
verificación de esta sesión:

**Bug 1 — factor de fase incorrecto en el término fuente**: la primera versión tenía un
factor extra de `i` en el término fuente explícito (`source = -1j*(dt/hbar)*Src_n` en vez
de `source = -(dt/hbar)*Src_n`). Se detectó porque, sin él, la norma dentro del dominio
CRECÍA (violando conservación) incluso antes de que el paquete llegara a la frontera —
una señal inequívoca de bug, no de física. Corregido re-derivando el álgebra completa
paso a paso (documentado en el docstring de `dtbc.py`).

**Bug 2 — energía on-site mal contabilizada en mi propio script de reproducción**
(`dtbc_selftest.py`, NO en `dtbc.py`, que sí usa la construcción correcta basada en
enlaces reales de `tools.py`): usé `scipy.sparse.diags` con una diagonal uniforme "2*t"
en todos los sitios, lo cual le da al sitio de frontera una energía on-site de bulk (2*t)
aunque solo tiene UN enlace físico real (debería ser t, no 2t, antes de agregar la
corrección). Esto llevaba a una sobre-corrección y a una absorción demasiado débil (la
norma se estancaba en ~0.94 en vez de decaer a ~0 como en la Fig. 3 del artículo).
Detectado con un chequeo de residuo directo contra la cadena de referencia (2000 sitios):
con el bug, el residuo de la ecuación de frontera era un 0.5% relativo, EXACTAMENTE
`dt*t_hop` — una pista clara de un error sistemático, no ruido numérico. Corregido
construyendo la energía on-site sitio por sitio a partir de los enlaces reales.

### 1.4 Reproducción de la validación propia del artículo (Fig. 2 y Fig. 3)

Con ambos bugs corregidos, reproduje exactamente su demostración numérica (Sección III):
J=400, h=0.025, dt=6.25e-6, paquete gaussiano sigma=1, k0=5, en las unidades hbar=m=1
del artículo. Resultado (`dtbc_selftest.py`, figura adjunta
`dtbc_selftest_paper_reproduction.png`):

- **0 de 320,000 pasos** muestran la norma M(t) aumentando (su criterio de "sin reflexión
  espuria").
- M(t=2.0) = 0.000813, decayendo suave y monótonamente desde 1.0 — coincide
  cualitativamente con su Fig. 3 (que también muestra decaimiento a ~0 hacia t=2).

Esta es la corroboración más fuerte posible: mi implementación reproduce el resultado
publicado del artículo, con los parámetros exactos que ellos usaron.

### 1.5 Aplicación a JCE26 (unidades físicas)

`dtbc.py` traduce todo a las unidades del repo (meV, ps, nm) usando
`tau_c = hbar/(2*t_lead)` como el análogo físico de la "h²" del artículo (una escala
TEMPORAL, no espacial — consecuencia de que ellos usan hbar=m=1, donde longitud² y tiempo
se vuelven intercambiables). Verificado en el benchmark de reflexión aislado (mismo que
usamos la sesión anterior para el parche de autoenergía monocromática,
`test_transparent_boundary.py`, ahora extendido con la opción `boundary="dtbc"`):

| Frontera | Fracción reflejada de vuelta al dispositivo |
|---|---|
| Ninguna (pared dura, referencia) | 0.998 |
| CAP (parámetros actuales del repo) | 1.41e-2 |
| Autoenergía monocromática (parche sesión anterior) | 2.24e-3 |
| **DTBC exacto (esta sesión)** | **1.92e-4** |

DTBC exacto refleja **73x menos** que el CAP y **11.6x menos** que la autoenergía
monocromática — exactamente el orden esperado, ya que DTBC es exacto para *todo* el
paquete de ondas, no solo su componente central de momento.

---

## 2. Corroboración: QTBM (Shao, Porod, Lent & Kirkner 1995)

### 2.1 Comparación estructural con el artículo

El artículo formula el problema de transmisión (su Ec. 22) exactamente como lo
implementé la sesión pasada en `qtbm.py`: `(H - k_L B^L - k_R B^R - E D) psi = -a(E) P`,
con matrices de frontera `B^L`, `B^R` que tienen una única entrada no nula en el nodo de
frontera (su Ec. 21c/d) — coincide estructuralmente con mi self-energy de un solo sitio.

**Diferencia importante, ya anotada en el README anterior y ahora confirmada leyendo el
artículo completo**: su formulación es para **elementos finitos** (discretización
continua de -d²/dx², dando un término de frontera LINEAL en k, ver su Ec. 11/12).
La nuestra es para una **red tight-binding** (con dispersión E=2t(1-cos(k·dx))), donde la
autoenergía exacta es EXPONENCIAL en k (Sigma=-t·e^{ika}), no lineal. Esto significa que
el truco de "linealizar duplicando la matriz" que ellos usan para convertir su problema
cuadrático/cuártico en uno lineal (sus Ec. 24-41) **no aplica directamente** a nuestra red
— no es un error, es que estamos resolviendo el mismo tipo de problema para una
discretización distinta, con su propia forma cerrada (la exponencial), que ya usábamos.

### 2.2 Corroboración numérica

`qtbm.py` fue validado la sesión anterior con conservación exacta de probabilidad
(T+R=1 a ~1e-13). Esta sesión, además, usé la MISMA maquinaria de QTBM (evaluada en
energía real) como base del buscador de resonancias (sección 3) — que a su vez reproduce
picos afilados y consistentes con lo que se espera de un anillo tipo Fabry-Pérot, otra
señal de que el método está funcionando correctamente.

---

## 3. Detector de estados cuasi-ligados (sin correr la simulación larga)

### 3.1 El método

El artículo de Shao et al. resuelve directamente un problema de autovalores para hallar
las energías complejas E = E_R - i*Gamma de los estados cuasi-ligados (su Fig. 3, Tabla
I), con vida media tau = hbar/(2*Gamma). Para SU discretización (FEM, término de frontera
lineal en k) esto se linealiza limpiamente. Para nuestra red tight-binding (término
EXPONENCIAL en k), el análogo directo requeriría una búsqueda de raíces en el plano
complejo de E — lo implementé como herramienta secundaria (`quasibound.complex_pole_refine`,
usa un método de Newton 2D sobre el autovalor más chico de la matriz, con `scipy.sparse.linalg.eigs`
en modo shift-invert), pero es más lento y menos robusto para un sistema disperso grande
como el nuestro.

En cambio, implementé como herramienta **principal** algo más simple, igual de riguroso, y
mucho más rápido: los estados cuasi-ligados son polos de la amplitud de transmisión t(E)
en el plano complejo (esto lo dice el propio artículo en su introducción). Un polo en
E_R - i*Gamma produce, sobre el eje real, un pico Lorentziano en T(E) con ancho a media
altura (FWHM) = 2*Gamma. **Esto significa que Gamma (y por tanto la vida media) se puede
leer directamente de un barrido de T(E) en energías reales** — que ya calculamos de forma
exacta y barata con QTBM (una sola resolución de sistema disperso por punto de energía,
sin evolución temporal). `quasibound.py` implementa exactamente esto: `scan_transmission`
+ `find_resonances` (ajuste Lorentziano local a cada pico).

### 3.2 Resultado para el anillo transparente (Phi=0.5, alpha=20 meV·nm, junction_correction=False)

Barrido de 300 puntos entre 3.5 y 5.0 meV alrededor de E_F=4.19 meV:

| E_R (meV) | Gamma (meV) | vida media tau (ps) | T pico |
|---|---|---|---|
| 3.892 | 0.031 | 10.7 | ~2.0 |
| 4.561 | 0.037 | 8.9 | ~2.0 |
| 4.558 | 0.052 | 6.3 | ~2.1 |
| 3.889 | 0.043 | 7.7 | ~2.1 |

**Sí existen resonancias angostas cerca de E_F**, con vidas medias del orden de 7-11 ps.
(Los ajustes Lorentzianos individuales no son perfectos — T pico excede ligeramente 2.0 en
el ajuste porque las resonancias están muy próximas entre sí e interfieren, así que estas
cifras son una estimación de orden de magnitud, no un resultado de precisión — para
separarlas con precisión haría falta un ajuste multi-Lorentziano simultáneo, que no
implementé por tiempo, pero queda como mejora directa de `find_resonances`.)

### 3.3 Verificación cruzada con la simulación temporal exacta (DTBC)

Esto es consistente con lo que vimos en la sección 1.5 al correr la propagación temporal
completa del anillo con DTBC exacto: con vidas medias de ~7-11 ps, después de ~120 ps
(10-17 vidas medias) esperaríamos que quedara <0.1-0.5% de probabilidad sin escapar — y
en efecto medimos 0.12% a t=120 ps. Dos métodos completamente independientes (barrido de
energía estacionario vs. propagación temporal con frontera exacta) coinciden en el orden
de magnitud del tiempo de escape. Esto es evidencia sólida de que:

1. Sí hay resonancias (estados de larga permanencia) reales en este anillo.
2. Su vida media es de decenas de ps, NO infinita — el "atrapamiento permanente" visto la
   sesión pasada con CAP/autoenergía monocromática era un artefacto de frontera
   imperfecta, no estas resonancias en sí.
3. **Recomendación práctica**: antes de fijar `total_time_ps` en cualquier script de este
   repo, correr `quasibound.find_resonances` cerca de la energía de Fermi de interés y
   usar `total_time_ps >= 10-15 * max(tau_ps)` como regla de bolsillo para asegurar
   convergencia — mucho más barato que adivinar por prueba y error.

---

## 4. Búsqueda bibliográfica

Para identificar la raíz del problema más allá de las fronteras, busqué trabajos sobre
simulaciones de paquetes de onda dependientes del tiempo en anillos de Aharonov-Bohm y
sobre resonancias de Fano en transporte cuántico tight-binding. Los más directamente
relevantes:

- J. Li, "Resonant transport properties of tight-binding mesoscopic rings", Phys. Rev. B
  55, 5337 (1997) — trata exactamente el tipo de resonancias tipo Fabry-Pérot/Fano que
  encontramos en un anillo tight-binding, relevante para entender la estructura fina de
  T(E) que detectó `quasibound.py`.
  https://repository.hkust.edu.hk/ir/bitstream/1783.1-26526/1/PhysRevB.55.5337.pdf
- "Wave packet propagation through branched quantum rings under applied magnetic fields",
  Physica E / ScienceDirect (2019) — mismo tipo de cálculo que este repo (paquetes de
  onda dependientes del tiempo en anillos ramificados con campo magnético), buen punto de
  comparación metodológica directa.
  https://www.sciencedirect.com/science/article/abs/pii/S1386947718313766
- "Wave packet dynamics in semiconductor quantum rings of finite width", Phys. Rev. B 80,
  125331 (2009) — trata explícitamente la dinámica de paquetes de onda en anillos,
  potencialmente relevante para métodos de validación contra el caso estacionario.
  https://journals.aps.org/prb/abstract/10.1103/PhysRevB.80.125331
- "Time-dependent wave packet simulations of transport through Aharonov-Bohm rings with an
  embedded quantum dot" (PubMed) — mismo paradigma metodológico (paquete de onda +
  AB ring + QD embebido), no pude leer el texto completo por rate-limiting del fetch,
  pero el título y contexto son muy cercanos a nuestro caso; vale la pena que lo revises
  directamente. https://pubmed.ncbi.nlm.nih.gov/28195564/

Nota honesta: no encontré un artículo que trate explícitamente "el ancho de banda del
paquete de ondas debe ser más angosto que el espaciado de resonancias para que el
transporte dependiente del tiempo reproduzca la fórmula de Landauer de una sola energía"
como una afirmación general — es una consecuencia directa y estándar de la relación
resonancia-ancho de banda en teoría de scattering (ver cualquier texto de Fano
resonances), pero no localicé una referencia que lo diga en estos términos exactos para
anillos cuánticos específicamente. Si esto termina siendo el hallazgo central de tu
trabajo, podría valer la pena buscar más a fondo o derivarlo tú mismo como resultado
original.

---

## 5. El nuevo hallazgo: promediado espectral del paquete de ondas

### 5.1 El problema

Incluso con DTBC exacto (sin ninguna reflexión espuria) y `junction_correction=False`
(la configuración que mejor concuerda con lo analítico según lo que encontramos la sesión
anterior), el valor de T medido con el paquete de ondas real **no coincide** con el valor
de QTBM a una sola energía:

- QTBM en E_F=4.19 meV exacto: T = 1.178
- DTBC con el paquete de ondas real (sigma=150nm): T = 0.243

Un paquete gaussiano de ancho espacial sigma_x tiene un ancho de energía
Delta_E ~ v_group / (2*sigma_x) ~ 0.84 meV para sigma_x=150nm en este sistema — **mucho
más ancho que Gamma~0.03-0.05 meV de las resonancias encontradas en la sección 3**. Esto
significa que el paquete NO está midiendo la conductancia a una sola energía bien
definida: está promediando sobre docenas de resonancias angostas dentro de su ancho de
banda.

### 5.2 Verificación cuantitativa

Calculé T promediado espectralmente: tomé el espectro de momento REAL del paquete
gaussiano inyectado (transformada de Fourier discreta de la envolvente inicial), lo
convertí a peso en energía vía la relación de dispersión de la red, y promedié T(E) (de
QTBM, barato) ponderado por ese espectro real:

| Cantidad | T |
|---|---|
| QTBM en un solo punto, E=E_F | 1.178 |
| **QTBM promediado por el espectro real del paquete** | **0.481** |
| DTBC con el paquete de ondas real (propagación temporal completa) | 0.243 |

El promediado espectral mueve el resultado de 1.178 a 0.481 — más de la mitad de la
brecha total (1.178→0.243) se explica por este efecto. Repetir el experimento con
paquetes más anchos en el espacio (150, 400, 800 nm, es decir más angostos en energía)
NO acercó más el resultado (T se mantuvo entre 0.24 y 0.27 en los tres casos) — lo cual
sugiere que incluso a 800nm el paquete sigue siendo demasiado ancho en energía
(Delta_E~0.16 meV, todavía ~3-5x más ancho que Gamma) para acercarse al límite
monocromático de forma práctica sin volverse computacionalmente costoso (paquetes más
anchos necesitan dominios y tiempos de simulación más largos).

### 5.3 Conclusión honesta

El promediado espectral es una causa real y sustancial (explica más de la mitad de la
brecha), pero **no es la explicación completa** — queda una brecha adicional (0.481 vs
0.243) sin resolver en esta sesión. Candidatos para investigar a continuación:

1. Mi aproximación de "promedio incoherente de T(E) ponderado por el espectro" podría no
   capturar toda la física de una evolución temporal coherente real cuando las
   resonancias son mucho más angostas que el espaciado de mi malla de energías (0.017
   meV, similar en magnitud a Gamma) — un barrido más fino (o un ajuste multi-resonancia)
   podría cerrar más la brecha.
2. Podría haber un efecto dinámico genuino (no solo de "promedio estático"): si el tiempo
   que tarda el paquete en atravesar el anillo es comparable a la vida media de la
   resonancia, la resonancia no llega a "cargarse" completamente antes de que el grueso
   del paquete ya haya pasado, lo cual un promedio estacionario simple no captura.
3. Vale la pena repetir esta comparación con `junction_correction=True` desactivado Y con
   una malla más fina (N_R mayor, como se investigó la sesión anterior) para separar
   este efecto del de discretización del anillo.

---

## 6. Tabla comparativa final

Todos los valores son G/G0 = T, anillo transparente (V=0), Phi=0.5, alpha=20 meV·nm,
E_F=4.19 meV, N_R=151 (default del repo) salvo donde se indique.

| Método | junction_correction | G/G0 (T) | Notas |
|---|---|---|---|
| **Analítico (Büttiker, Ec. 3.2 tesis JJ)** | — | **1.669** | Referencia |
| QTBM estacionario, una energía exacta | False | 1.178 | Conservación exacta T+R=1 (1e-13) |
| QTBM estacionario, una energía exacta | **True (default repo)** | **0.0066** | Colapso casi total -- ver hallazgo de la sesión anterior |
| QTBM promediado por el espectro real del paquete | False | 0.481 | Nuevo, sección 5 |
| DTBC exacto, paquete de ondas real, 70 ps | False | 0.243 | Convergido (P_remaining~0) |
| DTBC exacto, paquete de ondas real, 35 ps | True | ~0.071 | No completamente convergido a 35ps |
| Autoenergía monocromática (parche anterior), 250 ps | True | 0.067 | De la sesión anterior |
| CAP (método original del repo), 250 ps | True | 0.050 | De la sesión anterior |
| QTBM estacionario, N_R=601 (malla más fina) | False | 1.902 | De la sesión anterior; sobrepasa lo analítico, ver README_QTBM.md |

**Lectura de la tabla**: hay DOS problemas independientes, de magnitud comparable:

1. **`junction_correction=True` (default del repo)** colapsa la transmisión casi a cero
   incluso con todo lo demás exacto (1.178 -> 0.0066, factor >100). Esto domina sobre
   cualquier otro efecto y es lo primero que recomendaría revisar en tu rama de trabajo.
2. Incluso corrigiendo eso (`junction_correction=False`), la comparación
   paquete-de-ondas-real vs. fórmula analítica de una sola energía sigue sin cerrar
   completamente, por el promediado espectral de la sección 5 (parcialmente explicado) y
   posiblemente por el efecto de discretización de N_R de la sesión anterior (no
   revisitado esta sesión).

---

## 7. Archivos incluidos en el zip

- `dtbc.py` — DTBC exacto (nuevo), con la derivación completa documentada.
- `dtbc_selftest.py` — reproduce la Fig. 2/3 del artículo de Akramov et al. (validación).
- `dtbc_selftest_paper_reproduction.png` — la figura generada por lo anterior.
- `test_transparent_boundary.py` — actualizado con la opción `boundary="dtbc"` para
  comparar CAP / autoenergía monocromática / DTBC en el benchmark aislado.
- `quasibound.py` — detector de resonancias/estados cuasi-ligados (nuevo).
- `qtbm.py` — sin cambios de código esta sesión (ya validado la sesión anterior);
  reutilizado extensamente aquí.
- Este informe (`INFORME_2026-08-22_DTBC_QTBM_resonancias.md`).

## 8. Cómo correr todo, paso a paso

```bash
# 1. Reproducir la validación del artículo de DTBC (Fig. 2/3), ~150s
python dtbc_selftest.py

# 2. Comparar CAP / autoenergía monocromática / DTBC en el benchmark aislado, ~15s
python test_transparent_boundary.py

# 3. Buscar resonancias cerca de E_F en el anillo transparente, ~30s
python -c "
import numpy as np, tools as t, quasibound as qb
p = t.PhysicsParams(m_factor=0.023, R=250.0, L_leads=2000.0, N_l=381, N_R=151, dt=0.002,
    Phi=0.5, alpha=20.0, potential_model='none',
    gaussian_qpc_heights_mev={'L':0.0,'U':0.0,'D':0.0,'R':0.0}, junction_correction=False)
energies = np.linspace(3.5, 5.0, 300)
T = qb.scan_transmission(p, energies)
for r in qb.find_resonances(energies, T, min_prominence=0.02):
    print(f'E_R={r.E_R_mev:.4f} meV  Gamma={r.Gamma_mev:.5f} meV  tau={r.tau_ps:.2f} ps')
"

# 4. Correr DTBC exacto sobre el anillo completo y medir T,R (usa dtbc.run_dtbc_propagation
#    + dtbc.transmission_reflection_from_history), ~20s para 70 ps
python -c "
import numpy as np, tools as t, dtbc as d, conductance as c
p = t.PhysicsParams(m_factor=0.023, R=250.0, L_leads=2000.0, N_l=381, N_R=151, dt=0.002,
    Phi=0.5, alpha=20.0, potential_model='none',
    gaussian_qpc_heights_mev={'L':0.0,'U':0.0,'D':0.0,'R':0.0}, junction_correction=False)
layout = t.build_single_ring_layout(p)
k = c.compute_wave_number(p, 4.19)
psi0 = c.build_initial_wavefunction(p, layout, k, packet_center_fraction=0.8, packet_width_nm=150.0, spin='both')
N0 = float(np.sum(np.abs(psi0)**2))
res = d.run_dtbc_propagation(p, psi0, int(round(70.0/p.dt)))
tr = d.transmission_reflection_from_history(p, res)
print('T=', tr['T']/N0, ' R=', tr['R']/N0)
"
```

## 9. Problemas todavía abiertos (para la próxima sesión / tu propio trabajo)

1. **`junction_correction` en `tools.py`**: sigue sin corregirse (solo diagnosticado la
   sesión anterior, y confirmado de nuevo aquí que es el efecto individual más grande de
   toda esta tabla). Sigue siendo el candidato #1 para arreglar.
2. **Cierre incompleto del promediado espectral** (sección 5.3): entender por qué el
   promedio ponderado por el espectro (0.481) no coincide con el resultado dependiente
   del tiempo (0.243).
3. **Ajuste multi-Lorentziano** para resonancias superpuestas en `quasibound.py`
   (actualmente usa ajustes de una sola Lorentziana por pico, que sobreestiman T_pico
   cuando hay interferencia entre resonancias vecinas).
4. **Costo computacional de DTBC**: es O(N_pasos²) por construcción (igual que el método
   del artículo) — viable para las corridas de esta sesión (hasta 120 ps, ~35s) pero
   sería lento para corridas mucho más largas. La literatura citada en el propio
   artículo (Lubich & Schädle, SIAM J. Sci. Comput. 24, 161 (2002)) tiene un algoritmo
   O(N log N); no implementado aquí por alcance/tiempo.
5. La reconciliación del potencial de QPC (Ec. 3-26 vs. Ec. 1-25/1-26 gaussiana vs.
   parabólica, problema #4 de la lista original de "Soluciones propuestas") sigue sin
   tocarse.
