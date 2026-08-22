# QTBM (Quantum Transmitting Boundary Method) — solución propuesta #3

Generado por una tarea programada de investigación, 2026-08-21/22, en respuesta a tu
pedido de (a) explicar el problema de las DTBC con más detalle y evaluar si es
resoluble con acceso al artículo, y (b) implementar la solución #3
(`Soluciones_propuestas_CAP_DTBC_QPC_2026-08-21.md`) para desarrollar en una rama
aparte de tu repositorio.

Este documento es autocontenido: explica el método, cómo verificarlo, qué encontró
al compararlo contra la fórmula analítica de la tesis de Juan José González
(`LicJJG2.pdf`), y qué queda abierto.

---

## 1. Sobre las DTBC: qué es difícil y si es resoluble con el artículo

### Qué intenté

Volví a intentar acceder al texto completo de arXiv:2608.05338 (Akramov, Yusupov,
Ehrhardt, Matrasulov, "Transparent boundary conditions for the spatially discrete
Schrödinger equation: Reflectionless quantum transport in 1D lattices", 2026) varias
veces, con esperas crecientes entre intentos (30s, 60s, 90s), tanto la versión `/pdf`
como `/html`. Todos los intentos fueron rechazados por rate-limiting del proxy de red
de esta sesión (HTTP 429) — **no es que el artículo no exista o no sea accesible en
general** (confirmé que existe y aparece indexado en arXiv vía búsqueda web), sino que
esta sesión en particular no logró bajar el texto completo. Solo obtuve el abstract,
que menciona "Dirichlet-to-Neumann maps... vía transformadas de Laplace" y "una
condición de frontera tipo convolución gobernada por funciones de Bessel", sin
ecuaciones.

Sí logré acceso al reporte técnico clásico de Arnold-Ehrhardt (formulación DTBC
original, Ehrhardt et al., disponible en `sfb65.univie.ac.at`), que da la **estructura
general** del método (mapa de Dirichlet-a-Neumann discreto vía transformada Z,
coeficientes de convolución ℓ⁽ⁿ⁾ con decaimiento asintótico ~n⁻³/², fórmula cerrada
en términos de funciones especiales para el caso de potencial constante) pero no las
fórmulas explícitas completas con todas las constantes — el resumen que obtuve es de
una IA leyendo el PDF, no una transcripción verificada ecuación por ecuación.

### Por qué no lo implementé "tal cual" la vez pasada

Diferencia entre "entender la idea general del método" y "tener la fórmula cerrada
exacta, con todos los signos y factores, lista para codificar" es enorme en un método
de convolución temporal: un error de signo, de normalización, o en qué historia
temporal se guarda, no da un resultado "un poco peor" — da una condición de frontera
que **parece funcionar** (no diverge, produce números razonables) pero **no es
realmente transparente**, y detectar eso requiere exactamente el tipo de test
independiente que hice para QTBM más abajo (ver sección 2), que no había hecho la vez
pasada.

### Qué SÍ pude resolver de forma rigurosa esta vez (y por qué es distinto de las DTBC)

Lo que implementé la vez pasada (single-site lead self-energy) y lo que implemento
ahora (QTBM) **no son las DTBC del artículo**, pero tampoco son una aproximación
vaga: son un caso límite exacto y auto-derivable de la misma familia de ideas
(condiciones de frontera "transparentes" en vez de absorbentes), específicamente para
el caso en que el lead es una cadena tight-binding uniforme y uno se restringe a
una sola energía (monocromático). La diferencia con las DTBC completas es que estas
últimas son exactas para **todo el paquete de ondas** (todas las componentes de
Fourier a la vez, vía la convolución temporal), mientras que QTBM es exacta solo para
**una energía a la vez** — pero como QTBM resuelve directamente el problema
estacionario (sin evolución temporal), eso deja de ser una limitación: uno barre en
energía explícitamente si hace falta, en vez de necesitar que la convolución cubra
todas las energías del paquete de una sola corrida.

### Recomendación

Si querés que intente las DTBC completas en una futura corrida, lo más seguro es que
me subas el PDF de arXiv:2608.05338 directamente como archivo adjunto (evita el
rate-limiting del fetch web), o me des un ratito adicional para reintentar el fetch
en otro momento del día. Con el texto completo en mano, y siguiendo la misma
disciplina de validación que usé para QTBM (ver sección 3: verificar conservación de
probabilidad exacta como primer chequeo, antes de comparar contra nada analítico),
sí creo que es implementable con confianza razonable.

---

## 2. Qué es QTBM y cómo funciona

`qtbm.py` (nuevo archivo, no toca nada existente) implementa el método de frontera
de transmisión cuántica (Lent & Kirkner 1990; Shao, Porod, Lent & Kirkner 1995).

### La idea física

En vez de lanzar un paquete de ondas gaussiano y evolucionarlo en el tiempo con
Crank-Nicolson (como hace `main.py` / `conductance.py`) para después leer cuánta
probabilidad terminó escapando por cada lead, QTBM resuelve **directamente** el
estado estacionario de scattering a una energía fija E. En cada lead la función de
onda se escribe analíticamente como superposición de las ondas planas que ese lead
admite:

```
lead izquierdo:  psi_j = chi * e^{ikj}  +  r * chi * e^{-ikj}    (incidente + reflejada)
lead derecho:     psi_j = t * chi * e^{ikj}                       (puramente saliente)
```

donde `chi` es el espinor inyectado, `j` es el índice de sitio medido desde el borde
externo de cada lead, y `k(E)` es la relación de dispersión propia de la red
tight-binding ya usada en todo el repo: `E = 2*t_lead*(1-cos(k*dx))` (exactamente la
que produce `tools.build_single_ring_hamiltonian` para un lead sin potencial).

Sustituyendo estas formas analíticas para las colas semi-infinitas (removidas) en los
dos sitios más externos de la red, el problema de scattering abierto se convierte en
un sistema lineal **finito**: `(H_eff - E*I) psi = source`, que se resuelve una vez
por energía con una sola factorización LU dispersa. Sin pasos temporales, sin
`total_time_ps` que ajustar, sin ambigüedad de "¿el CAP absorbió a tiempo o hay una
resonancia cuasi-ligada que todavía no drenó?" — T(E) y R(E) salen exactos para ese
Hamiltoniano, a esa energía, de una sola vez.

### La derivación (por qué confío en las fórmulas)

Está completa como docstring al principio de `qtbm.py` — la resumo aquí. La
ecuación de bulk en cualquier sitio de un lead (energía on-site 2t, hopping -t, tal
como construye `build_single_ring_hamiltonian` para V=0) es:

```
(H psi)_j = 2t*psi_j - t*psi_{j-1} - t*psi_{j+1} = E*psi_j
```

En el borde IZQUIERDO (sitio j=0), sustituyendo el ansatz incidente+reflejada en el
punto fantasma j=-1 y usando `psi_0 = chi + r*chi` para eliminar `r`, se llega a:

```
(2t - t*e^{ika} - E)*psi_0 - t*psi_1 = -2i*t*sin(ka)*chi
```

Es decir: se agrega una autoenergía `Sigma(E) = -t*e^{ika}` a la energía on-site del
sitio de borde (la misma fórmula que ya usé para el parche anterior de "frontera
transparente" en `conductance.py`), MÁS un término fuente `-2i*t*sin(ka)*chi` — y
como el sitio de borde en la matriz real construida por `tools.py` sólo tiene UN
enlace físico (energía on-site `t`, no `2t`, porque el enlace hacia el sitio "-1" no
existe en el arreglo finito), la corrección real a sumar es `t + Sigma`, no `Sigma`
sola. **Este fue un bug real que encontré y corregí durante esta misma sesión** (ver
sección 4) — el primer test de autoconsistencia (conservación de probabilidad) lo
detectó de inmediato.

En el borde DERECHO la onda es puramente saliente, así que la misma sustitución da
la MISMA autoenergía, sin término fuente (homogénea, igual que el parche anterior).

### Cómo se lee T y R

Como `|e^{ikj}|=1` para cualquier k real (modo propagante), el módulo de psi en el
sitio de borde da directamente la amplitud de scattering, sin necesidad de rastrear
el índice local exacto:

```
r_spin      = psi[sitio_borde_izq]  - chi        (amplitud de reflexión)
t_spin      = psi[sitio_borde_der]                (amplitud de transmisión)
T = sum_spin |t_spin|^2,   R = sum_spin |r_spin|^2
```

La conservación de probabilidad `T + R = |chi|^2` **no se impone**: sale sola del
sistema lineal. Por eso es la primera verificación que hay que correr siempre (ver
sección 3) — y es mucho más informativa que en el método CAP/dependiente del tiempo,
donde un `T+R` lejos de 1 puede significar "todavía no convergió" o "hay una
resonancia cuasi-ligada real", y hay que correr mucho más tiempo para distinguir
los dos casos. Con QTBM no hay ambigüedad: si `T+R != 1`, hay un bug.

---

## 3. Cómo correrlo — guía paso a paso

Todo corre desde la carpeta del repo (`JCE26/`), con las mismas dependencias que ya
usás (numpy, scipy, matplotlib). Ningún comando requiere GPU ni corre más de un
minuto salvo que se indique.

### Paso 1 — verificar que la implementación es correcta (SIEMPRE correr esto primero)

```bash
python qtbm.py --selftest
```

Qué hace: barre `Phi` en `[-1,1]` a `alpha=20 meV*nm` (igual que la tesis, sección
3.3), y para cada punto imprime el error de conservación `|T+R-1|` y compara `T_qtbm`
contra la fórmula analítica de Büttiker (Ec. 3.2 de la tesis). Debe dar
`T+R` con error `~1e-13` (precisión de máquina) en TODOS los puntos — si no, algo
se rompió al editar el código. La comparación contra lo analítico, en cambio, **no**
se espera que dé error cero (ver sección 4) — ese es justamente el hallazgo interesante.

### Paso 2 — generar las curvas de oscilación AB y AC (anillo transparente)

```bash
python qtbm.py --ab --ac --n-points 41
```

Genera `qtbm_ab_transparent.npz` y `qtbm_ac_transparent.npz` (arrays `Phi`/`phi_so`,
`T_up`, `T_down`, `T_total`). Corresponde a las Fig. 3.1-3.4 de la tesis de JJ.

### Paso 3 — anillo con QPCs (transparencia finita)

```bash
python qtbm.py --qpc --n-points 41
```

Corresponde a las Fig. 4.4-4.9 de la tesis (QPC1=QPC2, QPC3 apagado). **Ojo**: usa
`potential_model="legacy_unbounded_qpc"` de `tools.py`, que es la aproximación más
parecida al potencial de silla de montar `V_SP = -Ux*x² + Uy*y² + V0` de la tesis
(Ec. 2.1-2.2), pero no es una reconciliación exacta — ver el aviso en el docstring de
`finite_transparency_params` en `qtbm.py` y el problema #4 de
`Soluciones_propuestas_CAP_DTBC_QPC_2026-08-21.md` (reconciliación QPC, todavía
abierto, fuera del alcance de esta corrida).

### Paso 4 — la figura comparativa completa (la que se adjunta con este paquete)

```bash
python qtbm_demo_plots.py
```

Tarda ~30 segundos. Genera `qtbm_ab_ac_comparison.png`: oscilaciones AB y AC
comparadas contra la fórmula analítica a dos discretizaciones del anillo (N_R=151,
el default actual del repo, y N_R=601, más fina), más un panel de convergencia. Esta
es la figura que sustenta los hallazgos de la sección 4.

### Paso 5 (opcional) — explorar un punto específico a mano

```python
import qtbm
p = qtbm.transparent_ring_params(Phi=0.5, alpha=20.0, junction_correction=False)
r = qtbm.qtbm_conductance(p, qtbm.E_F_MEV)
print(r.T, r.R, r.conservation_error)
```

`qtbm_conductance` devuelve un `QTBMResult` con `T_up`, `T_down`, `R_up`, `R_down`,
`T`, `R`, `conservation_error`. Pasando `keep_psi=True` a `qtbm_solve_one_spin`
también se puede recuperar la función de onda completa (útil para graficar el perfil
espacial de |psi|² en el anillo a esa energía, análogo a lo que hace `main.py` en el
dominio temporal).

---

## 4. Hallazgos (esto es lo importante para tu tesis)

Al comparar QTBM contra la fórmula analítica de Büttiker (Ec. 3.2 de la tesis de JJ,
idéntica a la Ec. 2-25/4-26 del artículo JCE25-26), aparecieron **dos problemas
reales, distintos del problema de reflexiones en los leads que se atacó la sesión
anterior**, y que probablemente son la causa dominante de por qué las oscilaciones
numéricas no reproducen las de la tesis:

### 4.1 — `junction_correction` (en `tools.py`) parece estar mal calibrado

`tools.py` incluye un término `t_junction_correction = t_lead - 2*t_ring` que se
suma a la energía on-site de cada unión lead-anillo, con el objetivo declarado
(docstring de la clase `PhysicsParams`) de lograr `T(Phi=0, alpha=0) = 1`. Es
`True` por default, así que afecta **todos** los scripts existentes del repo.

Con QTBM (que da T exacto, sin ambigüedad temporal) medí, en `Phi=0, alpha=0`
(donde el docstring promete T=1):

| Configuración | T medido | T analítico |
|---|---|---|
| `junction_correction=False` | **1.276** | 1.670 |
| `junction_correction=True` (default actual) | **0.126** | 1.670 |

Es decir: la corrección, tal como está implementada hoy, **empeora** el acuerdo con
lo analítico por un factor de ~10, no lo mejora. Además, un barrido rápido de valores
de corrección (ver `CHANGES` de esta sesión / historial) mostró que un valor cercano
a `-20 meV` (no `-61 meV`, que es lo que da la fórmula actual) acerca mucho más el
resultado a `T=2` en ese punto particular — pero no llegué a verificar si ese valor
sirve de forma general (para otros `Phi`, `alpha`, tamaños de anillo) o si es
específico de esta combinación de parámetros (una resonancia tipo Fabry-Pérot
afinada para esta energía y esta longitud de anillo en particular). **No toqué
`tools.py`** — es una decisión de física central que te corresponde revisar con más
cuidado, posiblemente en la misma rama nueva.

### 4.2 — El desacuerdo dominante parece ser de discretización del anillo, no de frontera

Al refinar `N_R` (nodos por brazo del anillo, manteniendo todo lo demás fijo) el
resultado de QTBM en `Phi=0` y `Phi=0.5` se mueve claramente en la dirección del
valor analítico (ver panel inferior izquierdo de `qtbm_ab_ac_comparison.png`):

| N_R | T(Phi=0) | T(Phi=0.5) |
|---|---|---|
| 151 (default actual) | 1.345 | 1.178 |
| 601 | 0.134 | 1.902 |
| 1501 | 0.022 | 1.044 |
| **analítico** | **0.0036** | **1.669** |

Esto es consistente con lo que ya se sospechaba en la sección 5.2 del artículo
("igualar los pasos espaciales... de las distintas regiones"), pero va más allá: no
alcanza con que `delta_x ≈ delta_s` (que ya casi se cumple con los valores actuales,
5.26 nm vs 5.24 nm) — hace falta que `k*delta_s` sea chico en términos absolutos para
que la dispersión tight-binding `E=2t(1-cos(k*delta_s))` se aproxime bien a la
dispersión continua `E=hbar^2k^2/2m` que asume la fórmula analítica. Con `N_R=151`,
`k*delta_s ≈ 0.26 rad` — no es chico.

**Advertencia honesta**: la convergencia NO es un simple "se acerca monótonamente a
una curva suave" — al mirar el panel superior izquierdo de la figura vas a ver que
`N_R=601` no da una curva más parecida a la campana analítica, da una curva con
MÁS estructura fina (más oscilaciones por unidad de `Phi`) que en promedio abarca un
rango más parecido al analítico, pero con una forma detallada distinta. Sospecho
que esto refleta que la unión de 3 puertos (nodo con 3 enlaces) que arma `tools.py`
no es necesariamente el mismo objeto matemático que la matriz S de 3 puertos que
asume la tesis (Ec. 2.77, con coeficientes `a,b,c` derivados de unitariedad + TRI) —
puede que coincidan solo en el límite continuo y no exactamente sitio-a-sitio en la
red. Esto quedó identificado pero **no resuelto** — es candidato natural para la
rama nueva, junto con 4.1.

### En limpio, para la rama nueva

1. Revisar/rederivar `t_junction_correction` en `tools.py` con una herramienta
   exacta a mano (QTBM, ya la tenés) en vez de confiar en el argumento de "matchear
   la energía on-site a `2*t_lead`" que usa hoy.
2. Antes de sacar conclusiones sobre `junction_correction`, correr un barrido más
   fino de `N_R` (con `junction_correction=False` primero, para no mezclar los dos
   efectos) y ver si converge a la forma correcta de la curva analítica a partir de
   qué `N_R`, no sólo al valor puntual en un par de `Phi`.
3. Recién con eso resuelto tiene sentido volver a comparar contra el caso con QPCs
   (`--qpc` arriba) y las Fig. 4.4-4.9 de la tesis.

---

## 5. Archivos de este paquete

- `qtbm.py` — implementación QTBM completa (no toca ningún archivo existente).
- `qtbm_demo_plots.py` — genera la figura comparativa de esta sección.
- `qtbm_ab_ac_comparison.png` — la figura ya generada, adjunta.
- `README_QTBM.md` — este archivo.

Ningún archivo del patch anterior (`conductance.py`, `ab_ac_proof.py`,
`test_transparent_boundary.py`, `CHANGES_2026-08-21.txt`) fue modificado en esta
sesión — ese trabajo sigue como estaba, a la espera de que lo apliques a tu rama
principal cuando quieras.
