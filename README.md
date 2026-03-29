# Reinforcement Learning Homework

Repositorio correspondiente a la tarea de Aprendizaje por Refuerzo.

## Contenido

Se implementan y comparan los siguientes métodos:

- DDQN (implementación propia)
- DQN oficial de Stable-Baselines3 como baseline de comparación
- PPO (implementación propia)
- PPO oficial de Stable-Baselines3
- NES
- GA

## Problemas considerados

- MinAtar Breakout: `MinAtar/Breakout-v1`
- Atari Breakout: `ALE/Breakout-v5`

## Estructura del proyecto

- `src/common/`: utilidades y entornos
- `src/ddqn/`: implementación y evaluación de DDQN
- `src/ppo/`: implementación y evaluación de PPO
- `src/evolutionary/`: implementación y evaluación de NES/GA
- `src/analysis/`: comparación global de métodos

## Instalación


```bash
pip install -r requirements.txt
```

## Ejemplos de ejecución

### DDQN propio en MinAtar:

```bash
python -m src.ddqn.train_ddqn --env minatar --seed 0 --total_steps 200000
```

### DQN oficial SB3 en MinAtar:

```bash
python -m src.ddqn.train_sb3_dqn --env minatar --seed 0 --total_steps 200000
```


### PPO propio en MinAtar:

```bash
python -m src.ppo.train_ppo --env minatar --seed 0 --total_steps 200000
```

### PPO oficial SB3 en MinAtar:

```bash
python -m src.ppo.train_sb3_ppo --env minatar --seed 0 --total_steps 200000
```

### NES en MinAtarr:

```bash
python -m src.evolutionary.train_nes_ga --method nes --env minatar --seed 0 --generations 200
```

### GA en MinAtar:

```bash
python -m src.evolutionary.train_nes_ga --method ga --env minatar --seed 0 --generations 200
```
### Comparación global:

```bash
python -m src.analysis.compare_all_methods --results_dir results --output_dir analysis_outputs
```

## Resultados esperados

Los scripts generan:

- curvas de convergencia
- estadísticas descriptivas
- comparaciones por pares con Wilcoxon rank-sum
- ranking promedio
- diagrama de diferencias críticas
## Nota metodológica
Stable-Baselines3 no incluye una implementación oficial de Double-DQN.
Por ello, la comparación correspondiente se realiza entre:

- DDQN implementado desde cero
- DQN oficial de Stable-Baselines3

## Reproducibilidad

Se recomienda utilizar el notebook Tarea_3_RL.ipynb, el cual clona este repositorio, instala dependencias y ejecuta los experimentos principales.