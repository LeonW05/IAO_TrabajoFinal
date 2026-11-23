# Modelo B — Metadatos y texto

Clustering de los mismos juegos del modelo A, añadiendo reseñas en inglés.
Metacritic y SteamSpy se reservan para la validación externa.

## Seguimiento

- [x] Crear la carpeta y la documentación.
- [x] Definir cómo seleccionar las reseñas y crear el script común
  `scripts/games_reviews.py`.
- [x] Descargar 100 reseñas en inglés por juego (2.000 registros válidos).
- [x] Auditar la cobertura y fijar el umbral de 10 reseñas (1.942 juegos).
- [ ] Crear `prepare_features.py`: preparar la descripción y las reseñas,
  aplicar el umbral de 10 reseñas y combinarlas con los metadatos del
  modelo A.
- [ ] Crear `cluster_metadata_text.py`: probar algoritmos y números de
  grupos sobre la muestra de 1.942 juegos.
- [ ] Ejecutar el Modelo A sobre los 1.942 juegos (`A_sub`) para comparar
  sobre la misma muestra.

**Lo que sigue (en el README principal):** generación de figuras, comparación
A × B (ARI y tabla cruzada), validaciones externas (SteamSpy, Metacritic) y
redacción de conclusiones.
