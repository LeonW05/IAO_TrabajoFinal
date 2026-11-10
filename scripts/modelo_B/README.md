# Modelo B — Metadatos y texto

Clustering de los mismos juegos del modelo A, añadiendo reseñas en inglés.
Metacritic y SteamSpy se reservan para la validación externa.

## Seguimiento

- [x] Crear la carpeta y la documentación.
- [ ] Definir cómo seleccionar 30–50 reseñas por juego y descargarlas con
  el futuro script común `scripts/games_reviews.py`.
- [ ] Crear `prepare_features.py`: preparar el texto y combinarlo con los
  metadatos del modelo A; decidir qué hacer si faltan reseñas.
- [ ] Crear `cluster_metadata_text.py`: probar algoritmos y números de grupos.
- [ ] Evaluar silhouette, Davies-Bouldin y estabilidad entre semillas.
- [ ] Guardar y describir los grupos y compararlos con el modelo A.

Todavía no hay scripts del modelo. Usará `data/interim/final_sample.csv`
y sus reseñas descargadas.
