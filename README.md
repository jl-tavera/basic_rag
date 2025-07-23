# Basic RAG 



## Data Ingestion

Para la etapa de ingestión documental se implementó una clase orientada a objetos llamada `DoclingLoader`, diseñada para encapsular y automatizar todo el proceso de lectura, limpieza, segmentación y chunking de documentos PDF, utilizando como backend la biblioteca `Docling`.

### Configuración Parametrizable

Los valores de configuración utilizados por la clase `DoclingLoader` se definen de manera externa en un archivo `config.json`, lo cual permite realizar experimentos fácilmente (por ejemplo, con MLflow) variando parámetros como el tamaño de los fragmentos o el tokenizer utilizado.


### Métodos de la clase `DoclingLoader`

- **`__init__`**  
  - Inicializa el cargador con parámetros como `tokenizer`, `chunk_size`, `max_tokens`, etc.  
  - Permite personalizar el proceso de parsing, limpieza y fragmentación.

- **`_clean_text(text: str) -> str`**  
  - Limpia caracteres especiales, espacios extra y saltos de línea del texto.

- **`_filter_chunks(chunks: List[str]) -> List[str]`**  
  - Elimina fragmentos con menos tokens que el umbral mínimo (`min_token_threshold`).

- **`_chunk_with_overlap(text: str) -> List[str]`**  
  - Aplica segmentación con ventana deslizante (`chunk_size`) y solapamiento (`chunk_overlap`).  
  - Útil para mantener contexto en fragmentos largos.

- **`load() -> List[str]`**  
  - Flujo principal de carga: parsea el PDF, serializa, limpia, filtra y fragmenta.  
  - Devuelve una lista de fragmentos textuales listos para embeddings o búsqueda semántica.

