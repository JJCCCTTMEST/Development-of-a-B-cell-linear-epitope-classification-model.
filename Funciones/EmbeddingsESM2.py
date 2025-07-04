import torch
import pandas as pd
from Bio import SeqIO
from typing import Union
from transformers import AutoModel, AutoTokenizer

def generar_embeddings(data: Union[str, pd.DataFrame], 
                        model_name: str, 
                        output_path: str, 
                        col_secuencia: str = None,
                        etiqueta: int = None, 
                        expandir: bool = False):
    """
    Genera embeddings a partir de un archivo FASTA o un DataFrame con secuencias.
    
    Args:
        data (str o pd.DataFrame): Ruta a un archivo FASTA o un DataFrame con secuencias.
        model_name (str): Nombre del modelo de ESM2 a utilizar.
        output_path (str): Ruta del archivo de salida (CSV o JSON).
        col_secuencia (str, opcional): Si se usa un DataFrame, nombre de la columna con las secuencias.
        etiqueta (int, opcional): Etiqueta opcional (1 o 0) a agregar.
        expandir (bool, opcional): Si True, los embeddings se expanden en columnas separadas.

    Returns:
        pd.DataFrame: DataFrame con embeddings añadidos.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data_list = []

    if isinstance(data, str):  # Si es un archivo FASTA
        for record in SeqIO.parse(data, "fasta"):
            sequence_id = record.id
            sequence = str(record.seq)
            data_list.append((sequence_id, sequence))
        df = pd.DataFrame(data_list, columns=["ID", "Secuencia"])
    
    elif isinstance(data, pd.DataFrame):  # Si es un DataFrame
        if col_secuencia is None or col_secuencia not in data.columns:
            raise ValueError("Debes especificar una columna válida con secuencias.")
        df = data.copy()
        df["ID"] = df.index.astype(str)  # Usar el índice como ID si no hay
        df = df.rename(columns={col_secuencia: "Secuencia"})

    else:
        raise ValueError("El parámetro 'data' debe ser un archivo FASTA o un DataFrame.")

    embeddings_list = []
    
    for _, row in df.iterrows():
        sequence_id = row["ID"]
        sequence = row["Secuencia"]

        # Tokenizar la secuencia
        inputs = tokenizer(sequence, return_tensors="pt", padding=True).to(device)

        # Generar embedding
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state

        # Promediar embeddings
        sequence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy().tolist()
        embeddings_list.append(sequence_embedding)

    # Agregar los embeddings al DataFrame
    df["Modelo"] = model_name
    df["Etiqueta"] = etiqueta
    df["Embedding"] = embeddings_list

    if expandir:
        # Expande los embeddings en columnas separadas
        embedding_cols = pd.DataFrame(df["Embedding"].tolist(), index=df.index)
        df = df.drop(columns=["Embedding"]).join(embedding_cols)

    # Guardar archivo de salida
    if expandir:
        df.to_csv(output_path, index=False)
    else:
        df.to_json(output_path, orient="records", lines=True)

    print(f"Embeddings guardados en {output_path}")
    return df



"""
def generar_embeddings_fasta(fasta_path, etiqueta=None, expandir=False):
    
    Genera embeddings para todas las secuencias en un archivo FASTA utilizando ESM2.
    Modelos disponibles: 
    - esm2_t48_15B_UR50D -> 5120 
    - esm2_t36_3B_UR50D -> 2560 
    - esm2_t33_650M_UR50D -> 1280 
    - esm2_t30_150M_UR50D -> 640 
    - esm2_t12_35M_UR50D -> 480 
    - esm2_t6_8M_UR50D -> 320 
    Args:
        fasta_path (str): Ruta al archivo FASTA de entrada.
        etiqueta (int, opcional): Etiqueta a agregar (1 o 0). Por defecto, None.
        expandir (bool, opcional): Si True, expande los embeddings en columnas separadas.

    Returns:
        pd.DataFrame: DataFrame con columnas ["ID", "Modelo", "Etiqueta", "Embedding"] o embeddings expandidos.
    
    data = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequence_id = record.id
        sequence = str(record.seq)

        # Tokenizar la secuencia
        inputs = tokenizer(sequence, return_tensors="pt", padding=True).to(device)

        # Generar embedding
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state

        # Promediar los embeddings
        sequence_embedding = embeddings.mean(dim=1).squeeze().cpu().numpy().tolist()

        # Agregar al dataset
        row = {"ID": sequence_id, "Modelo": model_name, "Etiqueta": etiqueta, "Embedding": sequence_embedding}
        data.append(row)

    df = pd.DataFrame(data)

    if expandir:
        # Expandir embeddings en columnas separadas
        embedding_cols = pd.DataFrame(df["Embedding"].tolist(), index=df.index)
        df = df.drop(columns=["Embedding"]).join(embedding_cols)

    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generar embeddings de proteínas con ESM2")
    parser.add_argument("fasta_path", help="Ruta al archivo FASTA de entrada")
    parser.add_argument("--etiqueta", type=int, choices=[0, 1], help="Etiqueta opcional (1 o 0)", default=None)
    parser.add_argument("--output", help="Ruta del archivo CSV de salida", default="embeddings.csv")
    parser.add_argument("--expandir", action="store_true", help="Expande los embeddings en columnas separadas")

    args = parser.parse_args()

    # Generar embeddings
    df = generar_embeddings_fasta(args.fasta_path, args.etiqueta, args.expandir)

    # Guardar en CSV sin truncar
    if args.expandir:
        df.to_csv(args.output, index=False)
    else:
        df.to_json(args.output, orient="records", lines=True)  # Guardado en JSON si no se expanden
    
    print(f"Embeddings guardados en {args.output}")
"""