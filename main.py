import argparse
import logging
from typing import Any

from deepface import DeepFace
import lancedb
import pyarrow as pa

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "yunet"
VECTOR_DIMENSION = 128  # Depends on model
TABLE_NAME = "face_embeddings"

logger = logging.getLogger(__name__)


def run_index(db: Any, img: str, id: str) -> None:
    try:
        represent_rsp = DeepFace.represent(
            img_path=img,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
        )
    except Exception:
        logger.exception("Failed DeepFace.represent")
        return

    if not represent_rsp:
        logger.warning("DeepFace.represent return empty")
        return

    represent_obj = represent_rsp[0] if isinstance(represent_rsp[0], dict) else represent_rsp[0][0]
    represent_embedding = represent_obj.get("embedding")
    if not represent_embedding:
        logger.warning("DeepFace.represent return no embedding")
        return

    try:
        table = db.open_table(TABLE_NAME)
        row = {
            "id": id,
            "vector": represent_embedding,
        }
        table.add([row])
    except Exception:
        logger.exception("Failed lancedb table add")
        return


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_index = subparsers.add_parser("index")
    parser_index.add_argument("--img", required=True)
    parser_index.add_argument("--id", required=True)

    args = parser.parse_args()

    db = lancedb.connect("deepface.lancedb")

    if TABLE_NAME not in db.table_names():
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), VECTOR_DIMENSION)),
            ]
        )
        db.create_table(TABLE_NAME, schema=schema)

    if args.command == "index":
        run_index(db, args.img, args.id)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
