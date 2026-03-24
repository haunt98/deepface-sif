import argparse
import logging

from deepface import DeepFace

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "yunet"
DISTANCE_METRIC = "cosine"

logger = logging.getLogger(__name__)


def run_index(args: argparse.Namespace) -> None:
    try:
        represent_rsp = DeepFace.represent(
            img_path=args.img,
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

    logger.info(represent_embedding)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_index = subparsers.add_parser("index")
    parser_index.add_argument("--img", required=True)
    parser_index.add_argument("--id", required=True)

    args = parser.parse_args()

    if args.command == "index":
        run_index(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
