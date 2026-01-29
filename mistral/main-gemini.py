import sys
import os
import argparse
import logging
from typing import Optional
from dataclasses import dataclass

from mistralai import Mistral
from dotenv import load_dotenv

# Configuration du logging pour les erreurs
logging.basicConfig(format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Constantes
DEFAULT_MODEL = "mistral-small-latest" # "tiny" est souvent déprécié, "small" est un bon compromis
DEFAULT_TEMPERATURE = 0.7

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interface CLI pour Mistral AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("prompt", nargs="*", help="Le prompt à envoyer (peut être combiné avec stdin)")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Modèle Mistral à utiliser")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Température (0.0 à 1.0)")
    parser.add_argument("-f", "--output-file", help="Sauvegarder la réponse dans un fichier")
    parser.add_argument("-l", "--max-tokens", type=int, help="Nombre maximum de tokens à générer")
    parser.add_argument("-s", "--no-stream", action="store_true", help="Désactiver l'affichage en streaming")
    
    return parser.parse_args()

def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        logger.error("La variable d'environnement MISTRAL_API_KEY est manquante.")
        sys.exit(1)
    return api_key

def get_full_content(args_prompt: list) -> str:
    """Combine stdin et les arguments du prompt."""
    input_data = []
    
    # Lecture depuis stdin (pipe)
    if not sys.stdin.isatty():
        input_data.append(sys.stdin.read().strip())
    
    # Ajout du prompt passé en argument
    if args_prompt:
        input_data.append(" ".join(args_prompt))
        
    return "\n\n".join(input_data).strip()

def main() -> None:
    args = parse_args()
    
    # Validation rapide
    if not (0 <= args.temperature <= 1):
        logger.error("La température doit être comprise entre 0.0 et 1.0")
        sys.exit(1)

    full_content = get_full_content(args.prompt)
    if not full_content:
        logger.error("Aucun input fourni (utilisez un argument ou un pipe stdin).")
        sys.exit(1)

    client = Mistral(api_key=get_api_key())
    
    messages = [{"role": "user", "content": full_content}]

    try:
        # Mode Streaming (Recommandé pour CLI)
        if not args.no_stream and not args.output_file:
            stream_response = client.chat.stream(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            for chunk in stream_response:
                content = chunk.data.choices[0].delta.content
                if content:
                    print(content, end="", flush=True)
            print() # Saut de ligne final
            
        # Mode standard (pour fichier ou si désactivé)
        else:
            chat_response = client.chat.complete(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            result = chat_response.choices[0].message.content
            
            if args.output_file:
                with open(args.output_file, "w", encoding="utf-8") as f:
                    f.write(result)
                print(f"Réponse enregistrée dans : {args.output_file}")
            else:
                print(result)

    except Exception as e:
        logger.error(f"Erreur lors de l'appel à Mistral : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

