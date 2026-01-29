import sys
import os
import argparse
from mistralai import Mistral
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Mistral CLI Tool (Pipe compatible)")
    parser.add_argument("prompt", nargs="*", help="Le prompt à envoyer à Mistral")
    parser.add_argument("-m", "--model", default="mistral-tiny", help="Modèle Mistral à utiliser")
    parser.add_argument("-t", "--temperature", type=float, default=0.7, help="Température (0.0 à 1.0)")
    
    args = parser.parse_args()

    # Récupération de la clé API
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Erreur: La variable d'environnement MISTRAL_API_KEY est manquante.")
        sys.exit(1)

    client = Mistral(api_key=api_key)

    # Lecture du pipe (stdin) s'il y en a un
    pipe_input = ""
    if not sys.stdin.isatty():
        pipe_input = sys.stdin.read()

    # Construction du prompt final
    user_prompt = " ".join(args.prompt)
    full_content = f"{pipe_input}\n\n{user_prompt}".strip()

    if not full_content:
        parser.print_help()
        sys.exit(0)

    try:
        chat_response = client.chat.complete(
            model=args.model,
            messages=[
                {"role": "user", "content": full_content},
            ],
            temperature=args.temperature
        )
        print(chat_response.choices[0].message.content)
    except Exception as e:
        print(f"Erreur lors de l'appel à Mistral: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

