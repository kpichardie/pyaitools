import logging
import subprocess
import asyncio
import json
import os
import sys
from datetime import datetime
from google import genai
from telegram import Update
from telegram.request import HTTPXRequest
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
AUTHORIZED_USER_ID = os.getenv('AUTHORIZED_USER_ID')  # Remplacez par VOTRE ID Telegram (pour que personne d'autre ne contrôle le bot)
TODO_FILE = os.getenv('TODO_FILE')
LOG_FILE = os.getenv('LOG_FILE')
SELF_PATH = "/app/autonomous.py"
MODEL_ID = os.getenv('MODEL_ID') # On utilise le modèle rapide

# Init Gemini
client=genai.Client(api_key=GEMINI_API_KEY)

# --- CONFIGURATION LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Journal pour le rapport horaire
action_log = []

# --- GESTION DES QUOTAS ET GENERATION ---
async def safe_generate(prompt, context: ContextTypes.DEFAULT_TYPE):
    """Génère du contenu avec gestion du quota 429 et push Telegram."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=MODEL_ID, contents=prompt)
        except Exception as e:
            if "429" in str(e):
                wait_time = 15
                logger.warning(f"Quota atteint (429). Tentative {attempt+1}")
                await context.bot.send_message(
                    chat_id=AUTHORIZED_USER_ID,
                    text=f"⏳ **Quota Gemini atteint (429).**\nPause de {wait_time}s avant nouvelle tentative..."
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                logger.error(f"Erreur API : {e}")
                raise e
    return None

# --- GESTION DE LA DB JSON ---
def load_todo():
    if not os.path.exists(TODO_FILE): return []
    try:
        with open(TODO_FILE, "r") as f: return json.load(f)
    except Exception as e:
        logger.error(f"Erreur chargement ToDo: {e}")
        return []

def save_todo(tasks):
    with open(TODO_FILE, "w") as f: json.dump(tasks, f, indent=4)

# --- MOTEUR D'EXÉCUTION ---
async def execute_system_command(command, context):
    logger.info(f"Exécution commande : {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            out = result.stdout if result.stdout else "✅ Succès."
            logger.info(f"Succès : {out[:100]}...")
            return out

        err = result.stderr if result.stderr else result.stdout
        logger.warning(f"Échec commande : {err}")

        prompt = f"La commande `{command}` a échoué : `{err}`. Analyse et donne la correction."
        analysis = await safe_generate(prompt, context)
        return f"❌ ERREUR :\n`{err}`\n\n💡 ANALYSE :\n{analysis.text if analysis else 'Indisponible'}"
    except Exception as e:
        logger.error(f"Erreur critique exécution : {e}")
        return f"⚠️ Erreur critique : {e}"

# --- LOGIQUE DE DÉCISION ET ÉVOLUTION ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        logger.warning(f"Tentative d'accès non autorisée de l'ID : {update.effective_user.id}")
        return

    user_text = update.message.text
    logger.info(f"Message reçu : {user_text}")
    temp_msg = await update.message.reply_text("🤔 *Réflexion de l'agent...*", parse_mode='Markdown')

    decision_prompt = f"""
    Demande utilisateur : "{user_text}"

    Options de réponse :
    1. Modifier un fichier/son code -> SELF_EDIT: [chemin] | [contenu complet]
    2. Lire un fichier -> READ_FILE: [chemin]
    3. Action système immédiate -> COMMAND: [bash]
    4. Ajouter à la ToDo -> TODO_ADD: [description]
    4. ToDo -> TODO_ADD: [desc] / TODO_LIST
    5. Logs -> READ_LOGS
    6. Discussion -> Texte libre.
    """

    try:
        response = await safe_generate(decision_prompt, context)
        if not response:
            await temp_msg.edit_text("❌ Impossible de répondre (Quota épuisé).")
            return
        res_text = response.text

    except Exception as e:
        logger.error(f"Erreur API Gemini : {e}")
        await temp_msg.edit_text("❌ Erreur de communication avec Gemini.")
        return

    # --- TRAITEMENT DES DÉCISIONS ---

    if "SELF_EDIT:" in res_text:
        try:
            parts = res_text.replace("SELF_EDIT:", "").split("|", 1)
            path, content = parts[0].strip(), parts[1].strip()
            if os.path.exists(path): os.rename(path, f"{path}.bak")
            with open(path, "w") as f: f.write(content)
            await temp_msg.edit_text(f"✅ `{path}` mis à jour. Redémarrage...")
            os._exit(0)
        except Exception as e:
            logger.error(f"Erreur auto-édition : {e}")
            await temp_msg.edit_text(f"❌ Erreur auto-édition : {e}")

    elif "READ_FILE:" in res_text:
        path = res_text.replace("READ_FILE:", "").strip()
        try:
            with open(path, "r") as f:
                data = f.read()
            await temp_msg.edit_text(f"📖 **Contenu de `{path}` :**\n```python\n{data[:3900]}\n```", parse_mode='Markdown')
        except Exception as e:
            await temp_msg.edit_text(f"❌ Erreur lecture : {e}")

    elif "READ_LOGS" in res_text:
        try:
            with open(LOG_FILE, "r") as f:
                # On prend les 20 dernières lignes
                lines = f.readlines()[-20:]
                data = "".join(lines)
            await temp_msg.edit_text(f"📜 **Derniers Logs :**\n```\n{data}\n```", parse_mode='Markdown')
        except Exception as e:
            await temp_msg.edit_text(f"❌ Erreur lecture logs : {e}")

    elif "COMMAND:" in res_text:
        cmd = res_text.split("COMMAND:")[1].strip().replace('`', '')
        await temp_msg.edit_text(f"⚙️ *Exécution :*\n`{cmd}`", parse_mode='Markdown')
        out = await execute_system_command(cmd, context)
        action_log.append(f"[{datetime.now().strftime('%H:%M')}] EXEC: {cmd}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"📄 *Résultat :*\n```\n{out[:4000]}\n```", parse_mode='Markdown')

    elif "TODO_ADD:" in res_text:
        desc = res_text.split("TODO_ADD:")[1].strip()
        tasks = load_todo()
        tasks.append({"description": desc, "status": "pending", "at": datetime.now().isoformat()})
        save_todo(tasks)
        logger.info(f"Tâche ajoutée : {desc}")
        await temp_msg.edit_text(f"📝 Ajouté à la ToDo : {desc}")

    elif "TODO_LIST" in res_text:
        tasks = load_todo()
        txt = "\n".join([f"- [{'✅' if t['status']=='done' else '⏳'}] {t['description']}" for t in tasks]) or "Liste vide."
        await temp_msg.edit_text(f"📋 *Ma ToDo List :*\n{txt}", parse_mode='Markdown')

    else:
        await temp_msg.edit_text(res_text)

# --- JOBS DE FOND ---
async def process_todo_job(context: ContextTypes.DEFAULT_TYPE):
    tasks = load_todo()
    pending_tasks = [t for t in tasks if t['status'] == 'pending']

    if not pending_tasks:
        return

    for t in tasks:
        if t['status'] == 'pending':
            logger.info(f"Traitement ToDo : {t['description']}")

            prompt = f"Génère UNIQUEMENT la commande bash pour : {t['description']}"
            res = await safe_generate(prompt, context)
            if res:
              await execute_system_command(res.text.strip().replace('`', ''), context)
              t['status'] = 'done'
              updated = True
              await context.bot.send_message(chat_id=AUTHORIZED_USER_ID, text=f"✅ *Tâche auto-finie :* {t['description']}")
    if updated: save_todo(tasks)

async def hourly_report(context: ContextTypes.DEFAULT_TYPE):
    if not action_log: return
    logs_summary = "\n".join(action_log)
    res = client.models.generate_content(model=MODEL_ID, contents=f"Résume ces actions : {logs_summary}")
    await context.bot.send_message(chat_id=AUTHORIZED_USER_ID, text=f"⏰ *Rapport Horaire*\n{res.text}", parse_mode='Markdown')
    action_log.clear()

# --- MAIN ---
if __name__ == '__main__':
    logger.info("Démarrage de l'agent...")
    # 1. Configuration d'un client réseau ultra-patient
    # connect_timeout : temps pour établir la connexion
    # read_timeout : temps pour recevoir la réponse
    robust_request = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0
    )

    # 2. Construction de l'application avec les paramètres de résilience
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .request(robust_request)
        .build()
    )
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))

    if app.job_queue:
        app.job_queue.run_repeating(process_todo_job, interval=600, first=10)
        app.job_queue.run_repeating(hourly_report, interval=10800, first=60)

    # 5. Lancement avec polling agressif
    logger.info("🚀 Agent en ligne. En attente de messages...")
    app.run_polling(drop_pending_updates=True)

