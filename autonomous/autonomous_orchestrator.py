"""
Agent Autonomous - Version Améliorée
Scheduler intelligent avec Telegram, SQLite, et gestion de tâches avec retry
Basé sur votre architecture existante avec Gemini AI
"""

import httpx
import logging
import subprocess
import asyncio
import json
import os
import sys
import sqlite3
import threading
import telegram
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager

from google import genai
from telegram import Update
from telegram.request import HTTPXRequest
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

# --- CONFIGURATION ---
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
AUTHORIZED_USER_ID = os.getenv('AUTHORIZED_USER_ID')
DB_FILE = os.getenv('DB_FILE', '/app/tasks.db')
LOG_FILE = os.getenv('LOG_FILE', '/app/autonomous.log')
MODEL_ID = os.getenv('MODEL_ID', 'gemini-2.0-flash-exp')
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '3'))
RETRY_DELAY_MINUTES = int(os.getenv('RETRY_DELAY_MINUTES', '5'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
FORCED_AI_REPORT = os.getenv('FORCED_AI_REPORT', False)

# Init Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# --- CONFIGURATION LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Journal pour le rapport
action_log = []


# --- ENUMS ET DATACLASSES ---
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskType(Enum):
    COMMAND = "command"
    FILE_EDIT = "file_edit"
    FILE_READ = "file_read"
    AI_ANALYSIS = "ai_analysis"
    PLAN = "plan"


@dataclass
class Task:
    id: Optional[int]
    user_id: int
    description: str
    task_type: TaskType
    parameters: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    updated_at: datetime
    retry_count: int
    max_retries: int
    parent_task_id: Optional[int]
    error_message: Optional[str]
    result: Optional[str]
    scheduled_at: Optional[datetime]


# --- DATABASE ---
class TaskDatabase:
    """Gestion de la base de données SQLite des tâches"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialise la base de données"""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    description TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    parent_task_id INTEGER,
                    error_message TEXT,
                    result TEXT,
                    scheduled_at TEXT,
                    FOREIGN KEY (parent_task_id) REFERENCES tasks(id)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON tasks(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_scheduled ON tasks(scheduled_at)")
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Context manager pour les connexions DB"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        except Exception as err:
            logger.error(f"❌ Échec sqlite : {err}")
        finally:
            conn.close()

    def create_task(self, task: Task) -> int:
        """Crée une nouvelle tâche"""
        with self.lock:
            with self._get_connection() as conn:
                cursor = conn.execute("""
                    INSERT INTO tasks (
                        user_id, description, task_type, parameters, status,
                        created_at, updated_at, retry_count, max_retries,
                        parent_task_id, scheduled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task.user_id, task.description, task.task_type.value,
                    json.dumps(task.parameters), task.status.value,
                    task.created_at.isoformat(), task.updated_at.isoformat(),
                    task.retry_count, task.max_retries, task.parent_task_id,
                    task.scheduled_at.isoformat() if task.scheduled_at else None
                ))
                conn.commit()
                return cursor.lastrowid

    def update_task(self, task: Task):
        """Met à jour une tâche"""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE tasks SET
                        status = ?, updated_at = ?, retry_count = ?,
                        error_message = ?, result = ?, scheduled_at = ?
                    WHERE id = ?
                """, (
                    task.status.value, datetime.now().isoformat(),
                    task.retry_count, task.error_message, task.result,
                    task.scheduled_at.isoformat() if task.scheduled_at else None,
                    task.id
                ))
                conn.commit()

    def get_task(self, task_id: int) -> Optional[Task]:
        """Récupère une tâche par ID"""
        with self.lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
                if row:
                    return self._row_to_task(row)
                return None

    def get_pending_tasks(self) -> List[Task]:
        """Récupère les tâches prêtes à être exécutées"""
        with self.lock:
            with self._get_connection() as conn:
                now = datetime.now().isoformat()
                rows = conn.execute("""
                    SELECT * FROM tasks
                    WHERE status = ?
                    AND (scheduled_at IS NULL OR scheduled_at <= ?)
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (TaskStatus.PENDING.value, now, MAX_WORKERS)).fetchall()
                return [self._row_to_task(row) for row in rows]

    def get_user_tasks(self, user_id: int, limit: int = 10) -> List[Task]:
        """Récupère les tâches d'un utilisateur"""
        with self.lock:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT * FROM tasks
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (user_id, limit)).fetchall()
                return [self._row_to_task(row) for row in rows]

    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        """Récupère les tâches par statut"""
        with self.lock:
            with self._get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status = ?",
                    (status.value,)
                ).fetchall()
                return [self._row_to_task(row) for row in rows]

    def _row_to_task(self, row) -> Task:
        """Convertit une row SQL en Task"""
        return Task(
            id=row['id'],
            user_id=row['user_id'],
            description=row['description'],
            task_type=TaskType(row['task_type']),
            parameters=json.loads(row['parameters']),
            status=TaskStatus(row['status']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            retry_count=row['retry_count'],
            max_retries=row['max_retries'],
            parent_task_id=row['parent_task_id'],
            error_message=row['error_message'],
            result=row['result'],
            scheduled_at=datetime.fromisoformat(row['scheduled_at']) if row['scheduled_at'] else None
        )

    def search_tasks(self, query: str, limit: int = 5) -> List[Task]:
        """Recherche des informations dans les tâches passées"""
        with self.lock:
            with self._get_connection() as conn:
                search_term = f"%{query}%"
                rows = conn.execute("""
                    SELECT * FROM tasks 
                    WHERE (description LIKE ? OR result LIKE ?) 
                    AND status = 'completed'
                    ORDER BY updated_at DESC 
                    LIMIT ?
                """, (search_term, search_term, limit)).fetchall()
                return [self._row_to_task(row) for row in rows]

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log l'erreur et prévient l'utilisateur."""
    logger.error(f"Exception lors de la gestion d'un update: {context.error}")
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            f"⚠️ **Erreur Interne :**\n`{str(context.error)[:100]}`"
        )

# --- GESTION DES QUOTAS AI (votre code original) ---
async def safe_generate(prompt: str, context: ContextTypes.DEFAULT_TYPE) -> Optional[Any]:
    """Génère du contenu avec gestion du quota 429"""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=MODEL_ID, contents=prompt).text
        except Exception as e:
            logger.warning(f"⚠️ Gemini indisponible ({e}). Bascule sur Ollama local...")

        # 2. Fallback sur Ollama (Local)
        try:
            # On utilise le client Ollama pointant vers le conteneur ollama
            async with httpx.AsyncClient(timeout=120.0) as http_client:
               payload = {
                   "model": OLLAMA_MODEL,
                   "prompt": prompt,
                   "stream": False
               }
               res = await http_client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
               res.raise_for_status()
               data = res.json()
            
               # On ajoute une petite notification pour savoir qu'on est en mode dégradé
               await context.bot.send_message(
                   chat_id=AUTHORIZED_USER_ID,
                   text="🏠 *Note : Gemini est hors ligne, réponse générée localement par Ollama.*",
                   parse_mode='Markdown'
               )
               return data.get('response')
        except Exception as ollama_err:
            logger.error(f"❌ Échec critique : Gemini ET Ollama sont HS. {ollama_err}")
            await context.bot.send_message(
                chat_id=AUTHORIZED_USER_ID,
                text="❌ Échec critique : Gemini ET Ollama sont HS.",
                parse_mode='Markdown'
            )
            return None   
    return None


# --- EXECUTEUR DE TÂCHES ---
class TaskExecutor:
    """Exécute les tâches de manière asynchrone"""

    def __init__(self, db: TaskDatabase):
        self.db = db
        self.active_workers = 0
        self.worker_lock = threading.Lock()
        self.running = False

    async def execute_task(self, task: Task, context: ContextTypes.DEFAULT_TYPE):
        """Exécute une tâche spécifique"""
        try:
            logger.info(f"[Task {task.id}] Exécution: {task.description}")

            task.status = TaskStatus.RUNNING
            self.db.update_task(task)

            # Exécuter selon le type
            if task.task_type == TaskType.COMMAND:
                result = await self._execute_command(task, context)
            elif task.task_type == TaskType.FILE_READ:
                result = await self._read_file(task)
            elif task.task_type == TaskType.FILE_EDIT:
                result = await self._edit_file(task)
            elif task.task_type == TaskType.AI_ANALYSIS:
                result = await self._ai_analysis(task, context)
            elif task.task_type == TaskType.PLAN:
                result = await self._create_plan(task, context)
            else:
                result = "Type de tâche inconnu"
                await context.bot.send_message(
                    chat_id=task.user_id,
                    text=f"**Tâche Unknown**\n{task.description}\n\n⚠️ ",
                    parse_mode='Markdown'
                )

            task.status = TaskStatus.COMPLETED
            task.result = result
            self.db.update_task(task)

            # Notifier l'utilisateur
            await context.bot.send_message(
                chat_id=task.user_id,
                text=f"✅ **Tâche #{task.id} terminée**\n{task.description}\n\n📄 Résultat:\n```\n{result[:3900]}\n```",
                parse_mode='Markdown'
            )

            logger.info(f"[Task {task.id}] Terminée avec succès")

        except Exception as e:
            logger.error(f"[Task {task.id}] Erreur: {e}")
            task.error_message = str(e)

            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.RETRYING
                task.scheduled_at = datetime.now() + timedelta(minutes=RETRY_DELAY_MINUTES)
                logger.info(f"[Task {task.id}] Retry prévu dans {RETRY_DELAY_MINUTES}min ({task.retry_count}/{task.max_retries})")
                await context.bot.send_message(
                    chat_id=task.user_id,
                    text=f"❌ **Tâche #{task.id} retry planned **\n{task.description}\n\n⚠️ Erreur: {str(e)}",
                    parse_mode='Markdown'
                )
            else:
                task.status = TaskStatus.FAILED
                logger.error(f"[Task {task.id}] Échec définitif après {task.max_retries} tentatives")

                # Notifier l'échec
                await context.bot.send_message(
                    chat_id=task.user_id,
                    text=f"❌ **Tâche #{task.id} échouée**\n{task.description}\n\n⚠️ Erreur: {str(e)}",
                    parse_mode='Markdown'
                )

            self.db.update_task(task)

    async def _execute_command(self, task: Task, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Exécute une commande système"""
        command = task.parameters.get('command', '')
        logger.info(f"Exécution commande: {command}")
        action_log.append(f"[{datetime.now().strftime('%H:%M')}] EXEC: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                out = result.stdout if result.stdout else "✅ Succès."
                return out

            # En cas d'erreur, demander une analyse AI
            err = result.stderr if result.stderr else result.stdout
            logger.warning(f"Échec commande: {err}")

            prompt = f"La commande `{command}` a échoué avec: `{err}`. Propose une correction."
            analysis = await safe_generate(prompt, context)

            return f"❌ ERREUR:\n{err}\n\n💡 ANALYSE:\n{analysis.text if analysis else 'Indisponible'}"

        except subprocess.TimeoutExpired:
            return "⏱️ Timeout: commande trop longue"
        except Exception as e:
            return f"⚠️ Erreur: {e}"

    async def _read_file(self, task: Task) -> str:
        """Lit un fichier"""
        path = task.parameters.get('path', '')
        try:
            with open(path, 'r') as f:
                content = f.read()
            return f"📖 Contenu de {path}:\n{content[:4000]}"
        except Exception as e:
            raise Exception(f"Impossible de lire {path}: {e}")

    async def _edit_file(self, task: Task) -> str:
        """Édite un fichier"""
        path = task.parameters.get('path', '')
        content = task.parameters.get('content', '')

        try:
            # Backup
            if os.path.exists(path):
                os.rename(path, f"{path}.bak")

            with open(path, 'w') as f:
                f.write(content)

            return f"✅ Fichier {path} mis à jour"
        except Exception as e:
            raise Exception(f"Impossible d'éditer {path}: {e}")

    async def _ai_analysis(self, task: Task, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Analyse avec AI"""
        prompt = task.parameters.get('prompt', '')
        response = await safe_generate(prompt, context)
        return response.text if response else "Analyse indisponible"

    async def _create_plan(self, task: Task, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Crée un plan avec sous-tâches"""
        topic = task.parameters.get('topic', task.description)

        # Demander à l'AI de créer le plan
        prompt = f"""Crée un plan d'action pour: {topic}

Retourne UNIQUEMENT un JSON avec cette structure:
{{
    "steps": [
        {{"description": "étape 1", "type": "command|file_read|ai_analysis"}},
        {{"description": "étape 2", "type": "command|file_read|ai_analysis"}}
    ]
}}"""

        response = await safe_generate(prompt, context)
        if not response:
            return "Impossible de créer le plan"

        try:
            # Parser le JSON
            plan_text = response.text.strip()
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0]

            plan = json.loads(plan_text)

            # Créer les sous-tâches
            for step in plan.get('steps', []):
                subtask = Task(
                    id=None,
                    user_id=task.user_id,
                    description=step['description'],
                    task_type=TaskType(step.get('type', 'ai_analysis')),
                    parameters={},
                    status=TaskStatus.PENDING,
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    retry_count=0,
                    max_retries=MAX_RETRIES,
                    parent_task_id=task.id,
                    error_message=None,
                    result=None,
                    scheduled_at=None
                )
                self.db.create_task(subtask)
                logger.info(f"Sous-tâche créée: {step['description']}")

            return f"✅ Plan créé avec {len(plan.get('steps', []))} étapes"

        except Exception as e:
            logger.error(f"Erreur parsing plan: {e}")
            return f"Erreur création plan: {e}"


# --- TELEGRAM HANDLERS ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère les messages Telegram (votre logique originale améliorée)"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        logger.warning(f"Accès non autorisé: {update.effective_user.id}")
        return

    user_text = update.message.text
    logger.info(f"Message reçu: {user_text}")
    temp_msg = await update.message.reply_text("🤔 *Analyse...*", parse_mode='Markdown')

    # Décision AI - On garde votre prompt original
    decision_prompt = f"""
Demande utilisateur: "{user_text}"

Réponds UNIQUEMENT avec UNE de ces actions au format exact:

Options d'actions:
COMMAND: [bash]
READ_FILE: [chemin]
EDIT_FILE: [chemin] | [contenu]
AI_ANALYSIS: [question]
DB_QUERY: [mot-clé à chercher dans la base] 
PLAN: [projet]
CHAT: [réponse]

Exemples:
- "lance un ping google" -> COMMAND: ping -c 3 google.com
- "lis le fichier X" -> READ_FILE: /path/to/file
- "crée un plan pour X" -> PLAN: déployer une app web
- "Quel est l'etat du job 1" DB_QUERY: 
"""

    try:
        response = await safe_generate(decision_prompt, context)
        if not response:
            await temp_msg.edit_text("❌ Quota épuisé")
            return

        res_text = response.strip()
        logger.info(f"Décision AI: {res_text[:100]}")

    except Exception as e:
        logger.error(f"Erreur AI: {e}")
        await temp_msg.edit_text(f"❌ Erreur: {e}")
        return

    # Créer la tâche selon la décision
    db = context.bot_data['db']
    executor = context.bot_data['executor']

    try:
       if res_text.startswith("COMMAND:"):
           logger.info("Start COMMAND")
           cmd = res_text.replace("COMMAND:", "").strip()
           task = Task(
               id=None,
               user_id=update.effective_user.id,
               description=user_text,
               task_type=TaskType.COMMAND,
               parameters={'command': cmd},
               status=TaskStatus.PENDING,
               created_at=datetime.now(),
               updated_at=datetime.now(),
               retry_count=0,
               max_retries=MAX_RETRIES,
               parent_task_id=None,
               error_message=None,
               result=None,
               scheduled_at=None
           )
           task.id = db.create_task(task)
           await temp_msg.edit_text(f"⚙️ Tâche #{task.id} créée: `{cmd}`", parse_mode='Markdown')
           # Exécution immédiate
           await executor.execute_task(task, context)

       elif res_text.startswith("READ_FILE:"):
           logger.info("Start READ_FILE")
           path = res_text.replace("READ_FILE:", "").strip()
           task = Task(
               id=None,
               user_id=update.effective_user.id,
               description=user_text,
               task_type=TaskType.FILE_READ,
               parameters={'path': path},
               status=TaskStatus.PENDING,
               created_at=datetime.now(),
               updated_at=datetime.now(),
               retry_count=0,
               max_retries=MAX_RETRIES,
               parent_task_id=None,
               error_message=None,
               result=None,
               scheduled_at=None
           )
           task.id = db.create_task(task)
           await temp_msg.edit_text(f"📖 Lecture de `{path}`...", parse_mode='Markdown')
           await executor.execute_task(task, context)

       elif res_text.startswith("EDIT_FILE:"):
           logger.info("Start EDIT_FILE")
           parts = res_text.replace("EDIT_FILE:", "").split("|", 1)
           if len(parts) == 2:
               path, content = parts[0].strip(), parts[1].strip()
               task = Task(
                   id=None,
                   user_id=update.effective_user.id,
                   description=user_text,
                   task_type=TaskType.FILE_EDIT,
                   parameters={'path': path, 'content': content},
                   status=TaskStatus.PENDING,
                   created_at=datetime.now(),
                   updated_at=datetime.now(),
                   retry_count=0,
                   max_retries=MAX_RETRIES,
                   parent_task_id=None,
                   error_message=None,
                   result=None,
                   scheduled_at=None
               )
               task.id = db.create_task(task)
               await temp_msg.edit_text(f"✏️ Édition de `{path}`...", parse_mode='Markdown')
               await executor.execute_task(task, context)

       elif res_text.startswith("PLAN:"):
           logger.info("Start PLAN")
           topic = res_text.replace("PLAN:", "").strip()
           task = Task(
               id=None,
               user_id=update.effective_user.id,
               description=user_text,
               task_type=TaskType.PLAN,
               parameters={'topic': topic},
               status=TaskStatus.PENDING,
               created_at=datetime.now(),
               updated_at=datetime.now(),
               retry_count=0,
               max_retries=MAX_RETRIES,
               parent_task_id=None,
               error_message=None,
               result=None,
               scheduled_at=None
           )
           task.id = db.create_task(task)
           await temp_msg.edit_text(f"📋 Création du plan...", parse_mode='Markdown')
           await executor.execute_task(task, context)

       elif res_text.startswith("AI_ANALYSIS:"):
           logger.info("Start AI_ANALYSIS")
           prompt = res_text.replace("AI_ANALYSIS:", "").strip()
           task = Task(
               id=None,
               user_id=update.effective_user.id,
               description=user_text,
               task_type=TaskType.AI_ANALYSIS,
               parameters={'prompt': prompt},
               status=TaskStatus.PENDING,
               created_at=datetime.now(),
               updated_at=datetime.now(),
               retry_count=0,
               max_retries=MAX_RETRIES,
               parent_task_id=None,
               error_message=None,
               result=None,
               scheduled_at=None
           )
           task.id = db.create_task(task)
           await temp_msg.edit_text(f"🤖 Analyse en cours...", parse_mode='Markdown')
           await executor.execute_task(task, context)

       elif res_text.startswith("DB_QUERY:"):
           logger.info("Start query DB")
           query = res_text.replace("DB_QUERY:", "").strip()
           tasks = db.search_tasks(query)
           
           if not tasks:
               await temp_msg.edit_text(f"🔍 Aucune information trouvée pour `{query}`")
               return

           response_text = f"🔍 **Infos trouvées pour '{query}':**\n\n"
           for t in tasks:
               # On affiche un résumé court de chaque tâche trouvée
               res_preview = (t.result[:100] + "...") if t.result else "Pas de résultat"
               response_text += f"📌 **#{t.id}** ({t.updated_at.strftime('%d/%m')}): {t.description}\n└ Result: `{res_preview}`\n\n"
           
           await temp_msg.edit_text(response_text, parse_mode='Markdown') 

       else:
           # CHAT - réponse directe sans créer de tâche
           await temp_msg.edit_text(res_text, parse_mode='Markdown')
    except telegram.error.BadRequest as e:
        if "Can't parse entities" in str(e):
          logger.warning("Formatage Markdown corrompu, envoi en texte brut.")
          # Deuxième tentative : Texte pur (sans parse_mode)
          try:
            await update.message.reply_text(res_text)
          except Exception as err:
            logger.error(f"❌ Échec total cmd status : {err}")
        else:
          # Autre type d'erreur BadRequest
          raise e


async def cmd_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Liste les tâches"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return

    db = context.bot_data['db']
    tasks = db.get_user_tasks(update.effective_user.id, limit=10)

    if not tasks:
        await update.message.reply_text("📋 Aucune tâche")
        return

    status_emoji = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.RUNNING: "🔄",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.RETRYING: "🔁",
        TaskStatus.CANCELLED: "🚫"
    }

    text = "📋 **Vos tâches:**\n\n"
    for t in tasks:
        emoji = status_emoji.get(t.status, "❓")
        text += f"{emoji} #{t.id} - {t.description[:50]}\n"
        text += f"   Type: {t.task_type.value} | Status: {t.status.value}\n"
        if t.retry_count > 0:
            text += f"   Retries: {t.retry_count}/{t.max_retries}\n"
        text += "\n"

    await update.message.reply_text(text, parse_mode='Markdown')


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche le statut d'une tâche"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return

    if not context.args:
        await update.message.reply_text("Usage: /status <task_id>")
        return

    try:
        task_id = int(context.args[0])
        db = context.bot_data['db']
        task = db.get_task(task_id)

        if not task or task.user_id != update.effective_user.id:
            await update.message.reply_text(f"❌ Tâche #{task_id} introuvable")
            return

        text = f"📊 **Tâche #{task.id}**\n\n"
        text += f"Description: {task.description}\n"
        text += f"Type: {task.task_type.value}\n"
        text += f"Status: {task.status.value}\n"
        text += f"Créée: {task.created_at.strftime('%Y-%m-%d %H:%M')}\n"

        if task.retry_count > 0:
            text += f"Tentatives: {task.retry_count}/{task.max_retries}\n"

        if task.error_message:
            text += f"\n⚠️ Erreur: {task.error_message}\n"

        if task.result:
            text += f"\n📄 Résultat:\n```\n{task.result[:3900]}\n```"

        await update.message.reply_text(text, parse_mode='Markdown')

    except ValueError:
        await update.message.reply_text("❌ ID invalide")
    except Exception as err:
        logger.error(f"❌ Échec total cmd status : {err}")


async def cmd_logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Affiche les derniers logs"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return

    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[-20:]
            data = "".join(lines)
        await update.message.reply_text(f"📜 **Logs:**\n```\n{data}\n```", parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur: {e}")


# --- JOBS DE FOND ---
async def process_pending_tasks(context: ContextTypes.DEFAULT_TYPE):
    """Traite les tâches en attente"""
    db = context.bot_data['db']
    executor = context.bot_data['executor']

    pending = db.get_pending_tasks()

    for task in pending:
        try:
            await executor.execute_task(task, context)
        except Exception as e:
            logger.error(f"Erreur traitement tâche {task.id}: {e}")


async def check_retrying_tasks(context: ContextTypes.DEFAULT_TYPE):
    """Vérifie les tâches en retry"""
    db = context.bot_data['db']
    retrying = db.get_tasks_by_status(TaskStatus.RETRYING)
    now = datetime.now()

    for task in retrying:
        if task.scheduled_at and task.scheduled_at <= now:
            task.status = TaskStatus.PENDING
            task.scheduled_at = None
            db.update_task(task)
            logger.info(f"Tâche {task.id} remise en pending pour retry")


async def hourly_report(context: ContextTypes.DEFAULT_TYPE):
    """Rapport horaire (votre code original)"""
    if not action_log:
        return

    logs_summary = "\n".join(action_log[-20:])
    try:
        if FORCED_AI_REPORT:
            res = safe_generate(contents=f"Résume ces actions en 3 lignes max: {logs_summary}" , context=context)
        else: 
            async with httpx.AsyncClient(timeout=120.0) as http_client:
               payload = {
                   "model": OLLAMA_MODEL,
                   "prompt": prompt,
                   "stream": False
               }
               res = await http_client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
               res.raise_for_status()
               data = res.json()
        await context.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text=f"⏰ **Rapport Horaire**\n{res.text}",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Erreur rapport: {e}")

    action_log.clear()


# --- VOTRE LOGIQUE (handle_message, safe_generate, etc.) ---

async def main():
    """Fonction principale asynchrone pour démarrer l'agent."""
    logger.info("🚀 Préparation de l'agent...")

    # Init DB et Executor
    db = TaskDatabase(DB_FILE)
    executor = TaskExecutor(db)
    # Configuration Telegram (votre code original)
    robust_request = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0
    )

    # Construction de l'application
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .request(robust_request)
        .build()
    )

    # Stocker DB et executor dans bot_data
    app.bot_data['db'] = db
    app.bot_data['executor'] = executor

    # Handlers
    app.add_handler(CommandHandler("ping", lambda u, c: u.message.reply_text("PONG 🏓")))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_error_handler(error_handler)

    # Configuration des tâches de fond
    if app.job_queue:
        app.job_queue.run_repeating(process_pending_tasks, interval=60, first=10)
        app.job_queue.run_repeating(check_retrying_tasks, interval=60, first=20)
        app.job_queue.run_repeating(hourly_report, interval=10800, first=60)

    # Démarrage propre
    logger.info("✅ Agent en ligne. En attente de messages...")
    try:
        logger.info("✅ Agent en ligne. En attente de messages...")
        # On initialise l'application
        await app.initialize()
        # On démarre l'app
        await app.start()

        # On démarre l'updater
        logger.info("✅ Agent Connecté. En attente de messages...")
        await app.updater.start_polling(drop_pending_updates=True)
        
        logger.info("✅ Agent Pulling. En attente de messages...")

        await app.bot.send_message(chat_id=AUTHORIZED_USER_ID, text="PONG 🏓 (Bot Online)")
        stop_event = asyncio.Event()
        await stop_event.wait()
            
    except telegram.error.Conflict:
        logger.error("💥 Conflit détecté ! Une autre instance tourne déjà. Fermeture dans 10s...")
        await asyncio.sleep(10)
        return # Sortie propre
    except Exception as e:
      logger.error(f"💥 Erreur fatale au démarrage main : {e}")

# --- MAIN ---
if __name__ == '__main__':
    try:
        # La méthode moderne pour lancer un script asynchrone
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("🛑 Agent arrêté proprement.")
    except Exception as e:
        logger.error(f"💥 Erreur fatale au démarrage __main__ : {e}")

