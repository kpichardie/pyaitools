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
KIMI_API_KEY = os.getenv('KIMI_API_KEY')
KIMI_BASE_URL = os.getenv('KIMI_BASE_URL', 'https://api.moonshot.cn/v1')
KIMI_MODEL = os.getenv('KIMI_MODEL', 'kimi-k2.5')
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

# Init AI clients
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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

# Agent version tracking
AGENT_VERSION = "2.1.0"

# Global state for pending self-update confirmations
# Structure: {user_id: {'task_id': int, 'analysis': dict, 'timestamp': datetime, 'expires_at': datetime}}
pending_confirmations = {}

# Startup verification state
startup_verification_pending = None  # Will store verification data after update


# --- STARTUP VERIFICATION FUNCTIONS ---
def get_current_git_commit() -> str:
    """Get the current git commit hash"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            cwd='/app',
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Failed to get git commit: {e}")
    return "unknown"


def get_last_stored_commit() -> str:
    """Get the last stored commit hash from file"""
    try:
        commit_file = '/app/.last_commit'
        if os.path.exists(commit_file):
            with open(commit_file, 'r') as f:
                return f.read().strip()
    except Exception as e:
        logger.warning(f"Failed to read last commit: {e}")
    return None


def store_current_commit(commit_hash: str):
    """Store the current commit hash to file"""
    try:
        commit_file = '/app/.last_commit'
        with open(commit_file, 'w') as f:
            f.write(commit_hash)
        logger.info(f"Stored current commit: {commit_hash[:8]}")
    except Exception as e:
        logger.error(f"Failed to store commit: {e}")


def get_commit_info(commit_hash: str) -> dict:
    """Get information about a commit"""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H|%an|%ae|%s|%ci', commit_hash],
            capture_output=True,
            text=True,
            cwd='/app',
            timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split('|', 4)
            if len(parts) >= 5:
                return {
                    'hash': parts[0],
                    'author': parts[1],
                    'email': parts[2],
                    'subject': parts[3],
                    'date': parts[4]
                }
    except Exception as e:
        logger.warning(f"Failed to get commit info: {e}")
    return None


def check_logs_since_update() -> dict:
    """Check logs for issues since the last update"""
    issues = {
        'errors': [],
        'warnings': [],
        'tracebacks': [],
        'total_errors': 0,
        'total_warnings': 0
    }
    
    try:
        if not os.path.exists(LOG_FILE):
            return issues
            
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        
        # Look at the last 500 lines for recent issues
        recent_lines = lines[-500:] if len(lines) > 500 else lines
        
        in_traceback = False
        current_traceback = []
        
        for line in recent_lines:
            line_lower = line.lower()
            
            # Detect errors
            if 'error' in line_lower or 'exception' in line_lower or 'critical' in line_lower:
                if 'traceback' in line_lower:
                    in_traceback = True
                    current_traceback = [line]
                else:
                    issues['errors'].append(line.strip())
                    issues['total_errors'] += 1
            
            # Detect warnings
            elif 'warning' in line_lower:
                issues['warnings'].append(line.strip())
                issues['total_warnings'] += 1
            
            # Capture traceback
            elif in_traceback:
                current_traceback.append(line)
                if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    # End of traceback
                    issues['tracebacks'].append(''.join(current_traceback))
                    current_traceback = []
                    in_traceback = False
        
        # Don't forget the last traceback if file ended during one
        if current_traceback:
            issues['tracebacks'].append(''.join(current_traceback))
            
    except Exception as e:
        logger.error(f"Error checking logs: {e}")
    
    return issues


async def analyze_startup_issues(context: ContextTypes.DEFAULT_TYPE, commit_info: dict, issues: dict) -> dict:
    """Query AI to analyze startup issues after an update"""
    
    # Prepare issue summary
    error_samples = issues['errors'][:10]
    warning_samples = issues['warnings'][:5]
    traceback_samples = issues['tracebacks'][:2]
    
    prompt = f"""Analyze the following startup issues after a self-update and provide recommendations.

Last Update Information:
- Commit: {commit_info['hash'][:8]}
- Author: {commit_info['author']}
- Date: {commit_info['date']}
- Subject: {commit_info['subject']}

Issues Detected Since Update:
- Total Errors: {issues['total_errors']}
- Total Warnings: {issues['total_warnings']}

Sample Errors:
{chr(10).join(error_samples) if error_samples else 'None'}

Sample Warnings:
{chr(10).join(warning_samples) if warning_samples else 'None'}

Sample Tracebacks:
{chr(10).join(traceback_samples) if traceback_samples else 'None'}

Analyze these issues and provide:
1. severity: "critical" | "high" | "medium" | "low"
2. summary: Brief description of the main problem
3. likely_cause: What probably caused these issues
4. recommendation: "revert" | "self_update" | "monitor"
5. confidence: 0-100 (how confident are you in this assessment)
6. explanation: Why you recommend this action

Return ONLY a valid JSON object with this exact structure:
{{
    "severity": "high",
    "summary": "Brief summary of the issue",
    "likely_cause": "Description of likely cause",
    "recommendation": "revert",
    "confidence": 85,
    "explanation": "Detailed explanation of the recommendation"
}}"""

    try:
        # Try Gemini first, then Kimi
        response_text = None
        
        if client and GEMINI_API_KEY:
            try:
                response = client.models.generate_content(model=MODEL_ID, contents=prompt)
                if response and response.text:
                    response_text = response.text
                    logger.info("Startup analysis: Using Gemini")
            except Exception as e:
                logger.warning(f"Gemini unavailable for startup analysis: {e}")
        
        if not response_text and KIMI_API_KEY:
            try:
                response_text = await query_kimi(prompt)
                if response_text:
                    logger.info("Startup analysis: Using Kimi K2.5")
            except Exception as e:
                logger.warning(f"Kimi unavailable for startup analysis: {e}")
        
        if not response_text:
            return None
        
        # Parse JSON
        text = response_text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return json.loads(text)
        
    except Exception as e:
        logger.error(f"Failed to analyze startup issues: {e}")
        return None


async def perform_startup_verification(context: ContextTypes.DEFAULT_TYPE):
    """Check if this is first start after update and verify for issues"""
    global startup_verification_pending
    
    try:
        current_commit = get_current_git_commit()
        last_commit = get_last_stored_commit()
        
        # If no previous commit stored, just store current and continue
        if not last_commit:
            logger.info("No previous commit stored. First run or commit tracking disabled.")
            store_current_commit(current_commit)
            return
        
        # If commits are the same, no update since last start
        if current_commit == last_commit:
            logger.info(f"No update detected. Running commit: {current_commit[:8]}")
            return
        
        # This is the first start after an update!
        logger.info(f"Update detected! Last: {last_commit[:8]} → Current: {current_commit[:8]}")
        
        # Get commit info
        commit_info = get_commit_info(current_commit)
        if not commit_info:
            commit_info = {
                'hash': current_commit,
                'author': 'Unknown',
                'email': '',
                'subject': 'Unknown',
                'date': 'Unknown'
            }
        
        # Check logs for issues
        await context.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text=f"🔄 **Startup Verification**\n\n"
                 f"Update detected!\n"
                 f"Previous: `{last_commit[:8]}`\n"
                 f"Current: `{current_commit[:8]}`\n\n"
                 f"Analyzing logs for issues...",
            parse_mode='Markdown'
        )
        
        issues = check_logs_since_update()
        
        # If no issues found, just notify and store
        if issues['total_errors'] == 0 and issues['total_warnings'] < 5:
            await context.bot.send_message(
                chat_id=AUTHORIZED_USER_ID,
                text=f"✅ **Startup Check Passed**\n\n"
                     f"Update: `{current_commit[:8]}`\n"
                     f"Subject: {commit_info['subject']}\n"
                     f"No significant issues detected.\n\n"
                     f"Agent is running normally.",
                parse_mode='Markdown'
            )
            store_current_commit(current_commit)
            return
        
        # Issues found - analyze with AI
        await context.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text=f"⚠️ **Issues Detected**\n\n"
                 f"Errors: {issues['total_errors']}\n"
                 f"Warnings: {issues['total_warnings']}\n\n"
                 f"Analyzing with AI...",
            parse_mode='Markdown'
        )
        
        analysis = await analyze_startup_issues(context, commit_info, issues)
        
        if not analysis:
            # AI analysis failed, provide basic info
            await context.bot.send_message(
                chat_id=AUTHORIZED_USER_ID,
                text=f"⚠️ **Update Verification Required**\n\n"
                     f"Commit: `{current_commit[:8]}`\n"
                     f"Subject: {commit_info['subject']}\n\n"
                     f"Issues detected:\n"
                     f"- {issues['total_errors']} errors\n"
                     f"- {issues['total_warnings']} warnings\n\n"
                     f"AI analysis unavailable.\n\n"
                     f"Options:\n"
                     f"/revert - Revert to previous commit\n"
                     f"/selfupdate - Trigger self-update to fix issues\n"
                     f"/ignore - Continue and mark as verified",
                parse_mode='Markdown'
            )
            
            # Store verification state
            startup_verification_pending = {
                'last_commit': last_commit,
                'current_commit': current_commit,
                'commit_info': commit_info,
                'issues': issues,
                'timestamp': datetime.now()
            }
            return
        
        # AI analysis successful
        severity_emoji = {
            'critical': '🔴',
            'high': '🟠',
            'medium': '🟡',
            'low': '🟢'
        }
        
        emoji = severity_emoji.get(analysis.get('severity', 'medium'), '🟡')
        recommendation = analysis.get('recommendation', 'monitor')
        confidence = analysis.get('confidence', 0)
        
        # Store verification state
        startup_verification_pending = {
            'last_commit': last_commit,
            'current_commit': current_commit,
            'commit_info': commit_info,
            'issues': issues,
            'analysis': analysis,
            'timestamp': datetime.now()
        }
        
        # Build recommendation message
        if recommendation == 'revert':
            action_text = f"{emoji} **CRITICAL: REVERT RECOMMENDED**\n\n"
            options_text = "Options:\n/revert - Revert to previous commit\n/ignore - Continue anyway (not recommended)"
        elif recommendation == 'self_update':
            action_text = f"{emoji} **Issues Detected: Self-Update Recommended**\n\n"
            options_text = "Options:\n/selfupdate - Trigger self-update to fix\n/ignore - Continue and monitor"
        else:
            action_text = f"{emoji} **Minor Issues: Monitor Recommended**\n\n"
            options_text = "Options:\n/ignore - Mark as verified and continue\n/selfupdate - Run self-update anyway"
        
        message = (
            f"{action_text}"
            f"Commit: `{current_commit[:8]}`\n"
            f"Subject: {commit_info['subject']}\n"
            f"Severity: {analysis.get('severity', 'unknown').upper()}\n"
            f"Confidence: {confidence}%\n\n"
            f"**Summary:**\n{analysis.get('summary', 'No summary')}\n\n"
            f"**Likely Cause:**\n{analysis.get('likely_cause', 'Unknown')}\n\n"
            f"**Explanation:**\n{analysis.get('explanation', 'No explanation')}\n\n"
            f"**Issues:** {issues['total_errors']} errors, {issues['total_warnings']} warnings\n\n"
            f"{options_text}"
        )
        
        await context.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text=message,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"Error during startup verification: {e}")
        # Don't block startup, just log the error
        store_current_commit(get_current_git_commit())


async def cmd_revert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Revert to the previous commit after a failed update"""
    global startup_verification_pending
    
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return
    
    if not startup_verification_pending:
        await update.message.reply_text(
            "❌ No pending verification found. Cannot revert.",
            parse_mode='Markdown'
        )
        return
    
    last_commit = startup_verification_pending['last_commit']
    current_commit = startup_verification_pending['current_commit']
    
    try:
        await update.message.reply_text(
            f"🔄 **Reverting Update**\n\n"
            f"From: `{current_commit[:8]}`\n"
            f"To: `{last_commit[:8]}`\n\n"
            f"Creating backup and reverting...",
            parse_mode='Markdown'
        )
        
        # Create backup of current state
        backup_dir = f"/app/backups/revert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(backup_dir, exist_ok=True)
        
        subprocess.run(
            ['cp', '/app/autonomous_orchestrator.py', f"{backup_dir}/autonomous_orchestrator.py"],
            check=True
        )
        
        # Revert to last commit
        result = subprocess.run(
            ['git', 'reset', '--hard', last_commit],
            capture_output=True,
            text=True,
            cwd='/app',
            timeout=30
        )
        
        if result.returncode == 0:
            await update.message.reply_text(
                f"✅ **Revert Successful**\n\n"
                f"Reverted to: `{last_commit[:8]}`\n\n"
                f"🔄 Restarting agent in 5 seconds...",
                parse_mode='Markdown'
            )
            
            # Clear verification state
            startup_verification_pending = None
            
            # Store the reverted commit
            store_current_commit(last_commit)
            
            # Restart
            await asyncio.sleep(5)
            # Get executor from context and restart
            executor = context.bot_data.get('executor')
            if executor:
                await executor._restart_container()
            else:
                # Fallback: just exit
                sys.exit(0)
        else:
            await update.message.reply_text(
                f"❌ **Revert Failed**\n\n"
                f"Error: `{result.stderr}`",
                parse_mode='Markdown'
            )
            
    except Exception as e:
        logger.error(f"Revert failed: {e}")
        await update.message.reply_text(
            f"❌ **Revert Failed**\n\n"
            f"Error: `{str(e)}`",
            parse_mode='Markdown'
        )


async def cmd_ignore_verification(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ignore startup verification issues and continue"""
    global startup_verification_pending
    
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return
    
    if not startup_verification_pending:
        await update.message.reply_text(
            "ℹ️ No pending verification to ignore.",
            parse_mode='Markdown'
        )
        return
    
    current_commit = startup_verification_pending['current_commit']
    
    # Store the commit to mark as verified
    store_current_commit(current_commit)
    startup_verification_pending = None
    
    await update.message.reply_text(
        f"✅ **Verification Ignored**\n\n"
        f"Current commit `{current_commit[:8]}` marked as verified.\n"
        f"Agent will continue running normally.",
        parse_mode='Markdown'
    )

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
    SELF_UPDATE = "self_update"


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
async def query_kimi(prompt: str) -> Optional[str]:
    """Query Kimi K2.5 API"""
    if not KIMI_API_KEY:
        return None
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            headers = {
                "Authorization": f"Bearer {KIMI_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": KIMI_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            res = await http_client.post(
                f"{KIMI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload
            )
            res.raise_for_status()
            data = res.json()
            return data.get('choices', [{}])[0].get('message', {}).get('content')
    except Exception as e:
        logger.warning(f"⚠️ Kimi API error: {e}")
        return None


async def safe_generate(prompt: str, context: ContextTypes.DEFAULT_TYPE) -> Optional[Any]:
    """Génère du contenu avec gestion du quota 429 - Fallback: Gemini → Kimi → Ollama"""
    
    # 1. Try Gemini (Primary)
    if client and GEMINI_API_KEY:
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            if response and response.text:
                return response.text
        except Exception as e:
            logger.warning(f"⚠️ Gemini indisponible ({e}). Bascule sur Kimi K2.5...")
    
    # 2. Fallback to Kimi K2.5
    if KIMI_API_KEY:
        try:
            kimi_response = await query_kimi(prompt)
            if kimi_response:
                await context.bot.send_message(
                    chat_id=AUTHORIZED_USER_ID,
                    text="🌙 *Note : Gemini est hors ligne, réponse générée par Kimi K2.5.*",
                    parse_mode='Markdown'
                )
                return kimi_response
        except Exception as e:
            logger.warning(f"⚠️ Kimi indisponible ({e}). Bascule sur Ollama local...")
    
    # 3. Fallback to Ollama (Local)
    try:
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
                text="🏠 *Note : Services cloud hors ligne, réponse générée localement par Ollama.*",
                parse_mode='Markdown'
            )
            return data.get('response')
    except Exception as ollama_err:
        logger.error(f"❌ Échec critique : Tous les services AI sont HS. {ollama_err}")
        await context.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text="❌ Échec critique : Gemini, Kimi ET Ollama sont HS.",
            parse_mode='Markdown'
        )
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
            elif task.task_type == TaskType.SELF_UPDATE:
                result = await self._execute_self_update(task, context)
                # Self-update returns early and handles its own completion
                return
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

    async def _execute_self_update(self, task: Task, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Execute self-improvement workflow with user confirmation"""
        force_mode = task.parameters.get('force', False)
        user_id = task.user_id
        
        try:
            # Step 1: Collect data
            await self._notify_progress(task, context, "📊 Collecting system data...")
            code = await self._read_own_code()
            logs = await self._read_recent_logs(lines=100)
            metrics = await self._get_task_metrics()
            
            # Step 2: Query AI for analysis (Gemini → Kimi, NEVER use Ollama for self-update)
            await self._notify_progress(task, context, "🤖 Analyzing with AI...")
            analysis = await self._query_self_analysis(code, logs, metrics)
            
            if not analysis:
                return "❌ Failed to get analysis from AI services (Gemini/Kimi)"
            
            # Step 3: Parse metrics
            confidence = analysis.get('code_confidence', 0)
            performance = analysis.get('performance', 0)
            resilience = analysis.get('resilience', 0)
            pertinence = analysis.get('pertinence', 0)
            
            # Step 4: Display results
            metrics_msg = f"""📊 Self-Analysis Results:
• Code Confidence: {confidence}%
• Performance: {performance}%
• Resilience: {resilience}%
• Pertinence: {pertinence}%

Proposed changes: {len(analysis.get('changes', []))} files

Summary: {analysis.get('summary', 'No summary provided')}"""
            
            await context.bot.send_message(
                chat_id=user_id,
                text=metrics_msg,
                parse_mode='Markdown'
            )
            
            # Step 5: Decision
            all_good = all(m >= 50 for m in [confidence, performance, resilience, pertinence])
            
            if force_mode:
                logger.warning(f"FORCE MODE: User {user_id} triggered force self-update")
                await context.bot.send_message(
                    chat_id=user_id,
                    text="⚠️ FORCE MODE: Applying changes without confirmation...",
                    parse_mode='Markdown'
                )
                result = await self._apply_self_update_changes(analysis, task, context)
                return f"Changes applied (force mode): {result}"
            
            if all_good:
                # Good metrics - ask for confirmation
                await context.bot.send_message(
                    chat_id=user_id,
                    text="✅ All metrics ≥ 50%. Apply changes?\nReply: YES or NO\nOr SHOWDIFF to see detailed changes",
                    parse_mode='Markdown'
                )
            else:
                # Low metrics - require explicit approval
                await context.bot.send_message(
                    chat_id=user_id,
                    text=f"⚠️ Low metrics detected (some < 50%). Review recommended.\nReply SHOWDIFF to see changes or YES to apply anyway.\nOr NO to cancel.",
                    parse_mode='Markdown'
                )
            
            # Store confirmation state
            pending_confirmations[user_id] = {
                'task_id': task.id,
                'analysis': analysis,
                'timestamp': datetime.now(),
                'expires_at': datetime.now() + timedelta(minutes=30)
            }
            
            return "Awaiting user confirmation (30 min timeout)"
            
        except Exception as e:
            logger.error(f"Self-update execution error: {e}")
            return f"❌ Self-update failed: {e}"
    
    async def _notify_progress(self, task: Task, context: ContextTypes.DEFAULT_TYPE, message: str):
        """Send progress notification"""
        await context.bot.send_message(
            chat_id=task.user_id,
            text=f"🔄 [Task #{task.id}] {message}",
            parse_mode='Markdown'
        )
    
    async def _read_own_code(self) -> str:
        """Read the agent's own source code"""
        try:
            with open('/app/autonomous_orchestrator.py', 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read own code: {e}")
            return ""
    
    async def _read_recent_logs(self, lines: int = 100) -> str:
        """Read recent log entries"""
        try:
            with open(LOG_FILE, 'r') as f:
                log_lines = f.readlines()
                return "".join(log_lines[-lines:])
        except Exception as e:
            logger.error(f"Failed to read logs: {e}")
            return ""
    
    async def _get_task_metrics(self) -> Dict[str, Any]:
        """Get task execution metrics from database"""
        try:
            completed = len(self.db.get_tasks_by_status(TaskStatus.COMPLETED))
            failed = len(self.db.get_tasks_by_status(TaskStatus.FAILED))
            pending = len(self.db.get_tasks_by_status(TaskStatus.PENDING))
            retrying = len(self.db.get_tasks_by_status(TaskStatus.RETRYING))
            
            total = completed + failed + pending + retrying
            success_rate = (completed / total * 100) if total > 0 else 0
            
            return {
                'total_tasks': total,
                'completed': completed,
                'failed': failed,
                'pending': pending,
                'retrying': retrying,
                'success_rate': round(success_rate, 2),
                'agent_version': AGENT_VERSION
            }
        except Exception as e:
            logger.error(f"Failed to get task metrics: {e}")
            return {'error': str(e)}
    
    async def _query_self_analysis(self, code: str, logs: str, metrics: Dict) -> Optional[Dict]:
        """Query AI for self-analysis - Gemini primary, Kimi fallback (NEVER use Ollama for this)"""
        prompt = f"""Analyze the following autonomous agent code and provide improvement suggestions.

Current Code:
```python
{code[:8000]}
```

Recent Logs (last 100 lines):
```
{logs}
```

Task Metrics:
- Total Tasks: {metrics.get('total_tasks', 0)}
- Success Rate: {metrics.get('success_rate', 0)}%
- Failed: {metrics.get('failed', 0)}
- Current Version: {metrics.get('agent_version', 'unknown')}

Evaluate and provide:
1. code_confidence: 0-100 (how confident are you in the code quality)
2. performance: 0-100 (how well does it perform)
3. resilience: 0-100 (how resilient is the error handling)
4. pertinence: 0-100 (how pertinent are the features)
5. summary: Brief description of the main improvement
6. changes: List of specific changes to make, each with:
   - file: path to file
   - description: what to change
   - new_content: complete new content for the file

Return ONLY a valid JSON object with this exact structure:
{{
    "code_confidence": 75,
    "performance": 80,
    "resilience": 65,
    "pertinence": 70,
    "summary": "Brief summary of improvements",
    "changes": [
        {{
            "file": "/app/autonomous_orchestrator.py",
            "description": "Description of change",
            "new_content": "complete file content here"
        }}
    ]
}}"""
        
        response_text = None
        
        # Try Gemini first
        if client and GEMINI_API_KEY:
            try:
                response = client.models.generate_content(model=MODEL_ID, contents=prompt)
                if response and response.text:
                    response_text = response.text
                    logger.info("Self-analysis: Using Gemini")
            except Exception as e:
                logger.warning(f"⚠️ Gemini unavailable for self-analysis ({e})")
        
        # Fallback to Kimi if Gemini failed
        if not response_text and KIMI_API_KEY:
            try:
                response_text = await query_kimi(prompt)
                if response_text:
                    logger.info("Self-analysis: Using Kimi K2.5")
            except Exception as e:
                logger.warning(f"⚠️ Kimi unavailable for self-analysis ({e})")
        
        if not response_text:
            logger.error("❌ Failed to get self-analysis from Gemini or Kimi")
            return None
        
        # Parse JSON from response
        try:
            text = response_text.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text)
        except Exception as e:
            logger.error(f"Failed to parse self-analysis response: {e}")
            return None
    
    async def _apply_self_update_changes(self, analysis: Dict, task: Task, context: ContextTypes.DEFAULT_TYPE) -> str:
        """Apply the self-update changes"""
        changes = analysis.get('changes', [])
        backup_dir = None
        
        if not changes:
            return "No changes to apply"
        
        try:
            # Step 1: Create backup
            backup_dir = f"/app/backups/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir, exist_ok=True)
            
            await self._notify_progress(task, context, "💾 Creating backup...")
            subprocess.run(['cp', '/app/autonomous_orchestrator.py', f"{backup_dir}/autonomous_orchestrator.py"], check=True)
            
            # Step 2: Apply changes
            await self._notify_progress(task, context, "✏️ Applying changes...")
            for change in changes:
                file_path = change.get('file')
                new_content = change.get('new_content')
                
                if file_path and new_content:
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    logger.info(f"Updated file: {file_path}")
            
            # Step 3: Git commit
            await self._notify_progress(task, context, "📦 Creating git commit...")
            
            # Check git status
            status_result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, cwd='/app')
            
            if status_result.stdout.strip():
                # There are changes to commit
                confidence = analysis.get('code_confidence', 0)
                performance = analysis.get('performance', 0)
                resilience = analysis.get('resilience', 0)
                pertinence = analysis.get('pertinence', 0)
                summary = analysis.get('summary', 'Self-update')
                
                subprocess.run(['git', 'add', '.'], check=True, cwd='/app')
                commit_msg = f"Self-update: {summary}\n\nMetrics: C:{confidence}% P:{performance}% R:{resilience}% Pe:{pertinence}%\nVersion: {AGENT_VERSION}"
                subprocess.run(['git', 'commit', '-m', commit_msg], check=True, cwd='/app')
                
                # Try to push if remote exists
                try:
                    subprocess.run(['git', 'push'], check=True, cwd='/app', capture_output=True)
                except:
                    logger.warning("Failed to push to remote, but commit was successful")
            
            # Step 4: Create changelog entry
            await self._notify_progress(task, context, "📝 Updating changelog...")
            self._update_changelog(analysis)
            
            # Step 5: Notify restart
            await context.bot.send_message(
                chat_id=task.user_id,
                text="✅ Changes applied and committed.\n🔄 Restarting agent in 5 seconds...",
                parse_mode='Markdown'
            )
            
            # Step 6: Schedule restart
            await asyncio.sleep(5)
            await self._restart_container()
            
            return "Changes applied successfully. Agent restarting..."
            
        except Exception as e:
            logger.error(f"Failed to apply self-update changes: {e}")
            # Restore from backup
            if backup_dir:
                try:
                    subprocess.run(['cp', f"{backup_dir}/autonomous_orchestrator.py", '/app/autonomous_orchestrator.py'], check=True)
                    logger.info("Restored from backup after failed update")
                except:
                    pass
            raise e
    
    def _update_changelog(self, analysis: Dict):
        """Update changelog with self-update information"""
        try:
            changelog_path = '/app/CHANGELOG.md'
            confidence = analysis.get('code_confidence', 0)
            performance = analysis.get('performance', 0)
            resilience = analysis.get('resilience', 0)
            pertinence = analysis.get('pertinence', 0)
            summary = analysis.get('summary', 'Self-update')
            
            entry = f"""## {AGENT_VERSION} ({datetime.now().strftime('%Y-%m-%d')})
- Self-improvement update
- Metrics: Confidence:{confidence}%, Performance:{performance}%, Resilience:{resilience}%, Pertinence:{pertinence}%
- Changes: {summary}
- Initiated by: Agent self-update system

"""
            
            if os.path.exists(changelog_path):
                with open(changelog_path, 'r') as f:
                    existing = f.read()
                with open(changelog_path, 'w') as f:
                    f.write(entry + existing)
            else:
                with open(changelog_path, 'w') as f:
                    f.write("# Changelog\n\n" + entry)
                    
        except Exception as e:
            logger.error(f"Failed to update changelog: {e}")
    
    async def _restart_container(self):
        """Trigger container restart"""
        try:
            # Method 1: Check if running in Docker and restart via docker command
            if os.path.exists('/.dockerenv'):
                # Get container ID
                with open('/proc/self/cgroup', 'r') as f:
                    content = f.read()
                    container_id = None
                    for line in content.split('\n'):
                        if 'docker' in line:
                            parts = line.split('/')
                            if len(parts) > 2:
                                container_id = parts[-1][:12]
                                break
                
                if container_id:
                    logger.info(f"Restarting container: {container_id}")
                    # This requires docker socket access
                    subprocess.run(['docker', 'restart', container_id], check=False)
                else:
                    # Fallback: exit and let orchestrator restart
                    logger.info("Container ID not found, exiting for restart")
                    sys.exit(0)
            else:
                # Not in Docker, just exit
                logger.info("Not in Docker container, exiting for restart")
                sys.exit(0)
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            # Fallback: exit anyway
            sys.exit(0)


# --- TELEGRAM HANDLERS ---
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gère les messages Telegram (votre logique originale améliorée)"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        logger.warning(f"Accès non autorisé: {update.effective_user.id}")
        return

    # Check if this is a confirmation response first
    if await handle_confirmation_response(update, context):
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


async def cmd_selfupdate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger manual self-improvement analysis via Telegram"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return
    
    args = context.args if context.args else []
    force_mode = 'force' in args
    
    # Check if there's already a pending confirmation for this user
    user_id = update.effective_user.id
    if user_id in pending_confirmations:
        await update.message.reply_text(
            "⏳ You already have a pending self-update confirmation.\n"
            "Reply YES, NO, or SHOWDIFF to the previous request first.",
            parse_mode='Markdown'
        )
        return
    
    # Start self-update task
    db = context.bot_data['db']
    executor = context.bot_data['executor']
    
    task = Task(
        id=None,
        user_id=user_id,
        description="Manual self-update triggered" + (" (FORCE)" if force_mode else ""),
        task_type=TaskType.SELF_UPDATE,
        parameters={'force': force_mode, 'initiated_by': 'telegram'},
        status=TaskStatus.PENDING,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        retry_count=0,
        max_retries=1,  # Self-update only retries once
        parent_task_id=None,
        error_message=None,
        result=None,
        scheduled_at=None
    )
    
    task.id = db.create_task(task)
    
    await update.message.reply_text(
        f"🔍 Self-update task #{task.id} created.{' Force mode enabled.' if force_mode else ''}\n"
        f"Starting analysis... This may take a minute.",
        parse_mode='Markdown'
    )
    
    # Execute immediately
    await executor.execute_task(task, context)


async def handle_confirmation_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Handle user responses to self-update confirmations. Returns True if handled."""
    user_id = update.effective_user.id
    
    # Check if user is authorized and has pending confirmation
    if str(user_id) != str(AUTHORIZED_USER_ID):
        return False
    
    if user_id not in pending_confirmations:
        return False
    
    user_text = update.message.text.upper().strip()
    confirmation = pending_confirmations[user_id]
    
    # Check if confirmation has expired
    if datetime.now() > confirmation['expires_at']:
        del pending_confirmations[user_id]
        await update.message.reply_text(
            "⌛ Confirmation expired. Run /selfupdate again if you want to retry.",
            parse_mode='Markdown'
        )
        return True
    
    db = context.bot_data['db']
    executor = context.bot_data['executor']
    task = db.get_task(confirmation['task_id'])
    
    if not task:
        del pending_confirmations[user_id]
        await update.message.reply_text("❌ Task not found. Please try again.")
        return True
    
    if user_text == 'YES':
        # Apply changes
        await update.message.reply_text("✅ Applying changes... This will take a moment.")
        
        try:
            analysis = confirmation['analysis']
            result = await executor._apply_self_update_changes(analysis, task, context)
            del pending_confirmations[user_id]
        except Exception as e:
            logger.error(f"Failed to apply self-update: {e}")
            await update.message.reply_text(f"❌ Failed to apply changes: {e}")
            del pending_confirmations[user_id]
        
        return True
        
    elif user_text == 'NO':
        await update.message.reply_text("❌ Self-update cancelled. No changes were made.")
        del pending_confirmations[user_id]
        return True
        
    elif user_text == 'SHOWDIFF':
        # Show detailed diff
        analysis = confirmation['analysis']
        changes = analysis.get('changes', [])
        
        if not changes:
            await update.message.reply_text("No changes to display.")
            return True
        
        diff_text = "📋 Proposed changes:\n\n"
        for i, change in enumerate(changes[:5], 1):  # Show max 5 changes
            file_path = change.get('file', 'unknown')
            description = change.get('description', 'No description')
            diff_text += f"{i}. **{file_path}**\n   {description}\n\n"
        
        if len(changes) > 5:
            diff_text += f"... and {len(changes) - 5} more changes\n"
        
        diff_text += "\nReply YES to apply or NO to cancel."
        
        await update.message.reply_text(diff_text, parse_mode='Markdown')
        return True
    
    # Not a confirmation command, let it pass through
    return False


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
            res = await safe_generate(f"Résume ces actions en 3 lignes max: {logs_summary}", context)
            report_text = res if res else "No report available"
        else: 
            async with httpx.AsyncClient(timeout=120.0) as http_client:
               payload = {
                   "model": OLLAMA_MODEL,
                   "prompt": f"Résume ces actions en 3 lignes max: {logs_summary}",
                   "stream": False
               }
               res = await http_client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
               res.raise_for_status()
               data = res.json()
               report_text = data.get('response', 'No report available')
        await context.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text=f"⏰ **Rapport Horaire**\n{report_text}",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Erreur rapport: {e}")

    action_log.clear()


async def cleanup_expired_confirmations(context: ContextTypes.DEFAULT_TYPE):
    """Clean up expired self-update confirmations"""
    global pending_confirmations
    now = datetime.now()
    expired_users = []
    
    for user_id, confirmation in pending_confirmations.items():
        if now > confirmation['expires_at']:
            expired_users.append(user_id)
    
    for user_id in expired_users:
        del pending_confirmations[user_id]
        try:
            await context.bot.send_message(
                chat_id=user_id,
                text="⌛ Self-update confirmation expired. Run /selfupdate again if you want to retry.",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Failed to notify user {user_id} of expired confirmation: {e}")
    
    if expired_users:
        logger.info(f"Cleaned up {len(expired_users)} expired confirmations")


async def daily_self_analysis(context: ContextTypes.DEFAULT_TYPE):
    """Trigger automatic daily self-improvement analysis"""
    try:
        logger.info("Starting daily self-analysis...")
        
        db = context.bot_data['db']
        executor = context.bot_data['executor']
        
        # Check if there's already a pending confirmation
        if pending_confirmations:
            logger.info("Skipping daily self-analysis: pending confirmation exists")
            return
        
        # Create self-update task
        if not AUTHORIZED_USER_ID:
            logger.error("AUTHORIZED_USER_ID not set, skipping daily self-analysis")
            return
            
        task = Task(
            id=None,
            user_id=int(AUTHORIZED_USER_ID),
            description="Daily automatic self-analysis",
            task_type=TaskType.SELF_UPDATE,
            parameters={'force': False, 'initiated_by': 'daily_job'},
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            retry_count=0,
            max_retries=1,
            parent_task_id=None,
            error_message=None,
            result=None,
            scheduled_at=None
        )
        
        task.id = db.create_task(task)
        logger.info(f"Created daily self-update task #{task.id}")
        
        # Execute the task
        await executor.execute_task(task, context)
        
    except Exception as e:
        logger.error(f"Error in daily self-analysis: {e}")


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
    app.add_handler(CommandHandler("selfupdate", cmd_selfupdate))
    app.add_handler(CommandHandler("list", cmd_list))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("logs", cmd_logs))
    app.add_handler(CommandHandler("revert", cmd_revert))
    app.add_handler(CommandHandler("ignore", cmd_ignore_verification))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_error_handler(error_handler)

    # Jobs
    if app.job_queue:
        app.job_queue.run_repeating(process_pending_tasks, interval=60, first=10)
        app.job_queue.run_repeating(check_retrying_tasks, interval=60, first=20)
        app.job_queue.run_repeating(cleanup_expired_confirmations, interval=60, first=30)
        app.job_queue.run_repeating(hourly_report, interval=10800, first=60)
        app.job_queue.run_repeating(daily_self_analysis, interval=86400, first=3600)  # Daily at 1 hour after start

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
        
        # Perform startup verification after update
        # Create a minimal context for startup verification
        from telegram.ext import CallbackContext
        startup_context = CallbackContext(app)
        startup_context._bot = app.bot
        startup_context._chat_id = int(AUTHORIZED_USER_ID) if AUTHORIZED_USER_ID else None
        startup_context.bot_data = app.bot_data
        await perform_startup_verification(startup_context)
        
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

