# Autonomous AI Agent - Project Context

## Overview

This project is an autonomous AI agent designed to automate tasks, communicate via Telegram, and continuously self-improve its capabilities. The agent operates within a containerized environment with persistent SQLite storage for task management.

## Core Architecture

### File Structure
```
autonomous/
├── autonomous.py               # Basic version (deprecated)
├── autonomous_orchestrator.py  # Main orchestrator (v2)
└── project-context.md          # This documentation
```

### Technology Stack
- **Language**: Python 3.11+
- **AI Models**: 
  - Primary: Google Gemini (Gemini 2.0 Flash)
  - Fallback: Local Ollama (Llama3 via HTTP API)
- **Database**: SQLite (persistent task storage)
- **Communication**: Telegram Bot API
- **Concurrency**: Asyncio with threading locks for DB operations

### Environment Variables

#### Required
```bash
TELEGRAM_TOKEN=<bot_token>
GEMINI_API_KEY=<google_api_key>
AUTHORIZED_USER_ID=<telegram_user_id>
```

#### Optional (with defaults)
```bash
DB_FILE=/app/tasks.db                    # SQLite database path
LOG_FILE=/app/autonomous.log             # Log file path
MODEL_ID=gemini-2.0-flash-exp            # Gemini model ID
MAX_WORKERS=3                            # Concurrent task limit
RETRY_DELAY_MINUTES=5                    # Retry interval
MAX_RETRIES=3                            # Max retry attempts per task
OLLAMA_HOST=http://localhost:11434       # Ollama API endpoint
OLLAMA_MODEL=llama3                      # Local model name
FORCED_AI_REPORT=false                   # Force Gemini for reports
```

## Task System

### Task Lifecycle
```
PENDING → RUNNING → [COMPLETED | FAILED | RETRYING]
                           ↓
                    (retry_count < max_retries)
                           ↓
                         RETRYING → PENDING (after delay)
```

### Task Types
1. **COMMAND**: Execute bash commands
2. **FILE_READ**: Read file contents
3. **FILE_EDIT**: Edit files with backup creation
4. **AI_ANALYSIS**: Perform AI-powered analysis
5. **PLAN**: Create multi-step plans with subtasks
6. **SELF_UPDATE**: Trigger self-improvement analysis and code update

### Database Schema (tasks table)
```sql
- id (PRIMARY KEY, AUTOINCREMENT)
- user_id (INTEGER, NOT NULL)
- description (TEXT, NOT NULL)
- task_type (TEXT, NOT NULL)
- parameters (JSON)
- status (TEXT, NOT NULL)
- created_at, updated_at (TIMESTAMP)
- retry_count, max_retries (INTEGER)
- parent_task_id (FK to tasks.id)
- error_message, result (TEXT)
- scheduled_at (TIMESTAMP)
```

## Communication Protocol

### Telegram Commands
- **/list**: Display last 10 tasks with status
- **/status <task_id>**: Show detailed task information
- **/logs**: Display last 20 log lines
- **/ping**: Health check (returns PONG)
- **/selfupdate [force]**: Trigger manual self-improvement analysis
  - Without args: Runs analysis and asks for confirmation if metrics < 50%
  - With `force`: Applies changes immediately (use with caution)
- **Text messages**: Interpreted by AI to create and execute tasks

### Message Types
The AI interprets user messages and responds with structured actions:
- `COMMAND: <bash command>`
- `READ_FILE: <path>`
- `EDIT_FILE: <path> | <content>`
- `AI_ANALYSIS: <question>`
- `DB_QUERY: <search_term>`
- `PLAN: <project_description>`
- `CHAT: <response>` (direct conversation)

### Notifications
The agent sends Telegram notifications for:
- Task creation and completion
- Errors and failures
- Retry scheduling
- Hourly reports (every 3 hours)
- Self-improvement proposals

## Self-Improvement System (Planned)

### Trigger Methods

#### 1. Automatic (Daily)
Runs every 24 hours via background job.

#### 2. Manual via Telegram (/selfupdate command)
User can trigger self-analysis on-demand:

**Command Options:**
- `/selfupdate` - Run analysis, request confirmation if metrics < 50%
- `/selfupdate force` - Skip confirmation, apply changes immediately

**Workflow:**
```
User sends: /selfupdate
↓
Agent acknowledges: "🔍 Starting self-analysis..."
↓
Collect data (code, logs, tasks)
↓
Query Gemini for evaluation
↓
Display metrics in Telegram:
  📊 Self-Analysis Results:
  • Code Confidence: 75%
  • Performance: 80%
  • Resilience: 65%
  • Pertinence: 70%
↓
If ALL metrics ≥ 50%:
  → Show proposed changes
  → Ask: "Apply changes? (yes/no/diff)"
  → If yes: Apply → Git commit → Restart
  
If ANY metric < 50%:
  → Ask: "Low confidence detected. Review changes?"
  → Show detailed diff
  → Wait for user approval
```

### Daily Analysis Workflow
1. **Data Collection** (Daily)
   - Code analysis: Read autonomous_orchestrator.py
   - Log analysis: Parse recent LOG_FILE entries
   - Task analysis: Query completed/failed tasks from SQLite

2. **AI Evaluation** (Using Gemini only - no local models)
   The agent queries Gemini with structured context:
   ```
   Context:
   - Current code: <full_source_code>
   - Recent logs: <last_100_lines>
   - Task metrics: <task_statistics>
   
   Evaluate and provide:
   - Code confidence: 0-100%
   - Performance metric: 0-100%
   - Resilience metric: 0-100%
   - Pertinence metric: 0-100%
   - Proposed changes: <diff_patch>
   ```

3. **Decision Logic**
   
   **Automatic Mode (Daily):**
   - If **ALL metrics ≥ 50%**: Proceed with changes automatically
   - If **ANY metric < 50%**: Request user advice via Telegram
   
   **Manual Mode (/selfupdate):**
   - Always request user confirmation before applying
   - Show detailed metrics and proposed diff
   - User options: `apply`, `cancel`, `showdiff`, `modify`
   
   **Force Mode (/selfupdate force):**
   - Skip all confirmations
   - Apply changes regardless of metrics
   - Use only for emergency fixes
   - Log warning: "Force update triggered by user"

4. **Change Application** (After approval)
   - Apply patch to codebase using `file_edit` operations
   - Create git commit: 
     ```bash
     git add autonomous_orchestrator.py
     git commit -m "Self-update: <description> - Metrics: C:75% P:80% R:65% Pe:70%"
     git push origin main
     ```
   - Create backup of current code: `cp autonomous_orchestrator.py backups/autonomous_orchestrator.py.<timestamp>`
   - Send notification: "✅ Changes applied and committed. Restarting..."
   - Trigger container restart (method depends on deployment):
     - Docker: `docker restart <container_name>`
     - Kubernetes: `kubectl rollout restart deployment/<name>`
     - Systemd: `systemctl restart autonomous-agent`
   - Agent restarts with new code
   - Post-restart verification: Send "🚀 Agent restarted successfully with v<new_version>"

### Safety Constraints
- **NEVER** use local Ollama models for self-improvement decisions
- **ALWAYS** validate proposed changes with Gemini (cloud API)
- **REQUIRE** explicit user approval for low-confidence changes (< 50%)
- **CREATE** git backups before applying changes
- **TEST** changes in isolated environment when possible
- **LOG** all force mode activations with user ID and timestamp
- **VERIFY** git repository is clean before self-update (no uncommitted changes)
- **NOTIFY** user before and after container restart

### Self-Update Telegram Handler Implementation

```python
async def cmd_selfupdate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger self-improvement analysis via Telegram"""
    if str(update.effective_user.id) != str(AUTHORIZED_USER_ID):
        return
    
    args = context.args if context.args else []
    force_mode = 'force' in args
    
    # Start self-update task
    task = Task(
        id=None,
        user_id=update.effective_user.id,
        description="Manual self-update triggered" + (" (FORCE)" if force_mode else ""),
        task_type=TaskType.SELF_UPDATE,
        parameters={'force': force_mode, 'initiated_by': 'telegram'},
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
    
    db = context.bot_data['db']
    executor = context.bot_data['executor']
    task.id = db.create_task(task)
    
    await update.message.reply_text(
        f"🔍 Self-update task #{task.id} created.{' Force mode enabled.' if force_mode else ''}",
        parse_mode='Markdown'
    )
    
    # Execute immediately
    await executor.execute_task(task, context)
```

### Self-Update Executor Logic

```python
async def _execute_self_update(self, task: Task, context: ContextTypes.DEFAULT_TYPE) -> str:
    """Execute self-improvement workflow"""
    force_mode = task.parameters.get('force', False)
    
    # Step 1: Collect data
    await self._notify_progress(task, context, "📊 Collecting system data...")
    code = await self._read_own_code()
    logs = await self._read_recent_logs(lines=100)
    metrics = await self._get_task_metrics()
    
    # Step 2: Query Gemini for analysis (NEVER use Ollama)
    await self._notify_progress(task, context, "🤖 Analyzing with Gemini...")
    analysis = await self._query_gemini_self_analysis(code, logs, metrics)
    
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

Proposed changes: {len(analysis.get('changes', []))} files"""
    
    await context.bot.send_message(
        chat_id=task.user_id,
        text=metrics_msg,
        parse_mode='Markdown'
    )
    
    # Step 5: Decision
    all_good = all(m >= 50 for m in [confidence, performance, resilience, pertinence])
    
    if force_mode:
        await self._apply_changes(analysis, task, context)
        return "Changes applied (force mode)"
    
    if all_good:
        # Good metrics - ask for confirmation
        await context.bot.send_message(
            chat_id=task.user_id,
            text="✅ All metrics ≥ 50%. Apply changes? Reply: YES or NO",
            parse_mode='Markdown'
        )
        # Wait for user response (handled by message handler)
        return "Awaiting user confirmation"
    else:
        # Low metrics - require explicit approval
        await context.bot.send_message(
            chat_id=task.user_id,
            text=f"⚠️ Low metrics detected. Review recommended.\nReply SHOWDIFF to see changes or YES to apply anyway.",
            parse_mode='Markdown'
        )
        return "Awaiting user approval (low confidence)"
```

### User Confirmation State Management

When a self-update requires user confirmation, the agent maintains state:

```python
# Global state for pending confirmations
pending_confirmations = {}

async def handle_confirmation_response(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user responses to self-update confirmations"""
    user_text = update.message.text.upper().strip()
    user_id = update.effective_user.id
    
    # Check if there's a pending confirmation for this user
    if user_id not in pending_confirmations:
        return False  # Not a confirmation response
    
    confirmation = pending_confirmations[user_id]
    task_id = confirmation['task_id']
    analysis = confirmation['analysis']
    
    if user_text == 'YES':
        # Apply changes
        await update.message.reply_text("✅ Applying changes...")
        await apply_self_update_changes(analysis, task_id, context)
        del pending_confirmations[user_id]
        return True
        
    elif user_text == 'NO':
        await update.message.reply_text("❌ Self-update cancelled.")
        del pending_confirmations[user_id]
        return True
        
    elif user_text == 'SHOWDIFF':
        # Show detailed diff
        diff = generate_diff(analysis)
        await update.message.reply_text(
            f"📋 Proposed changes:\n```diff\n{diff[:3900]}\n```",
            parse_mode='Markdown'
        )
        await update.message.reply_text(
            "Reply YES to apply or NO to cancel."
        )
        return True
    
    return False
```

**State Structure:**
```python
pending_confirmations[user_id] = {
    'task_id': 123,
    'analysis': {...},
    'timestamp': datetime.now(),
    'expires_at': datetime.now() + timedelta(minutes=30)
}
```

**Timeout Handling:**
- Confirmations expire after 30 minutes
- Expired confirmations are cleaned up by a background job
- User notified: "⌛ Confirmation expired. Run /selfupdate again."

## Execution Flow

### Startup Sequence
```
1. Load environment variables
2. Initialize logging (File + Console)
3. Create TaskDatabase instance
4. Create TaskExecutor instance
5. Build Telegram Application
6. Register handlers and commands
7. Start background jobs
8. Begin polling for messages
9. Send "Bot Online" notification
```

### Background Jobs
1. **process_pending_tasks**: Every 60 seconds
   - Fetch up to MAX_WORKERS pending tasks
   - Execute each task

2. **check_retrying_tasks**: Every 60 seconds
   - Find tasks scheduled for retry
   - Reset to PENDING status when due

3. **cleanup_expired_confirmations**: Every 60 seconds
   - Check pending_confirmations dict for expired entries
   - Remove entries older than 30 minutes
   - Notify users of expired confirmations

4. **daily_self_analysis**: Every 24 hours (86400 seconds)
   - Trigger automatic self-improvement analysis
   - Collect code, logs, and metrics
   - Query Gemini for evaluation
   - Apply changes if ALL metrics ≥ 50%
   - Request user approval if ANY metric < 50%

5. **hourly_report**: Every 3 hours (10800 seconds)
   - Summarize recent actions
   - Send report via Telegram
   - Prefer local Ollama (configurable with FORCED_AI_REPORT)

### Task Execution
```
1. Set status to RUNNING
2. Execute based on task_type:
   - COMMAND: subprocess.run with 60s timeout
   - FILE_READ: open() and read
   - FILE_EDIT: backup + write
   - AI_ANALYSIS: call safe_generate()
   - PLAN: create subtasks
   - SELF_UPDATE: Run self-analysis workflow (Gemini only)
     * Collect code, logs, metrics
     * Query Gemini for evaluation
     * Display metrics to user
     * Await confirmation (or auto-apply if metrics ≥ 50%)
     * Apply changes, git commit, restart
3. On success: status = COMPLETED, notify user
4. On failure: status = RETRYING (if retries remain) or FAILED
5. Update database with results/errors
```

## Error Handling & Resilience

### AI Fallback Chain
```
Gemini API → [if 429 or error] → Ollama Local
```

### Retry Mechanism
- Failed tasks retry up to MAX_RETRIES times
- RETRY_DELAY_MINUTES interval between attempts
- Exponential backoff can be implemented

### Error Handler
Catches Telegram exceptions and sends simplified error messages to user.

### Database Resilience
- Threading.Lock for concurrent access
- Context managers for connection handling
- check_same_thread=False for async compatibility

## Security Considerations

### Authorization
- Only AUTHORIZED_USER_ID can interact with the bot
- All messages validated against user ID

### Command Execution
- Shell commands executed with subprocess
- 60-second timeout prevents hanging
- All commands logged for audit

### File Access
- No path restrictions currently implemented
- Backups created before file edits (.bak extension)
- Consider implementing path allowlist

### API Keys
- All sensitive data in environment variables
- Never log API keys
- Consider key rotation mechanism

## Development & Deployment

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install python-telegram-bot google-generative-ai httpx

# Set environment variables
export TELEGRAM_TOKEN=xxx
export GEMINI_API_KEY=yyy
export AUTHORIZED_USER_ID=123456789

# Run
python autonomous_orchestrator.py
```

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY autonomous_orchestrator.py .
RUN pip install python-telegram-bot google-generative-ai httpx
CMD ["python", "autonomous_orchestrator.py"]
```

### Recommended Directory Structure (Container)
```
/app/
├── autonomous_orchestrator.py  # Main code
├── tasks.db                    # SQLite database (volume)
├── autonomous.log              # Log file (volume)
└── backups/                    # Code backups (volume)
```

## Version Tracking

The agent maintains version information to track self-updates:

**Version Format**: `v<major>.<minor>.<patch>`
- **Major**: Breaking changes to architecture
- **Minor**: New features or significant improvements  
- **Patch**: Bug fixes and minor optimizations

**Version Storage**:
- Embedded in code as `AGENT_VERSION = "2.1.0"`
- Stored in database for rollback tracking
- Git tags: `git tag v2.1.0 -m "Self-update: Improved error handling"`

**Changelog Generation**:
After each self-update, the agent generates a changelog entry:
```
## v2.1.0 (2026-02-05)
- Self-improvement update
- Metrics: Confidence:75%, Performance:80%, Resilience:65%, Pertinence:70%
- Changes: Refactored error handling, added metrics tracking
- Initiated by: Telegram /selfupdate command
```

## Future Enhancements

### In Progress
1. ✅ **Telegram-triggered self-update** (`/selfupdate` command)
2. ✅ **Git integration** for code changes
3. ✅ **Container restart mechanism**
4. ⏳ **Metrics dashboard** (planned for v2.2)

### Short Term
1. Rollback mechanism (restore from git history)
2. A/B testing for self-updates (staging environment)
3. Automatic dependency updates
4. Code quality gates (linting, type checking)

### Long Term
1. Multi-user support with role-based access
2. Plugin system for custom task types
3. Webhook mode for production deployment
4. Integration with external APIs
5. Advanced planning with dependency graphs

## Monitoring & Observability

### Logs
- Location: Configurable via LOG_FILE
- Format: `%(asctime)s - %(levelname)s - %(message)s`
- Contains: All actions, errors, AI decisions

### Metrics to Track
- Task success/failure rates
- Average execution time per task type
- AI API quota usage
- Retry frequency
- Self-improvement effectiveness

### Alerting
- Immediate Telegram notifications for failures
- Hourly summary reports
- Quota exhaustion warnings
- Error rate spikes

## Dependencies

### Python Packages
```
python-telegram-bot>=20.0
google-generative-ai>=0.3.0
httpx>=0.24.0
```

### External Services
- Telegram Bot API
- Google Gemini API
- Ollama (optional, local)

## Notes

### Known Limitations
1. Threading.Lock may cause issues with async code
2. MAX_WORKERS not enforced for user-triggered tasks
3. No graceful shutdown implemented
4. File paths not validated/escaped

### Best Practices
- Use environment variables for all configuration
- Regular database backups
- Monitor AI quota usage
- Test self-improvements in staging first
- Keep git history clean

---

*Document Version: 1.1*
*Last Updated: 2026-02-05*
*Agent Version: 2.0*
