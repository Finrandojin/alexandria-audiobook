        // --- Toast & Confirm utilities ---
        function showToast(message, type = 'info', duration = 4000) {
            const container = document.getElementById('toast-container');
            const bgClass = type === 'success' ? 'bg-success' :
                           type === 'error' ? 'bg-danger' :
                           type === 'warning' ? 'bg-warning text-dark' : 'bg-info';
            const id = 'toast-' + Date.now();
            const html = `
                <div id="${id}" class="toast align-items-center text-white ${bgClass} border-0" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">${escapeHtml(message)}</div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>`;
            container.insertAdjacentHTML('beforeend', html);
            const el = document.getElementById(id);
            const toast = new bootstrap.Toast(el, { delay: duration });
            toast.show();
            el.addEventListener('hidden.bs.toast', () => el.remove());
        }

        function showConfirm(message) {
            return new Promise((resolve) => {
                const body = document.getElementById('confirmModalBody');
                body.textContent = message;
                const modal = new bootstrap.Modal(document.getElementById('confirmModal'));
                const okBtn = document.getElementById('confirmModalOk');
                const cancelBtn = document.getElementById('confirmModalCancel');

                function cleanup() {
                    okBtn.removeEventListener('click', onOk);
                    cancelBtn.removeEventListener('click', onCancel);
                    document.getElementById('confirmModal').removeEventListener('hidden.bs.modal', onHidden);
                }
                let resolved = false;
                function onOk() { resolved = true; cleanup(); modal.hide(); resolve(true); }
                function onCancel() { resolved = true; cleanup(); modal.hide(); resolve(false); }
                function onHidden() { if (!resolved) { cleanup(); resolve(false); } }

                okBtn.addEventListener('click', onOk);
                cancelBtn.addEventListener('click', onCancel);
                document.getElementById('confirmModal').addEventListener('hidden.bs.modal', onHidden);
                modal.show();
            });
        }

        // Big/long-running jobs (batch review, batch script generation) bill a
        // remote GPU by the hour while they run - confirm before committing to
        // that, since there's no way for the app to check your actual Thunder
        // balance automatically. Local has no cloud cost, so it's a no-op there.
        async function confirmIfRemote(taskLabel) {
            if (!currentIsRemote) { return true; }
            return await showConfirm(
                `This will run ${taskLabel} on your REMOTE LLM (Thunder) and will bill your ` +
                `account by the hour while it's running. Continue on remote, or Cancel and switch ` +
                `to Local in Setup first?`
            );
        }

        function escapeHtml(str) {
            if (str == null) { return ''; }
            return String(str)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')
                .replace(/"/g, '&quot;')
                .replace(/'/g, '&#39;');
        }

        // Parse a numeric input's value, falling back to `def` when the field is
        // empty/non-numeric. Uses Number.isFinite (not `|| def`) so a deliberate 0
        // is preserved rather than treated as falsy.
        function getNumFieldValue(id, def, isInt = false) {
            const raw = document.getElementById(id).value;
            const v = isInt ? parseInt(raw, 10) : parseFloat(raw);
            return Number.isFinite(v) ? v : def;
        }

        // --- Desktop notifications ---
        const TASK_LABELS = {
            script: 'Script generation',
            review: 'Script review',
            nicknames: 'Nickname discovery',
            audio: 'Audio generation',
            batch_review: 'Batch review',
            batch_script: 'Batch script generation',
            lora_training: 'LoRA training',
            voicelab: 'Voice Lab pipeline',
            preparer: 'Dataset preparer',
            batch_preparer: 'Dataset preparer batch',
            dataset_builder: 'Dataset builder batch',
        };

        // Notify the user that a long-running job finished, but only if they've
        // navigated away from the tab (no point popping up a notification for
        // something they're already watching).
        function notifyJobDone(taskName, detail = '') {
            if (!('Notification' in window) || Notification.permission !== 'granted') { return; }
            if (document.visibilityState === 'visible' && document.hasFocus()) { return; }
            const title = `${TASK_LABELS[taskName] || taskName} finished`;
            try {
                new Notification(title, { body: detail || 'Switch back to Alexandria to see the results.', icon: '/favicon.ico' });
            } catch (e) { /* notifications are a nice-to-have */ }
        }

        // Ask for permission the first time the user interacts with the page —
        // most browsers require a user gesture before the prompt will appear.
        if ('Notification' in window && Notification.permission === 'default') {
            document.addEventListener('click', function requestNotificationPermission() {
                Notification.requestPermission();
                document.removeEventListener('click', requestNotificationPermission);
            }, { once: true });
        }

        // --- Navigation ---
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                // Remove active class from all links
                document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
                // Add active to clicked
                e.target.classList.add('active');

                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
                // Show target tab
                const targetId = e.target.dataset.tab + '-tab';
                document.getElementById(targetId).style.display = 'block';

                // Trigger tab specific loads
                if (e.target.dataset.tab === 'editor') {
                    loadChunks();
                } else if (e.target.dataset.tab === 'voices') {
                    loadVoices();
                } else if (e.target.dataset.tab === 'designer') {
                    loadDesignedVoices();
                } else if (e.target.dataset.tab === 'training') {
                    loadLoraDatasets();
                    loadLoraModels();
                } else if (e.target.dataset.tab === 'dataset-builder') {
                    dsbLoadProjects(dsbCurrentProject);
                } else if (e.target.dataset.tab === 'preparer') {
                    loadPreparerOutputs();
                } else if (e.target.dataset.tab === 'voicelab') {
                    loadVoicelabConfig();
                    voicelabInspect();
                    refreshVoicelabHealth();
                } else if (e.target.dataset.tab === 'reports') {
                    loadReports();
                    loadCheckpoints();
                }
            });
        });

        // --- Theme ---
        const THEMES = [
            { key: 'light',       icon: 'fa-sun',      label: 'Light'      },
            { key: 'night',       icon: 'fa-moon',     label: 'Night'      },
            { key: 'super-night', icon: 'fa-circle',   label: 'Super Night'},
        ];

        function applyTheme(key) {
            const html = document.documentElement;
            if (key === 'light') {
                html.removeAttribute('data-theme');
            } else {
                html.setAttribute('data-theme', key);
            }
            const t = THEMES.find(t => t.key === key) || THEMES[0];
            document.getElementById('theme-icon').className = `fas ${t.icon}`;
            document.getElementById('theme-label').textContent = t.label;
            localStorage.setItem('alex-theme', key);
        }

        function cycleTheme() {
            const current = document.documentElement.getAttribute('data-theme') || 'light';
            const idx = THEMES.findIndex(t => t.key === current);
            const next = THEMES[(idx + 1) % THEMES.length];
            applyTheme(next.key);
        }

        // Sync button label/icon with whatever the anti-flash snippet already applied
        (function() {
            const saved = localStorage.getItem('alex-theme') || 'light';
            applyTheme(saved);
        })();

        // --- API Helpers ---
        const API = {
            _handleError: async (res) => {
                if (res.ok) { return; }
                let detail = res.statusText;
                try {
                    const body = await res.json();
                    if (body && body.detail) { detail = body.detail; }
                } catch (e) { /* non-JSON error body; fall back to statusText */ }
                const err = new Error(detail);
                err.status = res.status;
                throw err;
            },
            get: async (url) => {
                const res = await fetch(url);
                await API._handleError(res);
                return res.json();
            },
            post: async (url, data) => {
                const res = await fetch(url, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                await API._handleError(res);
                return res.json();
            },
            del: async (url) => {
                const res = await fetch(url, { method: 'DELETE' });
                await API._handleError(res);
                return res.json();
            },
            upload: async (file) => {
                const formData = new FormData();
                formData.append('file', file);
                const res = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                await API._handleError(res);
                return res.json();
            }
        };

        // --- Setup Tab ---

        function toggleTTSMode() {
            const mode = document.getElementById('tts-mode').value;
            document.getElementById('tts-url-group').style.display = mode === 'external' ? '' : 'none';
            document.getElementById('tts-device-group').style.display = mode === 'local' ? '' : 'none';
            document.getElementById('tts-local-options').style.display = mode === 'local' ? '' : 'none';
        }

        function toggleSubBatchFields() {
            const enabled = document.getElementById('sub-batch-enabled').checked;
            ['sub-batch-min-group', 'sub-batch-ratio-group', 'sub-batch-max-items-group'].forEach(id => {
                document.getElementById(id).style.display = enabled ? '' : 'none';
            });
        }

        async function autoConfigureSettings() {
            const btn = document.getElementById('btn-auto-configure');
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Detecting…';

            try {
                const stats = await API.get('/api/system/stats');
                const { settings, summary } = _computeAutoSettings(stats);
                _applyAutoSettings(settings);

                const banner = document.getElementById('auto-config-banner');
                document.getElementById('auto-config-msg').innerHTML =
                    `<i class="fas fa-check-circle me-1 text-success"></i><strong>Auto-configured:</strong> ${escapeHtml(summary)} — review the TTS settings below, then click Save.`;
                banner.style.display = '';
                banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (e) {
                showToast('Hardware detection failed: ' + e.message, 'error');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<i class="fas fa-magic me-1"></i>Auto-Configure';
            }
        }

        function _computeAutoSettings(stats) {
            // get_gpu_stats() falls back to rocm-smi for VRAM totals even when torch
            // can't use the GPU (gpu_mismatch), so stats.gpu can be populated with a
            // real, large total_gb in exactly the case where generation will actually
            // run on CPU. Treat that the same as "no usable GPU" here - otherwise this
            // confidently configures high parallelism/local mode for a GPU that's
            // sitting there unused, which is worse than the safe CPU-tier defaults.
            const gpuUsable = !!stats.gpu && !stats.gpu_mismatch;
            const vram = gpuUsable ? stats.gpu.total_gb : 0;
            const gpuName = stats.gpu_name || null;
            const ramGb = stats.ram_gb || null;

            let tier, settings;

            // Tier boundaries and maxItems calibrated from tts_vram_benchmark.py on RX 9070 XT (17.1 GB):
            // Model footprint: 4.18 GB. Peak VRAM per batch: 4-item→7.3 GB, 8-item→9.6 GB, 16-item→11.8 GB.
            // RTF: 4→2.3x, 8→3.5x (sweet spot), 12→2.8x, 16→2.9x, 24→2.7x (length-ratio caps at ~13 anyway).
            if (!gpuUsable || vram < 5) {
                // Model alone needs ~4.2 GB; under 5 GB it may not load reliably.
                // Same safe settings whether there's truly no GPU or one that's
                // physically present but unusable by torch (gpu_mismatch) - only
                // the label differs, so the two cases share one branch instead of
                // two copies of an identical settings object.
                tier = stats.gpu_mismatch
                    ? 'GPU detected but unusable by torch (wrong build) - defaulting to safe CPU settings'
                    : 'No GPU / insufficient VRAM';
                settings = { ttsMode: 'external', parallelWorkers: 1, compileCodec: false,
                    batchGroupByType: false, subBatchEnabled: true, subBatchMinSize: 4,
                    subBatchRatio: 5, subBatchMaxItems: 4 };
            } else if (vram < 8) {
                // 5–8 GB: model fits (~4.2 GB) but headroom is tight; long chunks peak ~7.3 GB total
                tier = `Low VRAM (${vram.toFixed(1)} GB)`;
                settings = { ttsMode: 'local', parallelWorkers: 1, compileCodec: false,
                    batchGroupByType: true, subBatchEnabled: true, subBatchMinSize: 4,
                    subBatchRatio: 5, subBatchMaxItems: 4 };
            } else if (vram < 12) {
                // 8–12 GB: ~4–8 GB headroom; 6-item batches peak ~8 GB total (safe)
                tier = `Mid VRAM (${vram.toFixed(1)} GB)`;
                settings = { ttsMode: 'local', parallelWorkers: 1, compileCodec: false,
                    batchGroupByType: true, subBatchEnabled: true, subBatchMinSize: 4,
                    subBatchRatio: 5, subBatchMaxItems: 6 };
            } else if (vram < 20) {
                // 12–20 GB: 8-item batches peak ~9.6 GB total — best measured RTF (3.5x)
                tier = `High VRAM (${vram.toFixed(1)} GB)`;
                settings = { ttsMode: 'local', parallelWorkers: 2, compileCodec: false,
                    batchGroupByType: true, subBatchEnabled: true, subBatchMinSize: 4,
                    subBatchRatio: 5, subBatchMaxItems: 8 };
            } else if (vram < 30) {
                // 20–30 GB: 16-item batches peak ~11.8 GB total — comfortable headroom
                tier = `Large VRAM (${vram.toFixed(1)} GB)`;
                settings = { ttsMode: 'local', parallelWorkers: 2, compileCodec: false,
                    batchGroupByType: true, subBatchEnabled: true, subBatchMinSize: 4,
                    subBatchRatio: 5, subBatchMaxItems: 16 };
            } else {
                // 30+ GB: length-ratio splitter will cap practical batch size anyway
                tier = `Enthusiast VRAM (${vram.toFixed(1)} GB)`;
                settings = { ttsMode: 'local', parallelWorkers: 4, compileCodec: false,
                    batchGroupByType: true, subBatchEnabled: true, subBatchMinSize: 4,
                    subBatchRatio: 5, subBatchMaxItems: 24 };
            }

            const parts = [tier];
            if (gpuName) { parts.push(gpuName); }
            if (ramGb) { parts.push(`${Math.round(ramGb)} GB RAM`); }
            if (stats.cpu_count) { parts.push(`${stats.cpu_count} CPU threads`); }
            const summary = parts.join(' · ');

            return { settings, summary };
        }

        function _applyAutoSettings(s) {
            document.getElementById('tts-mode').value = s.ttsMode;
            toggleTTSMode();
            document.getElementById('parallel-workers').value = s.parallelWorkers;
            document.getElementById('compile-codec').checked = s.compileCodec;
            document.getElementById('batch-group-by-type').checked = s.batchGroupByType;
            document.getElementById('sub-batch-enabled').checked = s.subBatchEnabled;
            toggleSubBatchFields();
            document.getElementById('sub-batch-min-size').value = s.subBatchMinSize;
            document.getElementById('sub-batch-ratio').value = s.subBatchRatio;
            document.getElementById('sub-batch-max-items').value = s.subBatchMaxItems;
        }

        // Local/Remote LLM profile state. Each mode keeps its own base_url/key/model
        // so switching the toggle never clobbers the other location's settings.
        let llmProfiles = { local: {base_url:'', api_key:'local', model_name:''},
                            remote: {base_url:'', api_key:'local', model_name:''} };
        let currentLlmMode = 'local';
        let savedLlmMode = 'local';   // last-saved llm_mode; drives the "Active:" badge
        // Server-computed lmstudio_settings.is_remote_llm(llm_mode, base_url)
        // for the ACTIVE (saved) config - reflects llm_mode/base_url drift
        // that a bare `currentLlmMode === 'remote'` check would miss. Updated
        // from /api/config's response on load and after a successful save.
        let currentIsRemote = false;

        function renderConfigWarnings(config) {
            const banner = document.getElementById('config-warning-banner');
            const message = document.getElementById('config-warning-msg');
            const warnings = Array.isArray(config.config_warnings) ? config.config_warnings : [];
            if (!warnings.length) {
                message.textContent = '';
                banner.style.display = 'none';
                return;
            }
            const details = warnings.map((warning) => {
                const field = warning && warning.field && warning.field !== '$'
                    ? warning.field + ': ' : '';
                return field + ((warning && warning.message) || 'Invalid saved setting ignored');
            }).join('; ');
            const recovery = config.config_needs_backup
                ? ' Saving will preserve the damaged file as a backup before replacing it.'
                : '';
            message.textContent = 'Some saved configuration could not be used; safe defaults are shown.'
                + recovery + ' ' + details;
            banner.style.display = '';
        }

        function populateLlmInputs(mode) {
            const p = llmProfiles[mode] || {};
            document.getElementById('llm-url').value = p.base_url || '';
            document.getElementById('llm-key').value = p.api_key || 'local';
            document.getElementById('llm-model').value = p.model_name || '';
        }

        function syncCurrentLlmProfile() {
            llmProfiles[currentLlmMode] = {
                base_url: document.getElementById('llm-url').value,
                api_key: document.getElementById('llm-key').value,
                model_name: document.getElementById('llm-model').value
            };
        }

        // Reflects the last-SAVED llm_mode, not the dropdown's live selection
        // (currentLlmMode) - the gap between the two is what tells the user
        // their Location change hasn't taken effect yet.
        function renderActiveLlmModeBadge() {
            const badge = document.getElementById('llm-active-mode-badge');
            if (!badge) { return; }
            badge.textContent = 'Active: ' + (savedLlmMode === 'remote' ? 'Remote (Thunder / network)' : 'Local');
        }

        function onLlmModeChange(isInit) {
            const newMode = document.getElementById('llm-mode').value;
            if (!isInit && newMode !== currentLlmMode) {
                syncCurrentLlmProfile();           // stash the mode we're leaving
                currentLlmMode = newMode;
                populateLlmInputs(currentLlmMode); // show the mode we're entering
            }
            document.getElementById('llm-ssh-group').style.display =
                (newMode === 'remote') ? '' : 'none';
            document.getElementById('llm-test-result').innerHTML = '';
        }

        async function testLlmConnection() {
            const btn = document.getElementById('llm-test-btn');
            const out = document.getElementById('llm-test-result');
            const baseUrl = document.getElementById('llm-url').value.trim();
            if (!baseUrl) {
                out.className = 'ms-2 small text-danger';
                out.innerHTML = '<i class="fas fa-times me-1"></i>Enter a Base URL before testing.';
                return;
            }
            btn.disabled = true;
            out.className = 'ms-2 small text-muted';
            out.textContent = 'Testing…';
            try {
                const res = await API.post('/api/llm/test', {
                    base_url: document.getElementById('llm-url').value,
                    api_key: document.getElementById('llm-key').value,
                    model_name: document.getElementById('llm-model').value
                });
                if (res.ok) {
                    out.className = 'ms-2 small text-success';
                    const note = res.model_present === false
                        ? ` — warning: model "${escapeHtml(res.model)}" not in server list` : '';
                    out.innerHTML = `<i class="fas fa-check me-1"></i>Connected. Reply: "${escapeHtml(res.reply || '')}"${note}`;

                    const modeLabel = res.is_remote ? 'Remote' : 'Local';
                    const banner = document.getElementById('auto-config-banner');
                    document.getElementById('auto-config-msg').innerHTML =
                        `<i class="fas fa-check-circle me-1 text-success"></i><strong>Auto-configured:</strong> ` +
                        `${modeLabel} LLM connected (${escapeHtml(res.base_url)}, model "${escapeHtml(res.model)}") — click Save to apply.`;
                    banner.style.display = '';
                    banner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                } else {
                    out.className = 'ms-2 small text-danger';
                    const log = res.log_file ? ` (log: ${escapeHtml(res.log_file)})` : '';
                    out.innerHTML = `<i class="fas fa-times me-1"></i>Failed at ${escapeHtml(res.step)}: ${escapeHtml(res.error || '')}${log}`;
                }
            } catch (e) {
                out.className = 'ms-2 small text-danger';
                out.textContent = 'Test request failed: ' + (e.message || 'unknown error');
            } finally {
                btn.disabled = false;
            }
        }

        async function loadConfig() {
            document.getElementById('chunk-size').value = 3000;
            document.getElementById('max-tokens').value = 4096;

            try {
                const config = await API.get('/api/config');
                renderConfigWarnings(config);
                // Local/Remote LLM profiles: keep both in memory, show the active one.
                llmProfiles.local = config.llm_local || config.llm || {base_url:'', api_key:'local', model_name:''};
                llmProfiles.remote = config.llm_remote || {base_url:'', api_key:'local', model_name:''};
                currentLlmMode = config.llm_mode || 'local';
                savedLlmMode = currentLlmMode;
                currentIsRemote = !!config.is_remote;
                renderActiveLlmModeBadge();
                document.getElementById('llm-mode').value = currentLlmMode;
                document.getElementById('llm-ssh').value = config.llm_remote_ssh || '';
                populateLlmInputs(currentLlmMode);
                onLlmModeChange(true);
                document.getElementById('tts-mode').value = config.tts.mode || 'external';
                document.getElementById('tts-url').value = config.tts.url || 'http://127.0.0.1:7860';
                document.getElementById('tts-device').value = config.tts.device || 'auto';
                document.getElementById('tts-language').value = config.tts.language || 'English';
                document.getElementById('parallel-workers').value = config.tts.parallel_workers || 2;
                if (config.tts.batch_seed != null) {
                    document.getElementById('batch-seed').value = config.tts.batch_seed;
                }
                document.getElementById('compile-codec').checked = !!config.tts.compile_codec;
                if (config.tts.max_new_tokens != null) {
                    document.getElementById('tts-max-new-tokens').value = config.tts.max_new_tokens;
                }
                document.getElementById('batch-group-by-type').checked = !!config.tts.batch_group_by_type;
                document.getElementById('sub-batch-enabled').checked = config.tts.sub_batch_enabled !== false;
                toggleSubBatchFields();
                if (config.tts.sub_batch_min_size != null) {
                    document.getElementById('sub-batch-min-size').value = config.tts.sub_batch_min_size;
                }
                if (config.tts.sub_batch_ratio != null) {
                    document.getElementById('sub-batch-ratio').value = config.tts.sub_batch_ratio;
                }
                if (config.tts.sub_batch_max_items != null) {
                    document.getElementById('sub-batch-max-items').value = config.tts.sub_batch_max_items;
                }
                if (config.tts.pause_between_speakers_ms != null) {
                    document.getElementById('pause-between-speakers').value = config.tts.pause_between_speakers_ms;
                }
                if (config.tts.pause_same_speaker_ms != null) {
                    document.getElementById('pause-same-speaker').value = config.tts.pause_same_speaker_ms;
                }
                toggleTTSMode();

                // Load custom prompts if they exist and are non-empty
                if (config.prompts) {
                    if (config.prompts.system_prompt) {
                        document.getElementById('system-prompt').value = config.prompts.system_prompt;
                    }
                    if (config.prompts.user_prompt) {
                        document.getElementById('user-prompt').value = config.prompts.user_prompt;
                    }
                    if (config.prompts.review_system_prompt) {
                        document.getElementById('review-system-prompt').value = config.prompts.review_system_prompt;
                    }
                    if (config.prompts.review_user_prompt) {
                        document.getElementById('review-user-prompt').value = config.prompts.review_user_prompt;
                    }
                    if (config.prompts.persona_system_prompt) {
                        document.getElementById('persona-system-prompt').value = config.prompts.persona_system_prompt;
                    }
                    if (config.prompts.persona_user_prompt) {
                        document.getElementById('persona-user-prompt').value = config.prompts.persona_user_prompt;
                    }
                    if (config.prompts.persona_advanced_prompt) {
                        document.getElementById('persona-advanced-prompt').value = config.prompts.persona_advanced_prompt;
                    }
                }

                // If review/persona prompts are still empty, fetch defaults
                if (!document.getElementById('review-system-prompt').value || !document.getElementById('review-user-prompt').value
                    || !document.getElementById('persona-system-prompt').value || !document.getElementById('persona-user-prompt').value) {
                    try {
                        const defaults = await API.get('/api/default_prompts');
                        if (!document.getElementById('review-system-prompt').value && defaults.review_system_prompt) {
                            document.getElementById('review-system-prompt').value = defaults.review_system_prompt;
                        }
                        if (!document.getElementById('review-user-prompt').value && defaults.review_user_prompt) {
                            document.getElementById('review-user-prompt').value = defaults.review_user_prompt;
                        }
                        if (!document.getElementById('persona-system-prompt').value && defaults.persona_system_prompt) {
                            document.getElementById('persona-system-prompt').value = defaults.persona_system_prompt;
                        }
                        if (!document.getElementById('persona-user-prompt').value && defaults.persona_user_prompt) {
                            document.getElementById('persona-user-prompt').value = defaults.persona_user_prompt;
                        }
                        if (!document.getElementById('persona-advanced-prompt').value && defaults.persona_advanced_prompt) {
                            document.getElementById('persona-advanced-prompt').value = defaults.persona_advanced_prompt;
                        }
                    } catch (e) {
                        console.warn("Could not fetch default prompts", e);
                    }
                }

                // Load generation settings
                if (config.generation) {
                    if (config.generation.chunk_size) {
                        document.getElementById('chunk-size').value = config.generation.chunk_size;
                    }
                    if (config.generation.max_tokens) {
                        document.getElementById('max-tokens').value = config.generation.max_tokens;
                    }
                    if (config.generation.temperature != null) {
                        document.getElementById('temperature').value = config.generation.temperature;
                    }
                    if (config.generation.top_p != null) {
                        document.getElementById('top-p').value = config.generation.top_p;
                    }
                    if (config.generation.top_k != null) {
                        document.getElementById('top-k').value = config.generation.top_k;
                    }
                    if (config.generation.min_p != null) {
                        document.getElementById('min-p').value = config.generation.min_p;
                    }
                    if (config.generation.presence_penalty != null) {
                        document.getElementById('presence-penalty').value = config.generation.presence_penalty;
                    }
                    if (config.generation.banned_tokens && config.generation.banned_tokens.length > 0) {
                        document.getElementById('banned-tokens').value = config.generation.banned_tokens.join(', ');
                    }
                    document.getElementById('merge-narrators').checked = !!config.generation.merge_narrators;
                }

                // Show previously loaded file
                if (config.current_file) {
                    document.getElementById('upload-status').innerHTML =
                        `<span class="text-success"><i class="fas fa-check me-1"></i>Loaded: ${config.current_file}</span>`;
                }
            } catch (e) {
                console.error("Failed to load config", e);
            }
        }

        // Reset prompts and generation settings to factory defaults
        window.resetPrompts = async () => {
            try {
                const defaults = await API.get('/api/default_prompts');
                document.getElementById('system-prompt').value = defaults.system_prompt;
                document.getElementById('user-prompt').value = defaults.user_prompt;
                if (defaults.review_system_prompt) {
                    document.getElementById('review-system-prompt').value = defaults.review_system_prompt;
                }
                if (defaults.review_user_prompt) {
                    document.getElementById('review-user-prompt').value = defaults.review_user_prompt;
                }
                if (defaults.persona_system_prompt) {
                    document.getElementById('persona-system-prompt').value = defaults.persona_system_prompt;
                }
                if (defaults.persona_user_prompt) {
                    document.getElementById('persona-user-prompt').value = defaults.persona_user_prompt;
                }
                if (defaults.persona_advanced_prompt) {
                    document.getElementById('persona-advanced-prompt').value = defaults.persona_advanced_prompt;
                }
            } catch (e) {
                console.error("Failed to fetch default prompts", e);
                showToast("Failed to load default prompts from server.", 'error');
            }
            document.getElementById('chunk-size').value = 3000;
            document.getElementById('max-tokens').value = 4096;
            document.getElementById('temperature').value = 0.6;
            document.getElementById('top-p').value = 0.8;
            document.getElementById('top-k').value = 0;
            document.getElementById('min-p').value = 0;
            document.getElementById('presence-penalty').value = 0;
            document.getElementById('banned-tokens').value = '';
            document.getElementById('merge-narrators').checked = false;
        };

        // Toggle chevron on collapse
        document.getElementById('promptSettings')?.addEventListener('show.bs.collapse', () => {
            document.getElementById('prompt-chevron').classList.replace('fa-chevron-right', 'fa-chevron-down');
        });
        document.getElementById('promptSettings')?.addEventListener('hide.bs.collapse', () => {
            document.getElementById('prompt-chevron').classList.replace('fa-chevron-down', 'fa-chevron-right');
        });

        document.getElementById('config-form').addEventListener('submit', async (e) => {
            e.preventDefault();

            let chunkSize = parseInt(document.getElementById('chunk-size').value) || 3000;

            // Validate parallel workers
            let parallelWorkers = parseInt(document.getElementById('parallel-workers').value) || 2;
            parallelWorkers = Math.max(1, parallelWorkers);
            document.getElementById('parallel-workers').value = parallelWorkers;

            // Persist whatever's currently shown into the active profile first,
            // then send both profiles + the active one (mirrored server-side into `llm`).
            syncCurrentLlmProfile();
            const config = {
                llm: llmProfiles[currentLlmMode],
                llm_mode: currentLlmMode,
                llm_local: llmProfiles.local,
                llm_remote: llmProfiles.remote,
                llm_remote_ssh: document.getElementById('llm-ssh').value.trim() || null,
                tts: {
                    mode: document.getElementById('tts-mode').value,
                    url: document.getElementById('tts-url').value,
                    device: document.getElementById('tts-device').value,
                    language: document.getElementById('tts-language').value,
                    parallel_workers: parallelWorkers,
                    batch_seed: document.getElementById('batch-seed').value ? parseInt(document.getElementById('batch-seed').value) : null,
                    compile_codec: document.getElementById('compile-codec').checked,
                    max_new_tokens: getNumFieldValue('tts-max-new-tokens', 2048, true),
                    batch_group_by_type: document.getElementById('batch-group-by-type').checked,
                    sub_batch_enabled: document.getElementById('sub-batch-enabled').checked,
                    sub_batch_min_size: getNumFieldValue('sub-batch-min-size', 4, true),
                    sub_batch_ratio: getNumFieldValue('sub-batch-ratio', 5),
                    sub_batch_max_items: getNumFieldValue('sub-batch-max-items', 0, true),
                    pause_between_speakers_ms: getNumFieldValue('pause-between-speakers', 500, true),
                    pause_same_speaker_ms: getNumFieldValue('pause-same-speaker', 250, true)
                },
                prompts: {
                    system_prompt: document.getElementById('system-prompt').value,
                    user_prompt: document.getElementById('user-prompt').value,
                    review_system_prompt: document.getElementById('review-system-prompt').value,
                    review_user_prompt: document.getElementById('review-user-prompt').value,
                    persona_system_prompt: document.getElementById('persona-system-prompt').value,
                    persona_user_prompt: document.getElementById('persona-user-prompt').value,
                    persona_advanced_prompt: document.getElementById('persona-advanced-prompt').value
                },
                generation: {
                    chunk_size: chunkSize,
                    max_tokens: parseInt(document.getElementById('max-tokens').value) || 4096,
                    temperature: getNumFieldValue('temperature', 0.6),
                    top_p: getNumFieldValue('top-p', 0.8),
                    top_k: getNumFieldValue('top-k', 0, true),
                    min_p: getNumFieldValue('min-p', 0),
                    presence_penalty: getNumFieldValue('presence-penalty', 0.0),
                    banned_tokens: document.getElementById('banned-tokens').value
                        ? document.getElementById('banned-tokens').value.split(',').map(t => t.trim()).filter(t => t)
                        : [],
                    merge_narrators: document.getElementById('merge-narrators').checked
                }
            };
            try {
                await API.post('/api/config', config);
                savedLlmMode = currentLlmMode;
                renderActiveLlmModeBadge();
                // Refresh is_remote from the now-active saved config (not a
                // full loadConfig() - that resets unrelated fields like
                // chunk-size to hardcoded defaults).
                try {
                    const savedConfig = await API.get('/api/config');
                    currentIsRemote = !!savedConfig.is_remote;
                    renderConfigWarnings(savedConfig);
                }
                catch (e) { console.debug('is_remote refresh after save failed', e); }
                showToast('Configuration Saved!', 'success');
            } catch (e) {
                showToast('Error saving config: ' + e.message, 'error');
            }
        });

        // --- Script Tab ---
        async function loadExistingScriptUploads() {
            try {
                const uploads = await API.get('/api/uploads');
                const options = uploads.map(item =>
                    `<option value="${escapeHtml(item.filename)}">${escapeHtml(item.filename)} (${Math.ceil(item.size / 1024)} KB)</option>`
                ).join('');
                document.getElementById('existing-upload-select').innerHTML =
                    '<option value="">Choose an existing TXT/MD file…</option>' + options;
                scriptBatchUploads = uploads;
                renderScriptBatchUploads();
            } catch (e) {
                console.debug('Existing upload list unavailable', e);
            }
        }

        window.selectExistingScriptUpload = async () => {
            const select = document.getElementById('existing-upload-select');
            if (!select.value) { return; }
            const statusEl = document.getElementById('upload-status');
            try {
                const result = await API.post('/api/uploads/select', { filename: select.value });
                document.getElementById('file-upload').value = '';
                statusEl.innerHTML = `<span class="text-success"><i class="fas fa-check me-1"></i>Reusing: ${escapeHtml(result.stored_filename)}</span>`;
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger">Failed to select upload: ${escapeHtml(e.message)}</span>`;
            }
        };

        document.getElementById('file-upload').addEventListener('change', async () => {
            const fileInput = document.getElementById('file-upload');
            const statusEl = document.getElementById('upload-status');
            if (fileInput.files.length === 0) { return; }

            statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Loading file...</span>';
            try {
                const res = await API.upload(fileInput.files[0]);
                document.getElementById('existing-upload-select').value = '';
                const verb = res.reused ? 'Reused existing copy' : 'Loaded';
                statusEl.innerHTML = `<span class="text-success"><i class="fas fa-check me-1"></i>${verb}: ${escapeHtml(res.stored_filename)}</span>`;
                await loadExistingScriptUploads();
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>Failed to load file: ${escapeHtml(e.message)}</span>`;
            }
        });

        document.getElementById('btn-gen-script').addEventListener('click', async () => {
            if (document.getElementById('script-batch-mode').checked) {
                return _startBatchScript();
            }

            const fileInput = document.getElementById('file-upload');
            const statusEl = document.getElementById('upload-status');

            const hasLoadedFile = statusEl.innerHTML.includes('text-success');
            if (!hasLoadedFile && fileInput.files.length === 0) {
                statusEl.innerHTML = '<span class="text-danger"><i class="fas fa-exclamation-triangle me-1"></i>Please select a text file first using the file picker above.</span>';
                return;
            }

            const genBtn = document.getElementById('btn-gen-script');
            const cancelBtn = document.getElementById('btn-cancel-script');
            const pauseBtn = document.getElementById('btn-pause-script');
            genBtn.disabled = true;
            cancelBtn.style.display = 'inline-block';
            pauseBtn.style.display = 'inline-block';
            pauseBtn.innerHTML = '<i class="fas fa-pause me-1"></i>Pause';
            pauseBtn.classList.remove('btn-outline-success');
            pauseBtn.classList.add('btn-outline-warning');
            _resetPauseBtn('btn-pause-script');

            try {
                await API.post('/api/generate_script', {
                    strip_front_matter: _isStripFrontMatterChecked(),
                    first_person_narrator:
                        document.getElementById('script-first-person-narrator').value.trim() || null,
                });
                pollLogs('script', 'script-logs', () => {
                    if (!scriptBatchPoller) { genBtn.disabled = false; }
                    cancelBtn.style.display = 'none';
                    pauseBtn.style.display = 'none';
                });
            } catch (e) {
                genBtn.disabled = false;
                cancelBtn.style.display = 'none';
                pauseBtn.style.display = 'none';
                const detail = e.message || 'Unknown error';
                if (detail.includes('No input file')) {
                    statusEl.innerHTML = '<span class="text-danger"><i class="fas fa-exclamation-triangle me-1"></i>No file loaded. Please select a text file first.</span>';
                } else {
                    statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${escapeHtml(detail)}</span>`;
                }
            }
        });

        // Restores a pause/resume button to its "Pause"/btn-outline-warning
        // appearance. _makePauseResumeHandler derives paused/running state from
        // this button's own class, so every action that can leave a process
        // un-paused (cancel, fresh start) must call this to keep the DOM in
        // sync — otherwise the next click reads stale "Resume" styling and
        // calls the wrong endpoint.
        function _resetPauseBtn(btnId) {
            const btn = document.getElementById(btnId);
            if (!btn) { return; }
            btn.innerHTML = '<i class="fas fa-pause me-1"></i>Pause';
            btn.classList.remove('btn-outline-success');
            btn.classList.add('btn-outline-warning');
        }

        // Shared by every cancel-button handler in this file - posts to the
        // cancel endpoint, runs an optional onSuccess callback, and toasts on
        // failure. Added so a new cancel button doesn't have to remember to
        // copy the try/catch+toast pattern by hand.
        async function cancelTask(url, {
            onSuccess,
            errorMessage = (e) => 'Cancel failed: ' + (e.message || 'unknown error'),
            toastType = 'warning',
        } = {}) {
            try {
                await API.post(url, {});
                if (onSuccess) { onSuccess(); }
            } catch (e) {
                showToast(errorMessage(e), toastType);
            }
        }

        // Same "unknown error" fallback cancelTask's default errorMessage
        // uses, for callers (debounced autosaves) that POST a real body and
        // so can't go through cancelTask itself, which always posts {}.
        function _toastSaveError(action, e) {
            showToast(`Failed to save ${action}: ` + (e.message || 'unknown error'), 'warning');
        }

        function _makePauseResumeHandler(pauseUrl, resumeUrl, btnId) {
            // Retry once or twice on 503 ("starting up, retry in a moment") instead of
            // making the user notice the toast and click again themselves.
            const postWithRetry = async (url) => {
                for (let attempt = 0; ; attempt++) {
                    try {
                        return await API.post(url, {});
                    } catch (e) {
                        if (e.status === 503 && attempt < 2) {
                            await new Promise(r => setTimeout(r, 700));
                            continue;
                        }
                        throw e;
                    }
                }
            };
            // Paused/running is derived from the button's own class each time —
            // the button is the single source of truth, so there's nothing to
            // reset when a new run starts (every start path already restores the
            // button to its "Pause"/btn-outline-warning appearance).
            return async () => {
                const btn = document.getElementById(btnId);
                if (btn.disabled) { return; }
                const paused = btn.classList.contains('btn-outline-success');
                btn.disabled = true;
                try {
                    if (!paused) {
                        await postWithRetry(pauseUrl);
                        btn.innerHTML = '<i class="fas fa-play me-1"></i>Resume';
                        btn.classList.remove('btn-outline-warning');
                        btn.classList.add('btn-outline-success');
                    } else {
                        await postWithRetry(resumeUrl);
                        btn.innerHTML = '<i class="fas fa-pause me-1"></i>Pause';
                        btn.classList.remove('btn-outline-success');
                        btn.classList.add('btn-outline-warning');
                    }
                } catch (e) {
                    showToast((paused ? 'Resume' : 'Pause') + ' failed: ' + (e.message || 'unknown error'), 'warning');
                } finally {
                    btn.disabled = false;
                }
            };
        }

        const _scriptPauseResume = _makePauseResumeHandler(
            '/api/generate_script/pause', '/api/generate_script/resume', 'btn-pause-script');
        const _batchPauseResume  = _makePauseResumeHandler(
            '/api/generate_script/batch/pause', '/api/generate_script/batch/resume', 'btn-pause-batch-script');

        window.cancelScript = () => cancelTask('/api/generate_script/cancel', {
            onSuccess: () => _resetPauseBtn('btn-pause-script'),
        });
        window.pauseResumeScript      = _scriptPauseResume;

        // --- Script Batch Mode ---
        let scriptBatchQueue = [];
        let scriptBatchUploads = [];
        let scriptBatchPoller = null;

        // Checkboxes instead of a native <select multiple> - selecting a
        // subset of a multi-select normally requires ctrl/shift-click, which
        // doesn't work over a remote-desktop session with no modifier keys.
        // Reuses _renderScriptCheckboxList (also used by the review-batch and
        // cast-bulk pickers) via its keyField/renderLabel/onchange/emptyMessage
        // options instead of a parallel duplicate implementation.
        function renderScriptBatchUploads() {
            _renderScriptCheckboxList(scriptBatchUploads, {
                containerId: 'script-existing-uploads',
                checkClass: 'script-batch-upload-check',
                idPrefix: 'script-batch-upload-',
                keyField: 'filename',
                onchange: 'onScriptBatchFilesChange()',
                emptyMessage: 'No existing uploads found.',
                renderLabel: (item) => `${escapeHtml(item.filename)} (${Math.ceil(item.size / 1024)} KB)`,
            });
        }

        window.toggleScriptBatchMode = () => {
            const isBatch = document.getElementById('script-batch-mode').checked;
            document.getElementById('script-single-area').style.display = isBatch ? 'none' : 'block';
            document.getElementById('script-batch-area').style.display  = isBatch ? 'block' : 'none';
            document.getElementById('script-batch-status-msg').style.display = isBatch ? '' : 'none';
        };

        window.onScriptBatchFilesChange = () => {
            const files = document.getElementById('script-batch-files').files;
            const existing = [...document.querySelectorAll('.script-batch-upload-check:checked')];
            const tbody = document.getElementById('script-batch-queue-body');
            tbody.innerHTML = '';
            scriptBatchQueue = [];
            document.getElementById('script-batch-selected-count').textContent =
                `${files.length + existing.length} selected`;

            if (!files.length && !existing.length) {
                document.getElementById('script-batch-queue-container').style.display = 'none';
                return;
            }
            document.getElementById('script-batch-queue-container').style.display = 'block';

            [...files].forEach((file, i) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="text-truncate" style="max-width:350px;">${escapeHtml(file.name)}</td>
                    <td><input class="form-control form-control-sm" id="script-batch-narrator-${i}" maxlength="100" placeholder="Exact character name"></td>
                    <td id="script-batch-status-${i}"><span class="badge bg-secondary">Pending</span></td>
                `;
                tbody.appendChild(row);
                scriptBatchQueue.push({ file });
            });
            existing.forEach((checkbox) => {
                const i = scriptBatchQueue.length;
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="text-truncate" style="max-width:350px;">${escapeHtml(checkbox.dataset.name)} <span class="text-muted">(existing)</span></td>
                    <td><input class="form-control form-control-sm" id="script-batch-narrator-${i}" maxlength="100" placeholder="Exact character name"></td>
                    <td id="script-batch-status-${i}"><span class="badge bg-secondary">Pending</span></td>
                `;
                tbody.appendChild(row);
                scriptBatchQueue.push({ storedFilename: checkbox.dataset.name });
            });
        };

        window.scriptBatchSelectAll = (on) => _selectAllCheckboxes(
            'script-batch-upload-check', on, onScriptBatchFilesChange);

        window.scriptBatchSort = (mode) => {
            // `name` is only needed transiently for _sortScriptList's
            // comparator (it ignores the upload extension and duplicate
            // suffix when finding a book's volume number - "Volume
            // 10_3.txt" is volume 10) - dropped again before persisting
            // back into scriptBatchUploads since nothing else reads it.
            const sortable = scriptBatchUploads.map(item => ({
                name: item.filename.replace(/\.[^.]+$/, '').replace(/_\d+$/, ''),
                filename: item.filename,
                size: item.size,
            }));
            _sortScriptList(sortable, mode);
            scriptBatchUploads = sortable.map(({ filename, size }) => ({ filename, size }));
            renderScriptBatchUploads();
            onScriptBatchFilesChange();
        };

        window.cancelBatchScript = () => cancelTask('/api/generate_script/batch/cancel', {
            onSuccess: () => _resetPauseBtn('btn-pause-batch-script'),
        });

        window.pauseResumeBatchScript = _batchPauseResume;


        async function _startBatchScript() {
            if (!scriptBatchQueue.length) { showToast('No files selected', 'warning'); return; }
            if (!(await confirmIfRemote('this batch script generation'))) { return; }

            const btn = document.getElementById('btn-gen-script');
            const pauseBtn = document.getElementById('btn-pause-batch-script');
            const statusMsg = document.getElementById('script-batch-status-msg');
            btn.disabled = true;
            pauseBtn.style.display = 'inline-block';
            _resetPauseBtn('btn-pause-batch-script');
            document.getElementById('btn-cancel-batch-script').style.display = 'inline-block';
            statusMsg.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Uploading files…</span>';
            statusMsg.style.display = '';

            const logEl = document.getElementById('script-logs');
            logEl.innerHTML = '';

            try {
                // Upload files in parallel; map preserves queue order for `tasks`.
                const tasks = await Promise.all(scriptBatchQueue.map(async (item, index) => {
                    const firstPersonNarrator =
                        document.getElementById(`script-batch-narrator-${index}`).value.trim() || null;
                    if (item.storedFilename) {
                        return {
                            filename: item.storedFilename,
                            first_person_narrator: firstPersonNarrator,
                        };
                    }
                    const res = await API.upload(item.file);
                    // Use stored_filename if provided (handles epub→txt), otherwise fall back
                    return {
                        filename: res.stored_filename || res.filename,
                        first_person_narrator: firstPersonNarrator,
                    };
                }));

                const collisionPolicy = document.getElementById('script-collision-policy').value;
                const preflight = await API.post('/api/generate_script/batch/preflight', {
                    tasks, collision_policy: collisionPolicy
                });
                const scripts = [...new Set(preflight.books.flatMap(book => book.scripts))];
                const fallback = preflight.fallback_reason
                    ? `\nSafety adjustment: ${preflight.fallback_reason}` : '';
                const approved = await showConfirm(
                    `Batch preflight (${preflight.book_count} book${preflight.book_count === 1 ? '' : 's'}):\n` +
                    `Concurrency: ${preflight.workers} (LM Studio loaded for ${preflight.loaded_parallel})\n` +
                    `Context per worker: ${preflight.per_slot_context.toLocaleString()} tokens\n` +
                    `Largest predicted request: ${preflight.worst_request_tokens.toLocaleString()} tokens\n` +
                    `Writing systems detected: ${scripts.join(', ') || 'none'}${fallback}\n\nStart generation?`
                );
                if (!approved) {
                    statusMsg.innerHTML = '<span class="text-muted">Batch cancelled after preflight.</span>';
                    btn.disabled = false;
                    pauseBtn.style.display = 'none';
                    document.getElementById('btn-cancel-batch-script').style.display = 'none';
                    return;
                }

                statusMsg.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Processing…</span>';
                await API.post('/api/generate_script/batch/start', {
                    tasks, collision_policy: collisionPolicy,
                    strip_front_matter: _isStripFrontMatterChecked()
                });
                _pollScriptBatchLogs();
            } catch (e) {
                showToast('Failed to start batch: ' + e.message, 'error');
                btn.disabled = false;
                pauseBtn.style.display = 'none';
                document.getElementById('btn-cancel-batch-script').style.display = 'none';
            }
        }

        loadExistingScriptUploads();

        function _pollScriptBatchLogs() {
            const logEl = document.getElementById('script-logs');
            let offset = 0;
            // scriptBatchPoller is read elsewhere (pollLogs('script', ...)'s
            // onDone callbacks) to decide whether the single-script Generate
            // button should re-enable - it's a plain "is batch_script
            // currently polling" flag, not the timer id; the actual poll
            // loop is owned by _startPolling.
            scriptBatchPoller = true;

            _startPolling('batch_script', () => API.get('/api/status/batch_script'), {
                doneCheck: state => !state.running,
                onTick: state => {
                    const newLines = state.logs.slice(offset);
                    offset = state.logs.length;
                    newLines.forEach(line => {
                        const div = document.createElement('div');
                        div.textContent = line;
                        logEl.appendChild(div);
                    });
                    logEl.scrollTop = logEl.scrollHeight;

                    if (state.tasks) {
                        state.tasks.forEach((t, i) => {
                            const el = document.getElementById(`script-batch-status-${i}`);
                            if (!el) { return; }
                            const colours = { pending: 'secondary', running: 'primary', done: 'success', failed: 'danger', cancelled: 'warning' };
                            el.innerHTML = `<span class="badge bg-${colours[t.status] || 'secondary'}">${t.status}</span>`;
                        });
                    }
                },
                onDone: (state) => {
                    scriptBatchPoller = null;
                    notifyJobDone('batch_script');
                    document.getElementById('btn-gen-script').disabled = false;
                    document.getElementById('btn-pause-batch-script').style.display = 'none';
                    document.getElementById('btn-cancel-batch-script').style.display = 'none';
                    const tasks = state.tasks || [];
                    const done = tasks.filter(task => task.status === 'done').length;
                    const failed = tasks.filter(task => task.status === 'failed').length;
                    const cancelled = tasks.filter(task => task.status === 'cancelled').length;
                    const tone = failed ? 'text-warning' : 'text-muted';
                    document.getElementById('script-batch-status-msg').innerHTML =
                        `<span class="${tone}">Batch finished — ${done} completed, ${failed} failed, ${cancelled} cancelled. Failed books keep validated checkpoints.</span>`;
                    loadSavedScripts();
                }
            });
        }

        // --- Single review (with pause/cancel + character-name merging) ---
        function _isReviewDedupeChecked() {
            const cb = document.getElementById('review-dedupe-speakers');
            return cb ? cb.checked : true;
        }
        function _isStripFrontMatterChecked() {
            const cb = document.getElementById('script-strip-front-matter');
            return cb ? cb.checked : true;
        }
        function _showReviewControls(show) {
            document.getElementById('btn-pause-review').style.display = show ? 'inline-block' : 'none';
            document.getElementById('btn-cancel-review').style.display = show ? 'inline-block' : 'none';
            if (show) { _resetPauseBtn('btn-pause-review'); }
        }
        function _disableReviewButtons(disabled) {
            document.getElementById('btn-review-script').disabled = disabled;
            document.getElementById('btn-review-script-contextual').disabled = disabled;
        }
        function _onReviewDone() {
            _showReviewControls(false);
            _disableReviewButtons(false);
        }

        document.getElementById('btn-review-script').addEventListener('click', async () => {
            try {
                _disableReviewButtons(true);
                _showReviewControls(true);
                await API.post('/api/review_script', { dedupe_speakers: _isReviewDedupeChecked() });
                pollLogs('review', 'script-logs', _onReviewDone);
            } catch (e) {
                _onReviewDone();
                showToast("Failed to start review: " + e.message, 'error');
            }
        });

        document.getElementById('btn-review-script-contextual').addEventListener('click', async () => {
            try {
                const rawWindow = parseInt(document.getElementById('review-context-window').value, 10);
                const windowSize = Number.isFinite(rawWindow) ? Math.max(1, Math.min(rawWindow, 12)) : 4;
                _disableReviewButtons(true);
                _showReviewControls(true);
                const result = await API.post('/api/review_script_contextual', { window_size: windowSize, dedupe_speakers: _isReviewDedupeChecked() });
                const estimateEl = document.getElementById('review-context-estimate');
                if (estimateEl) {
                    estimateEl.innerText = result.estimated_calls
                        ? `Estimated LLM calls: ~${result.estimated_calls} for ${result.total_entries} entries with batches of ${result.batch_size}.`
                        : 'Contextual review started.';
                }
                pollLogs('review', 'script-logs', _onReviewDone);
            } catch (e) {
                _onReviewDone();
                showToast("Failed to start contextual review: " + e.message, 'error');
            }
        });

        const _reviewPauseResume = _makePauseResumeHandler(
            '/api/review_script/pause', '/api/review_script/resume', 'btn-pause-review');
        const _batchReviewPauseResume = _makePauseResumeHandler(
            '/api/review_script/batch/pause', '/api/review_script/batch/resume', 'btn-pause-batch-review');
        window.pauseResumeReview = _reviewPauseResume;
        window.pauseResumeBatchReview = _batchReviewPauseResume;
        window.cancelReview = () => cancelTask('/api/review_script/cancel', {
            onSuccess: () => _resetPauseBtn('btn-pause-review'),
        });
        window.cancelBatchReview = () => cancelTask('/api/review_script/batch/cancel', {
            onSuccess: () => _resetPauseBtn('btn-pause-batch-review'),
        });

        // --- Batch review ---
        let reviewBatchSelected = [];
        let reviewBatchScripts = [];   // current saved-script list, kept so we can re-sort without refetching

        window.toggleReviewBatchMode = () => {
            const isBatch = document.getElementById('review-batch-mode').checked;
            document.getElementById('review-single-area').style.display = isBatch ? 'none' : 'block';
            document.getElementById('review-batch-area').style.display = isBatch ? 'block' : 'none';
            document.getElementById('btn-review-batch-start').style.display = isBatch ? 'inline-block' : 'none';
            if (isBatch) { loadReviewBatchScripts(); }
        };

        // --- Shared helpers for saved-script checkbox pickers (Batch Review,
        // Cast bulk-apply) so the fetch/render/sort logic isn't duplicated. ---

        // Fetch /api/scripts into `containerId`'s picker, handing the result to
        // `onLoaded` (which should store it and call the matching render function).
        async function _loadScriptList(containerId, onLoaded) {
            const container = document.getElementById(containerId);
            try {
                const scripts = await API.get('/api/scripts');
                onLoaded(scripts);
            } catch (e) {
                container.innerHTML = `<span class="text-danger small">${escapeHtml(e.message || String(e))}</span>`;
            }
        }

        // Render `scripts` into a saved-script checkbox picker, preserving any
        // checkboxes the user already ticked (re-rendering rebuilds the inputs,
        // so capture selection first). `extra(i)` optionally renders trailing
        // per-row markup (e.g. a status indicator).
        function _renderScriptCheckboxList(items, { containerId, checkClass, idPrefix, extra,
                                                     keyField = 'name', renderLabel, onchange,
                                                     emptyMessage }) {
            const container = document.getElementById(containerId);
            if (!items.length) {
                container.innerHTML = `<span class="text-muted small">${emptyMessage ||
                    'No saved scripts found. Generate or save scripts first.'}</span>`;
                return;
            }
            const checked = new Set(
                Array.from(document.querySelectorAll(`.${checkClass}:checked`)).map(cb => cb.dataset.name)
            );
            container.innerHTML = items.map((item, i) => {
                const key = item[keyField];
                const label = renderLabel ? renderLabel(item)
                    : `${escapeHtml(item.name)} ${item.has_voice_config ? '<span class="badge bg-info ms-1">voices</span>' : ''}`;
                return `
                <div class="form-check${extra ? ' d-flex align-items-center justify-content-between' : ''}">
                    <div>
                        <input class="form-check-input ${checkClass}" type="checkbox" id="${idPrefix}${i}" data-name="${escapeHtml(key)}"${onchange ? ` onchange="${onchange}"` : ''} ${checked.has(key) ? 'checked' : ''}>
                        <label class="form-check-label small" for="${idPrefix}${i}">${label}</label>
                    </div>
                    ${extra ? extra(i) : ''}
                </div>`;
            }).join('');
        }

        // Shared by every checkbox-list picker's "Select all"/"Clear" pair
        // (script-batch-uploads, review-batch, cast-bulk). `onchange` is
        // optional - programmatically setting `.checked` doesn't fire a
        // checkbox's own inline onchange handler, so callers that need a
        // refresh after a bulk toggle (script-batch-uploads) pass one; the
        // others don't need it, same as before this helper existed.
        function _selectAllCheckboxes(checkClass, on, onchange) {
            document.querySelectorAll(`.${checkClass}`).forEach(cb => { cb.checked = on; });
            if (onchange) { onchange(); }
        }

        // Trailing number in a script name for numeric ("1→10") sorting,
        // e.g. "arc_8_-_volume_37" → 37. Names without a number sort last.
        function _getScriptVolumeNum(name) {
            const m = String(name).match(/(\d+)(?!.*\d)/);
            return m ? parseInt(m[1], 10) : Number.POSITIVE_INFINITY;
        }

        // Sort `list` (array of {name, ...}) in place per the A→Z / Z→A / 1→10 /
        // 10→1 / Reverse sort buttons shared by the saved-script pickers.
        function _sortScriptList(list, mode) {
            if (!list.length) { return; }
            const byName = (a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
            const byNum  = (a, b) => _getScriptVolumeNum(a.name) - _getScriptVolumeNum(b.name) || byName(a, b);
            switch (mode) {
                case 'az':       list.sort(byName); break;
                case 'za':       list.sort(byName).reverse(); break;
                case 'num-asc':  list.sort(byNum); break;
                case 'num-desc': list.sort(byNum).reverse(); break;
                case 'reverse':  list.reverse(); break;
            }
        }

        async function loadReviewBatchScripts() {
            await _loadScriptList('review-batch-list', (scripts) => {
                reviewBatchScripts = scripts;
                renderReviewBatchList();
            });
        }

        function renderReviewBatchList() {
            _renderScriptCheckboxList(reviewBatchScripts, {
                containerId: 'review-batch-list',
                checkClass: 'review-batch-check',
                idPrefix: 'rb-check-',
                extra: (i) => `<span class="small" id="rb-status-${i}"></span>`,
            });
        }

        window.reviewBatchSort = (mode) => {
            _sortScriptList(reviewBatchScripts, mode);
            renderReviewBatchList();
        };

        window.reviewBatchSelectAll = (on) => _selectAllCheckboxes('review-batch-check', on);

        async function startBatchReview() {
            const names = Array.from(document.querySelectorAll('.review-batch-check:checked')).map(cb => cb.dataset.name);
            if (!names.length) { showToast('Select at least one script to review.', 'warning'); return; }
            if (!(await confirmIfRemote('this batch review'))) { return; }
            reviewBatchSelected = names;
            const rawWindow = parseInt(document.getElementById('review-batch-context-window').value, 10);
            const contextWindow = Number.isFinite(rawWindow) ? Math.max(0, Math.min(rawWindow, 12)) : 0;

            const startBtn = document.getElementById('btn-review-batch-start');
            const pauseBtn = document.getElementById('btn-pause-batch-review');
            const cancelBtn = document.getElementById('btn-cancel-batch-review');
            const statusMsg = document.getElementById('review-batch-status-msg');
            startBtn.disabled = true;
            pauseBtn.style.display = 'inline-block';
            cancelBtn.style.display = 'inline-block';
            _resetPauseBtn('btn-pause-batch-review');
            statusMsg.style.display = '';
            statusMsg.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Starting batch review…</span>';

            try {
                await API.post('/api/review_script/batch/start', {
                    script_names: names,
                    context_window: contextWindow,
                    dedupe_speakers: _isReviewDedupeChecked(),
                    find_nicknames: document.getElementById('review-batch-find-nicknames').checked,
                    bidirectional: document.getElementById('review-batch-bidirectional').checked,
                });
                pollReviewBatch();
            } catch (e) {
                startBtn.disabled = false;
                pauseBtn.style.display = 'none';
                cancelBtn.style.display = 'none';
                statusMsg.innerHTML = `<span class="text-danger">${escapeHtml(e.message || String(e))}</span>`;
            }
        }

        // --- Nickname discovery + alias editor ---
        const _nickPauseResume = _makePauseResumeHandler(
            '/api/find_nicknames/pause', '/api/find_nicknames/resume', 'btn-pause-nick');
        window.pauseResumeNicknames = _nickPauseResume;
        window.cancelNicknames = () => cancelTask('/api/find_nicknames/cancel', {
            onSuccess: () => _resetPauseBtn('btn-pause-nick'),
        });

        async function findNicknames() {
            const btn = document.getElementById('btn-find-nicknames');
            btn.disabled = true;
            document.getElementById('btn-pause-nick').style.display = 'inline-block';
            document.getElementById('btn-cancel-nick').style.display = 'inline-block';
            _resetPauseBtn('btn-pause-nick');
            try {
                await API.post('/api/find_nicknames', {});
                pollLogs('nicknames', 'script-logs', async () => {
                    btn.disabled = false;
                    document.getElementById('btn-pause-nick').style.display = 'none';
                    document.getElementById('btn-cancel-nick').style.display = 'none';
                    await loadCharacterAliases(true);  // show what was found for review/edit
                });
            } catch (e) {
                btn.disabled = false;
                document.getElementById('btn-pause-nick').style.display = 'none';
                document.getElementById('btn-cancel-nick').style.display = 'none';
                showToast('Failed to start nickname discovery: ' + e.message, 'error');
            }
        }

        async function loadCharacterAliases(show) {
            const panel = document.getElementById('nickname-aliases-panel');
            let aliases = {};
            try { aliases = await API.get('/api/character_aliases'); } catch (e) { console.error('Failed to load character aliases:', e); aliases = {}; }
            const entries = Object.entries(aliases || {});
            if (show) { panel.style.display = 'block'; }
            const rowHtml = (a, c) => `
                <div class="input-group input-group-sm mb-1 nick-alias-row">
                    <input type="text" class="form-control nick-alias" placeholder="alias / nickname" value="${escapeHtml(a)}">
                    <span class="input-group-text">&rarr;</span>
                    <input type="text" class="form-control nick-canonical" placeholder="canonical name" value="${escapeHtml(c)}">
                    <button class="btn btn-outline-danger" type="button" onclick="this.closest('.nick-alias-row').remove()"><i class="fas fa-times"></i></button>
                </div>`;
            panel.innerHTML = `
                <div class="border rounded p-2">
                    <div class="small fw-bold mb-1">Character aliases <span class="text-muted">(applied automatically when you run Review)</span></div>
                    <div id="nick-alias-rows">
                        ${entries.length ? entries.map(([a, c]) => rowHtml(a, c)).join('') : '<div class="text-muted small mb-1">No aliases yet. Run "Find Nicknames" or add rows manually.</div>'}
                    </div>
                    <div class="d-flex gap-2 mt-1">
                        <button class="btn btn-sm btn-outline-secondary" type="button" onclick="addAliasRow()"><i class="fas fa-plus me-1"></i>Add</button>
                        <button class="btn btn-sm btn-success" type="button" onclick="saveCharacterAliases()"><i class="fas fa-save me-1"></i>Save aliases</button>
                    </div>
                </div>`;
        }

        window.addAliasRow = () => {
            const rows = document.getElementById('nick-alias-rows');
            const placeholder = rows.querySelector('.text-muted');
            if (placeholder) { placeholder.remove(); }
            const div = document.createElement('div');
            div.className = 'input-group input-group-sm mb-1 nick-alias-row';
            div.innerHTML = `
                <input type="text" class="form-control nick-alias" placeholder="alias / nickname">
                <span class="input-group-text">&rarr;</span>
                <input type="text" class="form-control nick-canonical" placeholder="canonical name">
                <button class="btn btn-outline-danger" type="button" onclick="this.closest('.nick-alias-row').remove()"><i class="fas fa-times"></i></button>`;
            rows.appendChild(div);
        };

        async function saveCharacterAliases() {
            const map = {};
            document.querySelectorAll('#nick-alias-rows .nick-alias-row').forEach(row => {
                const a = row.querySelector('.nick-alias').value.trim();
                const c = row.querySelector('.nick-canonical').value.trim();
                if (a && c) { map[a] = c; }
            });
            try {
                const res = await API.post('/api/character_aliases', map);
                showToast(`Saved ${res.count} alias${res.count !== 1 ? 'es' : ''}. Run Review to apply.`, 'success');
            } catch (e) {
                showToast('Save failed: ' + (e.message || ''), 'error');
            }
        }

        // One-line "N changes: X text, Y speaker, ..." breakdown for a finished book's badge tooltip.
        function _formatBookStats(s) {
            let txt = s.partial ? '(partial — not every pass completed) ' : '';
            txt += `${s.total_changes} changes: ${s.text_changed} text, ${s.speaker_changed} speaker, ` +
                      `${s.instruct_changed} instruct, +${s.entries_added}/-${s.entries_removed} entries`;
            if (s.narrators_merged) { txt += `, ${s.narrators_merged} narrators merged`; }
            if (s.speakers_merged) { txt += `, ${s.speakers_merged} speakers merged`; }
            if (s.batches_failed) { txt += `, ${s.batches_failed} batch(es) failed`; }
            if (s.batches_skipped_vram) { txt += `, ${s.batches_skipped_vram} batch(es) skipped (VRAM)`; }
            return txt;
        }

        function _formatTotalsLine(label, t) {
            if (!t || !t.books_done) { return `${label}: no books finished yet`; }
            let txt = `${label}: ${t.books_done} book(s), ${t.total_changes} total change(s) ` +
                      `(${t.text_changed} text, ${t.speaker_changed} speaker, ${t.instruct_changed} instruct, ` +
                      `+${t.entries_added}/-${t.entries_removed} entries)`;
            if (t.batches_failed) { txt += ` — ${t.batches_failed} batch(es) failed`; }
            return txt;
        }

        function _updateReviewBatchTotals(state) {
            const el = document.getElementById('review-batch-totals');
            if (!el) { return; }
            const fwd = state.totals_fwd, bwd = state.totals_bwd;
            const aliasesBwd = state.aliases_bwd || [];
            if (!(fwd && fwd.books_done) && !(bwd && bwd.books_done) && !aliasesBwd.length) {
                el.style.display = 'none';
                return;
            }
            let html = '';
            if (state.bidirectional) {
                html += `<div>${escapeHtml(_formatTotalsLine('Forward pass', fwd))}</div>`;
                html += `<div>${escapeHtml(_formatTotalsLine('Backward pass (hindsight)', bwd))}</div>`;
            } else {
                html += `<div>${escapeHtml(_formatTotalsLine('Totals', fwd))}</div>`;
            }
            if (aliasesBwd.length) {
                html += `<div class="mt-1"><strong>New characters found on the backward pass (${aliasesBwd.length}):</strong></div>`;
                html += '<ul class="mb-0 ps-3">' + aliasesBwd.map(a =>
                    `<li>'${escapeHtml(a.variant)}' &rarr; '${escapeHtml(a.canonical)}' <span class="text-muted">(${escapeHtml(a.book)})</span></li>`
                ).join('') + '</ul>';
            }
            el.innerHTML = html;
            el.style.display = '';
        }

        function pollReviewBatch() {
            const logEl = document.getElementById('script-logs');
            _startPolling('batch_review', () => API.get('/api/status/batch_review'), {
                doneCheck: state => !state.running,
                onTick: state => {
                    if (logEl) { logEl.innerText = (state.logs || []).join('\n'); logEl.scrollTop = logEl.scrollHeight; }
                    const colours = { pending: 'secondary', running: 'primary', done: 'success', incomplete: 'warning', failed: 'danger', cancelled: 'warning' };
                    (state.tasks || []).forEach(t => {
                        // Map by name (the list shows all scripts; only selected ones are tasks)
                        const cb = document.querySelector(`.review-batch-check[data-name="${CSS.escape(t.name)}"]`);
                        if (!cb) { return; }
                        const el = document.getElementById(cb.id.replace('rb-check-', 'rb-status-'));
                        if (el) {
                            el.innerHTML = `<span class="badge bg-${colours[t.status] || 'secondary'}">${t.status}</span>`;
                            el.title = t.stats ? _formatBookStats(t.stats) : '';
                        }
                    });
                    _updateReviewBatchTotals(state);
                },
                onDone: () => {
                    notifyJobDone('batch_review');
                    document.getElementById('btn-review-batch-start').disabled = false;
                    document.getElementById('btn-pause-batch-review').style.display = 'none';
                    document.getElementById('btn-cancel-batch-review').style.display = 'none';
                    document.getElementById('review-batch-status-msg').innerHTML =
                        '<span class="text-muted">Batch review complete.</span>';
                }
            });
        }

        // --- Persona Generation ---
        function toggleAdvancedPersonaOptions() {
            const advanced = document.getElementById('advanced-persona-toggle');
            const options = document.getElementById('advanced-persona-options');
            if (advanced && options) {
                options.style.display = advanced.checked ? 'flex' : 'none';
            }
        }

        async function generatePersonas() {
            const statusSpan = document.getElementById('persona-status');
            const cancelButton = document.getElementById('btn-cancel-personas');
            const advancedToggle = document.getElementById('advanced-persona-toggle');
            const batchInput = document.getElementById('persona-batch-size');
            const advanced = !!(advancedToggle && advancedToggle.checked);
            const batchSize = Math.max(1, Math.min(parseInt(batchInput?.value || '40', 10) || 40, 200));
            try {
                statusSpan.innerHTML = `<i class="fas fa-spinner fa-spin me-1"></i>${advanced ? 'Starting advanced...' : 'Starting...'}`;
                if (cancelButton) {
                    cancelButton.style.display = '';
                }
                await API.post('/api/generate_personas', { advanced, batch_size: batchSize });
                pollPersonaStatus();
            } catch (e) {
                showToast('Failed to start persona generation: ' + e.message, 'error');
                statusSpan.innerText = '';
                if (cancelButton) {
                    cancelButton.style.display = 'none';
                }
            }
        }

        async function cancelPersonas() {
            await cancelTask('/api/cancel_persona', {
                onSuccess: () => {
                    const statusSpan = document.getElementById('persona-status');
                    if (statusSpan) { statusSpan.innerText = 'Cancelling...'; }
                },
                errorMessage: (e) => 'Failed to cancel persona generation: ' + e.message,
                toastType: 'error',
            });
        }

        async function pollPersonaStatus() {
            const logEl = document.getElementById('voices-logs');
            const statusSpan = document.getElementById('persona-status');
            const cancelButton = document.getElementById('btn-cancel-personas');
            _startPolling('persona', () => API.get('/api/status/persona'), {
                intervalMs: 1500,
                doneCheck: status => !status.running,
                onTick: status => {
                    const advanced = !!(document.getElementById('advanced-persona-toggle')?.checked);
                    statusSpan.innerText = status.running ? (advanced ? 'Advanced running...' : 'Running...') : 'Finished';
                    if (cancelButton) {
                        cancelButton.style.display = status.running ? '' : 'none';
                    }
                    if (logEl) {
                        logEl.innerText = (status.logs || []).join('\n');
                        logEl.scrollTop = logEl.scrollHeight;
                    }
                },
                onDone: async () => {
                    // Refresh voices and caches
                    try { await loadVoices(); } catch (e) { console.debug('voices refresh failed', e); }
                    try { window._designedVoicesCache = await API.get('/api/voice_design/list'); } catch (e) { console.debug('designed-voices cache prefetch failed', e); }
                    try { window._cloneVoicesCache = await API.get('/api/clone_voices/list'); } catch (e) { console.debug('clone-voices cache prefetch failed', e); }
                    showToast('Persona generation finished', 'success');
                    statusSpan.innerText = '';
                    if (cancelButton) {
                        cancelButton.style.display = 'none';
                    }
                }
            });
        }

        // --- Voices Tab ---
        const AVAILABLE_VOICES = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"];

        function createVoiceCard(voice, index) {
            const config = voice.config || {};
            const voiceType = config.type || 'custom';

            return `
                <div class="card voice-card mb-3" data-voice="${escapeHtml(voice.name)}">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <h5 class="card-title">${escapeHtml(voice.name)} ${config.alias_of ? `<span class="badge bg-info ms-2" title="Alias of ${escapeHtml(config.alias_of)}">${escapeHtml(config.alias_of)}</span>` : ''}${(window._lineCounts && window._lineCounts[voice.name] != null) ? `<span class="badge bg-secondary ms-2" title="${window._lineCounts[voice.name]} lines in this book">${window._lineCounts[voice.name]} lines</span>` : ''}</h5>
                                <div class="form-text small text-muted mt-1">Alias of:</div>
                                <select class="form-select form-select-sm alias-select mt-1">
                                    <option value="">-- None --</option>
                                    ${(() => {
                                        const names = (window._voicesNames || []).filter(n => n !== voice.name);
                                        return names.map(n => `<option value="${escapeHtml(n)}" ${config.alias_of === n ? 'selected' : ''}>${escapeHtml(n)}</option>`).join('');
                                    })()}
                                </select>
                            </div>
                            <div class="col-md-9">
                                <div class="mb-2">
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="custom" ${voiceType === 'custom' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Custom Voice</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="builtin_lora" ${voiceType === 'builtin_lora' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Built-in Voice</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="clone" ${voiceType === 'clone' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Voice Clone</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="lora" ${voiceType === 'lora' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">LoRA Voice</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="design" ${voiceType === 'design' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Voice Design</label>
                                    </div>
                                    <div class="form-check form-check-inline">
                                        <input class="form-check-input voice-type" type="radio" name="type_${index}" value="ensemble" ${voiceType === 'ensemble' ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                        <label class="form-check-label">Together</label>
                                    </div>
                                </div>

                                <!-- Custom Options -->
                                <div class="custom-opts" style="display: ${voiceType === 'custom' ? 'block' : 'none'}">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <select class="form-select voice-select">
                                                ${AVAILABLE_VOICES.map(v => `<option value="${v}" ${config.voice === v ? 'selected' : ''}>${v}</option>`).join('')}
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <input type="text" class="form-control character-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${escapeHtml(config.character_style || config.default_style || '')}">
                                        </div>
                                    </div>
                                </div>

                                <!-- Built-in LoRA Options -->
                                <div class="builtin-lora-opts" style="display: ${voiceType === 'builtin_lora' ? 'block' : 'none'}">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <select class="form-select builtin-lora-select">
                                                <option value="">-- Select built-in voice --</option>
                                                ${(() => {
                                                    const models = (window._loraModelsCache || []).filter(m => m.builtin);
                                                    const males = models.filter(m => m.gender === 'male');
                                                    const females = models.filter(m => m.gender === 'female');
                                                    let html = '';
                                                    if (males.length) {
                                                        html += '<optgroup label="Male">';
                                                        html += males.map(m => `<option value="${escapeHtml(m.id)}" ${config.adapter_id === m.id ? 'selected' : ''} ${m.downloaded === false ? 'disabled' : ''}>${escapeHtml(m.name)}${m.downloaded === false ? ' (not downloaded)' : ''} — ${escapeHtml(m.description || '')}</option>`).join('');
                                                        html += '</optgroup>';
                                                    }
                                                    if (females.length) {
                                                        html += '<optgroup label="Female">';
                                                        html += females.map(m => `<option value="${escapeHtml(m.id)}" ${config.adapter_id === m.id ? 'selected' : ''} ${m.downloaded === false ? 'disabled' : ''}>${escapeHtml(m.name)}${m.downloaded === false ? ' (not downloaded)' : ''} — ${escapeHtml(m.description || '')}</option>`).join('');
                                                        html += '</optgroup>';
                                                    }
                                                    return html;
                                                })()}
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <input type="text" class="form-control builtin-lora-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${escapeHtml(voiceType === 'builtin_lora' ? (config.character_style || '') : '')}">
                                        </div>
                                    </div>
                                    <small class="text-muted mt-1 d-block">Grayed-out voices need to be downloaded first. Go to the <strong>Training</strong> tab to download them.</small>
                                </div>

                                <!-- Clone Options -->
                                <div class="clone-opts" style="display: ${voiceType === 'clone' ? 'block' : 'none'}">
                                    <div class="row g-2 mb-2 align-items-center">
                                        <div class="col">
                                            <select class="form-select designed-voice-select" onchange="onDesignedVoiceSelect(this)">
                                                <option value="">-- Select voice or enter path manually --</option>
                                                ${(window._cloneVoicesCache || []).length ? `<optgroup label="Uploaded Voices">
                                                    ${(window._cloneVoicesCache || []).map(v => `<option value="clone:${escapeHtml(v.id)}" ${config.ref_audio && config.ref_audio.includes(v.filename) ? 'selected' : ''}>${escapeHtml(v.name)}</option>`).join('')}
                                                </optgroup>` : ''}
                                                ${(window._designedVoicesCache || []).length ? `<optgroup label="Designed Voices">
                                                    ${(window._designedVoicesCache || []).map(v => `<option value="design:${escapeHtml(v.id)}" ${config.ref_audio && config.ref_audio.includes(v.filename) ? 'selected' : ''}>${escapeHtml(v.name)}</option>`).join('')}
                                                </optgroup>` : ''}
                                                <option value="__manual__" ${config.ref_audio && !(window._cloneVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) && !(window._designedVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) && config.ref_audio ? 'selected' : ''}>Custom path...</option>
                                            </select>
                                        </div>
                                        <div class="col-auto">
                                            <button class="btn btn-sm btn-outline-primary" onclick="uploadCloneVoice(this)" title="Upload audio file"><i class="fas fa-upload"></i> Upload</button>
                                            <input type="file" class="clone-voice-file-input" accept=".wav,.mp3,.flac,.ogg" style="display:none" onchange="handleCloneVoiceUpload(this)">
                                        </div>
                                    </div>
                                    <input type="text" class="form-control ref-text mb-2" placeholder="Reference Text" value="${escapeHtml(config.ref_text || '')}">
                                    <div class="input-group">
                                        <input type="text" class="form-control ref-audio" placeholder="Path to audio file" value="${escapeHtml(config.ref_audio || '')}" ${config.ref_audio && ((window._cloneVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) || (window._designedVoicesCache || []).some(v => config.ref_audio.includes(v.filename))) ? 'readonly' : ''}>
                                        <button class="btn btn-sm btn-outline-secondary clone-play-btn" onclick="playCloneVoice(this)" title="Play reference audio" style="display:${config.ref_audio ? 'inline-block' : 'none'}"><i class="fas fa-play"></i></button>
                                        <button class="btn btn-sm btn-outline-danger clone-delete-btn" onclick="deleteCloneVoice(this)" title="Delete uploaded voice" style="display:${config.ref_audio && (window._cloneVoicesCache || []).some(v => config.ref_audio.includes(v.filename)) ? 'inline-block' : 'none'}"><i class="fas fa-trash"></i></button>
                                    </div>
                                </div>

                                <!-- LoRA Options -->
                                <div class="lora-opts" style="display: ${voiceType === 'lora' ? 'block' : 'none'}">
                                    <div class="row g-2">
                                        <div class="col-md-6">
                                            <select class="form-select lora-adapter-select">
                                                <option value="">-- Select trained adapter --</option>
                                                ${(window._loraModelsCache || []).map(m => `<option value="${escapeHtml(m.id)}" ${config.adapter_id === m.id ? 'selected' : ''}>${escapeHtml(m.name)}</option>`).join('')}
                                            </select>
                                        </div>
                                        <div class="col-md-6">
                                            <input type="text" class="form-control lora-character-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${escapeHtml(voiceType === 'lora' ? (config.character_style || '') : '')}">
                                        </div>
                                    </div>
                                </div>

                                <!-- Voice Design Options -->
                                <div class="design-opts" style="display: ${voiceType === 'design' ? 'block' : 'none'}">
                                    <input type="text" class="form-control design-description mb-1" placeholder="Base voice description (e.g. Young strong soldier)" value="${escapeHtml(config.description || '')}">
                                    <span class="text-muted small">Per-line instruct is appended to this description as delivery/emotion direction</span>
                                    <div class="mt-2">
                                        <button type="button" class="btn btn-sm btn-outline-primary" onclick="openVoiceDesignEditor(this)">
                                            <i class="fas fa-wand-magic-sparkles me-1"></i>Re-design Voice
                                        </button>
                                    </div>
                                </div>

                                <!-- Ensemble Options -->
                                <div class="ensemble-opts" style="display: ${voiceType === 'ensemble' ? 'block' : 'none'}">
                                    <div class="ensemble-members small">${ensembleMembersMarkup(voice.name, config.members)}</div>
                                    <span class="text-muted small">Each character is voiced with whatever voice it already has, then mixed together. Clips are aligned to the longest, so this sounds like a chorus rather than exact unison.</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Suggest members for a compound name like "Petra and Subaru" or
        // "GOD/BUDDHA/OD LAGNA": keep the parts that are real characters.
        function suggestEnsembleMembers(name) {
            const names = window._voicesNames || [];
            const lower = new Map(names.map(n => [n.toLowerCase(), n]));
            return name.split(/\s+&\s+|\s+and\s+|\/|\s*\+\s*/i)
                .map(part => lower.get(part.trim().toLowerCase()))
                .filter(n => n && n !== name);
        }

        function ensembleMembersMarkup(name, members) {
            const names = (window._voicesNames || []).filter(n => n !== name);
            if (names.length === 0) {
                return '<div class="text-muted">No other characters to combine.</div>';
            }
            // Only prefill when nothing is saved yet — never override a choice.
            const selected = new Set(members && members.length ? members : suggestEnsembleMembers(name));
            return names.map(n => `
                <div class="form-check form-check-inline">
                    <input class="form-check-input ensemble-member" type="checkbox" value="${escapeHtml(n)}" ${selected.has(n) ? 'checked' : ''} onchange="saveVoicesDebounced()">
                    <label class="form-check-label">${escapeHtml(n)}</label>
                </div>
            `).join('');
        }

        window.toggleVoiceType = (radio) => {
            const card = radio.closest('.card-body');
            card.querySelector('.custom-opts').style.display = radio.value === 'custom' ? 'block' : 'none';
            card.querySelector('.builtin-lora-opts').style.display = radio.value === 'builtin_lora' ? 'block' : 'none';
            card.querySelector('.clone-opts').style.display = radio.value === 'clone' ? 'block' : 'none';
            card.querySelector('.lora-opts').style.display = radio.value === 'lora' ? 'block' : 'none';
            card.querySelector('.design-opts').style.display = radio.value === 'design' ? 'block' : 'none';
            card.querySelector('.ensemble-opts').style.display = radio.value === 'ensemble' ? 'block' : 'none';
            saveVoicesDebounced();
        };

        async function loadVoices() {
            // Refresh voice caches so dropdowns are populated
            try {
                window._designedVoicesCache = await API.get('/api/voice_design/list');
            } catch (e) { console.debug('designed-voices cache refresh failed', e); }
            try {
                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
            } catch (e) { console.debug('clone-voices cache refresh failed', e); }
            try {
                window._loraModelsCache = await API.get('/api/lora/models');
            } catch (e) { console.debug('lora-models cache refresh failed', e); }

            // Load the series cast library (also gives us per-character line counts for badges)
            try {
                await loadCastLibrary();
            } catch (e) { console.debug('cast library refresh failed', e); }

            const voices = await API.get('/api/voices');
            // Cache simple names for alias dropdowns
            window._voicesNames = voices.map(v => v.name);
            const container = document.getElementById('voices-list');
            if (voices.length === 0) {
                container.innerHTML = '<div class="alert alert-info">No voices found. Generate a script first.</div>';
                return;
            }
            container.innerHTML = voices.map((v, i) => createVoiceCard(v, i)).join('');

            // If any voice has no saved config, save defaults immediately
            if (voices.some(v => !v.config || Object.keys(v.config).length === 0)) {
                saveVoicesDebounced();
            }

            // Restore any pending voice suggestions onto the freshly rendered cards
            if (window._voiceSuggestions && Object.keys(window._voiceSuggestions).length) {
                renderVoiceSuggestions();
            }
        }

        // --- Auto-suggest best LoRA voice per character ---
        window._voiceSuggestions = {};

        async function suggestVoices() {
            const btn = document.getElementById('btn-suggest-voices');
            const status = document.getElementById('suggest-status');
            btn.disabled = true;
            status.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Analyzing characters and matching voices...';
            try {
                // Make sure lora caches are fresh so we can resolve suggested adapters in the dropdowns
                try { window._loraModelsCache = await API.get('/api/lora/models'); } catch (e) { console.debug('lora-models cache refresh failed', e); }
                const res = await API.post('/api/suggest_voices', {
                    only_unset: false,
                    cast: window._selectedCast || null,
                });
                window._voiceSuggestions = res.suggestions || {};
                const n = Object.keys(window._voiceSuggestions).length;
                if (n === 0) {
                    status.textContent = res.message || 'No suggestions available.';
                    document.getElementById('btn-apply-all-suggestions').style.display = 'none';
                    document.getElementById('btn-clear-suggestions').style.display = 'none';
                } else {
                    const methodLabel = res.method === 'llm' ? 'LLM' : 'heuristic';
                    status.innerHTML = `<i class="fas fa-check text-success me-1"></i>Suggested ${n} voice${n > 1 ? 's' : ''} (${methodLabel}). Review and apply below.`;
                    document.getElementById('btn-apply-all-suggestions').style.display = 'inline-block';
                    document.getElementById('btn-clear-suggestions').style.display = 'inline-block';
                }
                if (res.llm_warning) {
                    status.innerHTML += `<div class="text-warning small mt-1"><i class="fas fa-exclamation-triangle me-1"></i>${escapeHtml(res.llm_warning)}</div>`;
                }
                renderVoiceSuggestions();
            } catch (e) {
                status.innerHTML = `<i class="fas fa-times text-danger me-1"></i>${escapeHtml(e.message || String(e))}`;
            } finally {
                btn.disabled = false;
            }
        }

        function renderVoiceSuggestions() {
            document.querySelectorAll('.voice-card').forEach(card => {
                const name = card.dataset.voice;
                const body = card.querySelector('.card-body');
                let banner = card.querySelector('.voice-suggestion');
                const sugg = window._voiceSuggestions[name];
                if (!sugg) {
                    if (banner) { banner.remove(); }
                    return;
                }
                if (!banner) {
                    banner = document.createElement('div');
                    banner.className = 'voice-suggestion alert alert-success d-flex align-items-center justify-content-between py-2 px-3 mt-2 mb-0';
                    body.appendChild(banner);
                }
                const typeLabel = sugg.type === 'builtin_lora' ? 'Built-in' : 'LoRA';
                const reuseText = sugg.reused
                    ? `Reused by ${sugg.reuse_count_before} existing cast character${sugg.reuse_count_before !== 1 ? 's' : ''}`
                    : 'Unused in this series cast';
                const traitText = `${(sugg.character_gender || 'unknown').replace('_', ' ')} · ${(sugg.character_age_group || 'unknown').replace('_', ' ')} → ${(sugg.voice_gender || 'unknown').replace('_', ' ')} · ${(sugg.voice_age_group || 'unknown').replace('_', ' ')}`;
                const traitWarning = sugg.gender_fallback
                    ? 'No downloaded voice matched the known gender; fallback used.'
                    : (sugg.existing_trait_mismatch ? 'Existing recurring voice retained despite a trait mismatch.' : '');
                banner.innerHTML = `
                    <div class="me-2 small">
                        <i class="fas fa-wand-magic-sparkles me-1"></i>
                        <span class="badge ${sugg.priority === 'major' ? 'bg-primary' : 'bg-secondary'} me-1">${escapeHtml((sugg.priority || 'minor').toUpperCase())}</span>
                        <strong>${sugg.line_count || 0} lines · Suggested ${typeLabel} voice:</strong> ${escapeHtml(sugg.adapter_name)}
                        <span class="d-block"><strong>Style:</strong> ${escapeHtml(sugg.character_style || '')}</span>
                        <span class="d-block"><strong>Trait match:</strong> ${escapeHtml(traitText)} <span class="text-muted">(gender ${escapeHtml(sugg.gender_confidence || 'unknown')}, age ${escapeHtml(sugg.age_confidence || 'unknown')})</span></span>
                        ${sugg.trait_evidence ? `<span class="text-muted d-block">${escapeHtml(sugg.trait_evidence)}</span>` : ''}
                        ${traitWarning ? `<span class="text-warning d-block">${escapeHtml(traitWarning)}</span>` : ''}
                        <span class="d-block ${sugg.reused ? 'text-warning' : 'text-success'}">${escapeHtml(reuseText)}${sugg.forced_reuse ? ' (candidate pool exhausted)' : ''}</span>
                        ${sugg.reason ? `<span class="text-muted d-block">${escapeHtml(sugg.reason)}</span>` : ''}
                    </div>
                    <button class="btn btn-sm btn-success flex-shrink-0" data-voice="${escapeHtml(name)}" onclick="applyVoiceSuggestion(this.dataset.voice)"><i class="fas fa-check me-1"></i>Apply</button>
                `;
            });
            const container = document.getElementById('voices-list');
            Array.from(container.querySelectorAll('.voice-card'))
                .sort((a, b) => (window._voiceSuggestions[b.dataset.voice]?.line_count || window._lineCounts[b.dataset.voice] || 0)
                               - (window._voiceSuggestions[a.dataset.voice]?.line_count || window._lineCounts[a.dataset.voice] || 0))
                .forEach(card => container.appendChild(card));
        }

        function applySuggestionToCard(name) {
            const sugg = window._voiceSuggestions[name];
            if (!sugg) { return false; }
            const card = document.querySelector(`.voice-card[data-voice="${CSS.escape(name)}"]`);
            if (!card) { return false; }
            const body = card.querySelector('.card-body');

            // Select the correct voice type radio and reveal its options
            const radio = body.querySelector(`.voice-type[value="${sugg.type}"]`);
            if (radio) {
                radio.checked = true;
                toggleVoiceType(radio);
            }

            // Set the suggested adapter in the matching dropdown
            const selectClass = sugg.type === 'builtin_lora' ? '.builtin-lora-select' : '.lora-adapter-select';
            const select = body.querySelector(selectClass);
            if (select) {
                select.value = sugg.adapter_id;
                // If the option isn't present (cache mismatch), add it so the value sticks
                if (select.value !== sugg.adapter_id) {
                    const opt = new Option(sugg.adapter_name, sugg.adapter_id, true, true);
                    select.add(opt);
                    select.value = sugg.adapter_id;
                }
            }

            const styleInput = body.querySelector(sugg.type === 'builtin_lora'
                ? '.builtin-lora-style' : '.lora-character-style');
            if (styleInput) { styleInput.value = sugg.character_style || ''; }

            // Remove the banner and clear from pending suggestions
            const banner = card.querySelector('.voice-suggestion');
            if (banner) { banner.remove(); }
            delete window._voiceSuggestions[name];
            updateSuggestionToolbar();
            return true;
        }

        async function applyVoiceSuggestion(name) {
            const sugg = window._voiceSuggestions[name];
            if (!sugg) { return; }
            try {
                await API.post('/api/suggest_voices/apply', {
                    character: name, cast: window._selectedCast || null, suggestion: sugg,
                });
                applySuggestionToCard(name);
                await loadCastLibrary();
            } catch (e) {
                showToast('Failed to apply suggestion: ' + (e.message || String(e)), 'error');
            }
        }

        async function applyAllVoiceSuggestions() {
            const pending = { ...window._voiceSuggestions };
            if (!Object.keys(pending).length) { return; }
            try {
                await API.post('/api/suggest_voices/apply_bulk', {
                    cast: window._selectedCast || null, suggestions: pending,
                });
                Object.keys(pending).forEach(name => applySuggestionToCard(name));
                await loadCastLibrary();
            } catch (e) {
                showToast('Failed to apply suggestions: ' + (e.message || String(e)), 'error');
            }
        }

        function clearVoiceSuggestions() {
            window._voiceSuggestions = {};
            document.querySelectorAll('.voice-suggestion').forEach(b => b.remove());
            updateSuggestionToolbar();
            document.getElementById('suggest-status').textContent = '';
        }

        function updateSuggestionToolbar() {
            const remaining = Object.keys(window._voiceSuggestions).length;
            const show = remaining > 0 ? 'inline-block' : 'none';
            document.getElementById('btn-apply-all-suggestions').style.display = show;
            document.getElementById('btn-clear-suggestions').style.display = show;
            if (remaining === 0) {
                document.getElementById('suggest-status').innerHTML = '<i class="fas fa-check text-success me-1"></i>All suggestions applied.';
            }
        }

        // --- Series Cast: reuse character voices across books ---
        window._voiceLibrary = { casts: [], shared: [], current_characters: [] };
        window._lineCounts = {};
        window._selectedCast = window._selectedCast || '';
        let castBulkScripts = [];   // saved-script list for the "apply to multiple books" picker

        function setCastStatus(html, isError) {
            const el = document.getElementById('cast-status');
            if (el) { el.innerHTML = isError ? `<i class="fas fa-times text-danger me-1"></i>${html}` : html; }
        }

        async function loadCastLibrary() {
            const lib = await API.get('/api/voice_library');
            window._voiceLibrary = lib;
            window._lineCounts = {};
            (lib.current_characters || []).forEach(c => { window._lineCounts[c.name] = c.line_count; });

            // Populate the cast selector, preserving the current selection if still valid
            const sel = document.getElementById('cast-select');
            const castNames = (lib.casts || []).map(c => c.name);
            if (!castNames.includes(window._selectedCast)) {
                window._selectedCast = castNames[0] || '';
            }
            sel.innerHTML = castNames.length
                ? castNames.map(n => `<option value="${escapeHtml(n)}" ${n === window._selectedCast ? 'selected' : ''}>${escapeHtml(n)}</option>`).join('')
                : '<option value="">(no casts yet)</option>';

            const hasCast = !!window._selectedCast;
            document.getElementById('btn-cast-save').disabled = !hasCast;
            document.getElementById('btn-cast-apply').disabled = !hasCast;
            document.getElementById('btn-cast-apply-bulk').disabled = !hasCast;
            document.getElementById('btn-cast-delete').disabled = !hasCast;
            renderCastMembers();
        }

        function getSelectedCastObj() {
            return (window._voiceLibrary.casts || []).find(c => c.name === window._selectedCast) || null;
        }

        function onCastChange() {
            window._selectedCast = document.getElementById('cast-select').value;
            clearVoiceSuggestions();
            renderCastMembers();
        }

        function renderCastMembers() {
            const panel = document.getElementById('cast-panel');
            const cast = getSelectedCastObj();
            const shared = window._voiceLibrary.shared || [];
            if (!window._selectedCast) {
                panel.innerHTML = '<div class="alert alert-light border small mb-0">No casts yet. Click <strong>New</strong> to create one for this series, then save your configured character voices to it.</div>';
                return;
            }
            const memberRow = (m, castName) => `
                <li class="list-group-item d-flex justify-content-between align-items-center py-1 px-2">
                    <span class="small">${escapeHtml(m.name)}${m.generic && m.book_id ? ` — ${escapeHtml(m.book_id)}` : ''}
                        <span class="text-muted">(${escapeHtml(m.type || 'custom')}${m.line_count ? ', ' + m.line_count + ' lines' : ''})</span>
                        ${m.character_style ? `<span class="d-block text-muted">${escapeHtml(m.character_style)}</span>` : ''}</span>
                    <button class="btn btn-sm btn-link text-danger p-0" title="Remove from library" data-cast="${escapeHtml(castName)}" data-key="${escapeHtml(m.key)}" onclick="deleteCastMember(this.dataset.cast, this.dataset.key)"><i class="fas fa-times"></i></button>
                </li>`;
            const members = (cast && cast.members) || [];
            const usageRows = Object.entries((cast && cast.adapter_usage) || {}).sort((a, b) => b[1].character_count - a[1].character_count);
            panel.innerHTML = `
                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="small fw-bold mb-1">Cast members <span class="text-muted">(${members.length})</span></div>
                        <ul class="list-group list-group-flush border rounded">${members.length ? members.map(m => memberRow(m, window._selectedCast)).join('') : '<li class="list-group-item small text-muted py-2 px-2">None yet — use "Save to cast".</li>'}</ul>
                    </div>
                    <div class="col-md-6">
                        <div class="small fw-bold mb-1">Shared across series <span class="text-muted">(${shared.length})</span></div>
                        <ul class="list-group list-group-flush border rounded">${shared.length ? shared.map(m => memberRow(m, '__shared__')).join('') : '<li class="list-group-item small text-muted py-2 px-2">None yet — the narrator lands here automatically.</li>'}</ul>
                    </div>
                </div>`;
            if (usageRows.length) {
                panel.innerHTML += `<div class="small fw-bold mt-2">LoRA reuse in this series</div>
                    <ul class="list-group list-group-flush border rounded">${usageRows.map(([id, u]) =>
                        `<li class="list-group-item py-1 px-2 small"><strong>${escapeHtml(id)}</strong> — ${u.character_count} character${u.character_count !== 1 ? 's' : ''}, ${u.total_lines} lines<br><span class="text-muted">${escapeHtml((u.characters || []).join(', '))}</span></li>`
                    ).join('')}</ul>`;
            }
        }

        async function createCast() {
            const name = (prompt('Name this cast (e.g. the series title):') || '').trim();
            if (!name) { return; }
            try {
                await API.post('/api/voice_library/casts', { name });
                window._selectedCast = name;
                await loadCastLibrary();
                setCastStatus(`<i class="fas fa-check text-success me-1"></i>Created cast "${escapeHtml(name)}"`);
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); }
        }

        async function deleteCast() {
            if (!window._selectedCast) { return; }
            if (!confirm(`Delete cast "${window._selectedCast}"? Shared characters (e.g. narrator) are kept.`)) { return; }
            try {
                await API.del(`/api/voice_library/casts/${encodeURIComponent(window._selectedCast)}`);
                window._selectedCast = '';
                await loadCastLibrary();
                setCastStatus('<i class="fas fa-check text-success me-1"></i>Cast deleted');
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); }
        }

        async function deleteCastMember(cast, key) {
            try {
                await API.del(`/api/voice_library/casts/${encodeURIComponent(cast)}/members/${encodeURIComponent(key)}`);
                await loadCastLibrary();
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); }
        }

        // Save current-book characters into the selected cast
        function openCastSave() {
            if (!window._selectedCast) { return; }
            const panel = document.getElementById('cast-panel');
            const chars = window._voiceLibrary.current_characters || [];
            if (!chars.length) {
                panel.innerHTML = '<div class="alert alert-warning small mb-0">No characters in the current book. Generate a script first.</div>';
                return;
            }
            const defaultThreshold = 25;
            const row = c => {
                const isNarrator = c.name.trim().toLowerCase() === 'narrator';
                const checked = isNarrator || c.line_count >= defaultThreshold ? 'checked' : '';
                const scopeCell = isNarrator
                    ? `<select class="form-select form-select-sm cast-save-scope" data-name="${escapeHtml(c.name)}" style="width:auto;">
                           <option value="shared" selected>shared (whole series)</option>
                           <option value="cast">this cast only (different narrator)</option>
                       </select>`
                    : '';
                return `<tr>
                    <td><input type="checkbox" class="cast-save-check" data-name="${escapeHtml(c.name)}" ${checked}></td>
                    <td class="small">${escapeHtml(c.name)}</td>
                    <td class="small text-muted">${c.line_count}</td>
                    <td>${scopeCell}</td>
                </tr>`;
            };
            panel.innerHTML = `
                <div class="border rounded p-2">
                    <div class="d-flex align-items-center gap-2 mb-2 flex-wrap">
                        <span class="small fw-bold">Save to "${escapeHtml(window._selectedCast)}"</span>
                        <span class="small text-muted">Pre-select characters with ≥</span>
                        <input type="number" id="cast-save-threshold" class="form-control form-control-sm" style="width:70px;" value="${defaultThreshold}" min="0" onchange="reapplyCastSaveThreshold()">
                        <span class="small text-muted">lines</span>
                    </div>
                    <div class="alert alert-light border py-1 px-2 small mb-2"><i class="fas fa-info-circle me-1"></i>Only characters you've already configured will be saved. The narrator is stored as a shared voice for the whole series.</div>
                    <div class="table-responsive" style="max-height:260px;overflow-y:auto;">
                        <table class="table table-sm table-hover mb-0">
                            <thead class="table-light"><tr><th style="width:30px;"></th><th>Character</th><th>Lines</th><th>Narrator scope</th></tr></thead>
                            <tbody>${chars.map(row).join('')}</tbody>
                        </table>
                    </div>
                    <div class="d-flex gap-2 mt-2">
                        <button class="btn btn-sm btn-success" onclick="submitCastSave()"><i class="fas fa-floppy-disk me-1"></i>Save selected</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="renderCastMembers()">Cancel</button>
                    </div>
                </div>`;
        }

        function reapplyCastSaveThreshold() {
            const t = parseInt(document.getElementById('cast-save-threshold').value, 10) || 0;
            document.querySelectorAll('.cast-save-check').forEach(cb => {
                const name = cb.dataset.name;
                const isNarrator = name.trim().toLowerCase() === 'narrator';
                const count = window._lineCounts[name] || 0;
                cb.checked = isNarrator || count >= t;
            });
        }

        async function submitCastSave() {
            const checks = Array.from(document.querySelectorAll('.cast-save-check:checked'));
            const characters = checks.map(cb => cb.dataset.name);
            if (!characters.length) { setCastStatus('Select at least one character.', true); return; }
            // Narrators marked "this cast only" are saved as a series-specific (different) narrator
            const cast_specific = Array.from(document.querySelectorAll('.cast-save-scope'))
                .filter(s => s.value === 'cast' && characters.includes(s.dataset.name))
                .map(s => s.dataset.name);
            try {
                const res = await API.post('/api/voice_library/save', { cast: window._selectedCast, characters, cast_specific });
                await loadCastLibrary();
                const nc = res.saved.cast.length, ns = res.saved.shared.length;
                const skipped = characters.length - nc - ns;
                let msg = `Saved ${nc} to cast`;
                if (ns) { msg += `, ${ns} shared`; }
                if (skipped > 0) { msg += ` (${skipped} skipped — not configured)`; }
                setCastStatus(`<i class="fas fa-check text-success me-1"></i>${msg}`);
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); }
        }

        // Build the candidate-member pool for the current cast (shared + cast-specific)
        function _getCastMatchPool() {
            const cast = getSelectedCastObj();
            return [...(window._voiceLibrary.shared || []).map(m => ({ ...m, source: 'shared' })),
                    ...((cast && cast.members) || []).map(m => ({ ...m, source: 'cast' }))];
        }

        // Render the <tr> rows for a cast-match proposals table (shared by the
        // single-book and bulk apply flows).
        function _renderCastMatchRows(proposals, pool) {
            const optionsFor = (selKey) => '<option value="">— skip —</option>' + pool.map(m =>
                `<option value="${escapeHtml(m.key)}" ${m.key === selKey ? 'selected' : ''}>${escapeHtml(m.name)} (${m.source})</option>`).join('');
            return proposals.map(p => {
                const m = p.match;
                const fuzzy = m && !m.exact;
                const badge = m ? (m.exact ? '<span class="badge bg-success">exact</span>' : `<span class="badge bg-warning text-dark" title="Fuzzy match — please confirm">~${m.score}</span>`) : '<span class="badge bg-light text-muted border">no match</span>';
                return `<tr class="${fuzzy ? 'table-warning' : ''}">
                    <td><input type="checkbox" class="cast-apply-check" data-char="${escapeHtml(p.character)}" ${m ? 'checked' : ''}></td>
                    <td class="small">${escapeHtml(p.character)} <span class="text-muted">(${p.line_count})</span></td>
                    <td>${badge}</td>
                    <td><select class="form-select form-select-sm cast-apply-target" data-char="${escapeHtml(p.character)}">${optionsFor(m ? m.key : '')}</select></td>
                </tr>`;
            }).join('');
        }

        // Apply a cast to the current book (fuzzy match + confirm)
        async function openCastApply() {
            if (!window._selectedCast) { return; }
            const panel = document.getElementById('cast-panel');
            panel.innerHTML = '<div class="small text-muted"><i class="fas fa-spinner fa-spin me-1"></i>Matching characters...</div>';
            let res;
            try {
                res = await API.post('/api/voice_library/match', { name: window._selectedCast });
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); renderCastMembers(); return; }

            const pool = _getCastMatchPool();
            const anyMatch = res.proposals.some(p => p.match);
            const rows = _renderCastMatchRows(res.proposals, pool);

            panel.innerHTML = `
                <div class="border rounded p-2">
                    <div class="small fw-bold mb-1">Apply "${escapeHtml(window._selectedCast)}" to this book</div>
                    <div class="alert alert-light border py-1 px-2 small mb-2"><i class="fas fa-info-circle me-1"></i>Review matches before applying. <span class="badge bg-warning text-dark">~score</span> rows are fuzzy guesses — confirm or change the target. Applying overwrites those characters' current voices.</div>
                    ${anyMatch ? '' : '<div class="alert alert-warning small py-1 px-2">No matches found for this cast.</div>'}
                    <div class="table-responsive" style="max-height:300px;overflow-y:auto;">
                        <table class="table table-sm table-hover mb-0">
                            <thead class="table-light"><tr><th style="width:30px;"></th><th>Character (lines)</th><th>Match</th><th>Use voice</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>
                    <div class="d-flex gap-2 mt-2">
                        <button class="btn btn-sm btn-success" onclick="submitCastApply()"><i class="fas fa-check me-1"></i>Apply selected</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="renderCastMembers()">Cancel</button>
                    </div>
                </div>`;
        }

        // Shared by submitCastApply/submitCastApplyBulk - both build a
        // mapping from the same checked-rows/target-select DOM shape, then
        // diverge only in which endpoint they POST to.
        function _collectCastApplyMapping() {
            const mapping = {};
            document.querySelectorAll('.cast-apply-check:checked').forEach(cb => {
                const char = cb.dataset.char;
                const sel = document.querySelector(`.cast-apply-target[data-char="${CSS.escape(char)}"]`);
                if (sel && sel.value) { mapping[char] = sel.value; }
            });
            return mapping;
        }

        async function submitCastApply() {
            const mapping = _collectCastApplyMapping();
            if (!Object.keys(mapping).length) { setCastStatus('Nothing selected to apply.', true); return; }
            try {
                const res = await API.post('/api/voice_library/apply', { cast: window._selectedCast, mapping });
                setCastStatus(`<i class="fas fa-check text-success me-1"></i>Applied ${res.count} voice${res.count !== 1 ? 's' : ''}`);
                await loadVoices();  // re-render cards with the applied configs
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); }
        }

        // --- Apply a cast to multiple saved books at once ---

        // Render a saved-scripts checkbox picker into #cast-panel
        async function openCastApplyBulk() {
            if (!window._selectedCast) { return; }
            const panel = document.getElementById('cast-panel');
            panel.innerHTML = `
                <div class="border rounded p-2">
                    <div class="small fw-bold mb-2">Apply "${escapeHtml(window._selectedCast)}" to multiple books</div>
                    <div class="d-flex align-items-center gap-2 mb-2 flex-wrap">
                        <button class="btn btn-sm btn-outline-secondary" type="button" onclick="castBulkSelectAll(true)">Select all</button>
                        <button class="btn btn-sm btn-outline-secondary" type="button" onclick="castBulkSelectAll(false)">Clear</button>
                        <button class="btn btn-sm btn-outline-primary" type="button" onclick="loadCastBulkScripts()"><i class="fas fa-sync me-1"></i>Refresh</button>
                        <div class="btn-group btn-group-sm" role="group" aria-label="Sort scripts">
                            <button class="btn btn-outline-secondary" type="button" onclick="castBulkSort('az')" title="Sort by name A→Z">A→Z</button>
                            <button class="btn btn-outline-secondary" type="button" onclick="castBulkSort('za')" title="Sort by name Z→A">Z→A</button>
                            <button class="btn btn-outline-secondary" type="button" onclick="castBulkSort('num-asc')" title="Sort by volume number 1→10">1→10</button>
                            <button class="btn btn-outline-secondary" type="button" onclick="castBulkSort('num-desc')" title="Sort by volume number 10→1">10→1</button>
                        </div>
                        <button class="btn btn-sm btn-outline-secondary" type="button" onclick="castBulkSort('reverse')" title="Reverse current order"><i class="fas fa-exchange-alt me-1"></i>Reverse</button>
                    </div>
                    <div id="cast-bulk-list" class="border rounded p-2 mb-2" style="max-height:220px;overflow-y:auto;">
                        <span class="text-muted small">Loading…</span>
                    </div>
                    <div class="d-flex gap-2 mt-2">
                        <button class="btn btn-sm btn-success" onclick="openCastApplyBulkMatch()"><i class="fas fa-arrow-right me-1"></i>Continue</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="renderCastMembers()">Cancel</button>
                    </div>
                </div>`;
            await loadCastBulkScripts();
        }

        async function loadCastBulkScripts() {
            await _loadScriptList('cast-bulk-list', (scripts) => {
                castBulkScripts = scripts;
                renderCastBulkList();
            });
        }

        function renderCastBulkList() {
            _renderScriptCheckboxList(castBulkScripts, {
                containerId: 'cast-bulk-list',
                checkClass: 'cast-bulk-check',
                idPrefix: 'cb-check-',
            });
        }

        window.castBulkSort = (mode) => {
            _sortScriptList(castBulkScripts, mode);
            renderCastBulkList();
        };

        window.castBulkSelectAll = (on) => _selectAllCheckboxes('cast-bulk-check', on);

        // Fuzzy-match the union of characters across the selected books, then
        // render the same review table as openCastApply but for all of them at once.
        async function openCastApplyBulkMatch() {
            const script_names = Array.from(document.querySelectorAll('.cast-bulk-check:checked')).map(cb => cb.dataset.name);
            if (!script_names.length) { setCastStatus('Select at least one book.', true); return; }
            const panel = document.getElementById('cast-panel');
            panel.innerHTML = '<div class="small text-muted"><i class="fas fa-spinner fa-spin me-1"></i>Matching characters...</div>';
            let res;
            try {
                res = await API.post('/api/voice_library/match_bulk', { name: window._selectedCast, script_names });
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); renderCastMembers(); return; }

            const pool = _getCastMatchPool();
            const anyMatch = res.proposals.some(p => p.match);
            const rows = _renderCastMatchRows(res.proposals, pool);

            panel.innerHTML = `
                <div class="border rounded p-2">
                    <div class="small fw-bold mb-1">Apply "${escapeHtml(window._selectedCast)}" to ${res.book_count} selected book${res.book_count !== 1 ? 's' : ''}</div>
                    <div class="alert alert-light border py-1 px-2 small mb-2"><i class="fas fa-info-circle me-1"></i>Review matches before applying. <span class="badge bg-warning text-dark">~score</span> rows are fuzzy guesses — confirm or change the target. Each book only receives entries for characters that appear in it; applying overwrites those characters' current voices in that book's saved config.</div>
                    ${anyMatch ? '' : '<div class="alert alert-warning small py-1 px-2">No matches found for this cast.</div>'}
                    <div class="table-responsive" style="max-height:300px;overflow-y:auto;">
                        <table class="table table-sm table-hover mb-0">
                            <thead class="table-light"><tr><th style="width:30px;"></th><th>Character (total lines)</th><th>Match</th><th>Use voice</th></tr></thead>
                            <tbody>${rows}</tbody>
                        </table>
                    </div>
                    <div class="d-flex gap-2 mt-2">
                        <button class="btn btn-sm btn-success" id="btn-cast-apply-bulk-submit"><i class="fas fa-check me-1"></i>Apply selected</button>
                        <button class="btn btn-sm btn-outline-secondary" onclick="renderCastMembers()">Cancel</button>
                    </div>
                </div>`;
            document.getElementById('btn-cast-apply-bulk-submit').onclick = () => submitCastApplyBulk(script_names);
        }

        async function submitCastApplyBulk(script_names) {
            const mapping = _collectCastApplyMapping();
            if (!Object.keys(mapping).length) { setCastStatus('Nothing selected to apply.', true); return; }
            const panel = document.getElementById('cast-panel');
            try {
                const res = await API.post('/api/voice_library/apply_bulk', { cast: window._selectedCast, mapping, script_names });
                const total = res.results.reduce((sum, r) => sum + r.count, 0);
                const rows = res.results.map(r => `
                    <li class="list-group-item d-flex justify-content-between align-items-center py-1 px-2 small">
                        <span>${escapeHtml(r.name)}</span>
                        <span class="${r.error ? 'text-danger' : 'text-muted'}">${r.error ? escapeHtml(r.error) : `${r.count} applied`}</span>
                    </li>`).join('');
                panel.innerHTML = `
                    <div class="border rounded p-2">
                        <div class="small fw-bold mb-1">Applied "${escapeHtml(window._selectedCast)}" to ${res.results.length} book${res.results.length !== 1 ? 's' : ''}</div>
                        <div class="alert alert-light border py-1 px-2 small mb-2"><i class="fas fa-info-circle me-1"></i>${total} voice${total !== 1 ? 's' : ''} applied in total, across each book's saved <code>voice_config.json</code>. If one of these books is currently loaded in Step 2/3, reload it (Saved Scripts → Load) to see the updated voices.</div>
                        <ul class="list-group list-group-flush border rounded mb-2">${rows || '<li class="list-group-item small text-muted py-2 px-2">No books updated.</li>'}</ul>
                        <button class="btn btn-sm btn-outline-secondary" onclick="renderCastMembers()">Done</button>
                    </div>`;
                setCastStatus(`<i class="fas fa-check text-success me-1"></i>Applied to ${res.results.length} book${res.results.length !== 1 ? 's' : ''}`);
            } catch (e) { setCastStatus(escapeHtml(e.message || String(e)), true); }
        }

        function collectVoiceConfig() {
            const cards = document.querySelectorAll('.voice-card');
            const config = {};

            cards.forEach(card => {
                const name = card.dataset.voice;
                const alias = card.querySelector('.alias-select') ? card.querySelector('.alias-select').value : '';
                const type = card.querySelector('.voice-type:checked').value;

                if (type === 'ensemble') {
                    config[name] = {
                        type: 'ensemble',
                        members: Array.from(card.querySelectorAll('.ensemble-member:checked')).map(cb => cb.value),
                        seed: "-1"
                    };
                } else if (type === 'custom') {
                    config[name] = {
                        type: 'custom',
                        voice: card.querySelector('.voice-select').value,
                        character_style: card.querySelector('.character-style').value,
                        seed: "-1"
                    };
                } else if (type === 'clone') {
                    config[name] = {
                        type: 'clone',
                        ref_text: card.querySelector('.ref-text').value,
                        ref_audio: card.querySelector('.ref-audio').value,
                        seed: "-1"
                    };
                } else if (type === 'builtin_lora') {
                    const adapterId = card.querySelector('.builtin-lora-select').value;
                    const adapterEntry = (window._loraModelsCache || []).find(m => m.id === adapterId);
                    config[name] = {
                        type: 'builtin_lora',
                        adapter_id: adapterId,
                        adapter_path: adapterEntry?.adapter_path || '',
                        character_style: card.querySelector('.builtin-lora-style').value,
                        seed: "-1"
                    };
                } else if (type === 'lora') {
                    const adapterId = card.querySelector('.lora-adapter-select').value;
                    const adapterEntry = (window._loraModelsCache || []).find(m => m.id === adapterId);
                    config[name] = {
                        type: 'lora',
                        adapter_id: adapterId,
                        adapter_path: adapterEntry?.adapter_path || (adapterId ? `lora_models/${adapterId}` : ''),
                        character_style: card.querySelector('.lora-character-style').value,
                        seed: "-1"
                    };
                } else if (type === 'design') {
                    config[name] = {
                        type: 'design',
                        description: card.querySelector('.design-description').value,
                        seed: "-1"
                    };
                }
                // Include alias_of if set
                if (alias) {
                    config[name].alias_of = alias;
                }
            });
            return config;
        }

        let _voiceSaveTimer = null;
        function saveVoicesDebounced() {
            const statusEl = document.getElementById('voice-save-status');
            statusEl.innerHTML = '<i class="fas fa-circle text-warning" style="font-size:0.5em;"></i> unsaved';
            clearTimeout(_voiceSaveTimer);
            _voiceSaveTimer = setTimeout(async () => {
                const cards = document.querySelectorAll('.voice-card');
                if (cards.length === 0) { return; }
                try {
                    const config = collectVoiceConfig();
                    await API.post('/api/save_voice_config', config);
                    statusEl.innerHTML = '<i class="fas fa-check text-success me-1"></i>saved';
                    setTimeout(() => { statusEl.innerHTML = ''; }, 2000);
                } catch (e) {
                    console.error('Failed to save voice config:', e);
                    statusEl.innerHTML = '<i class="fas fa-times text-danger me-1"></i>save failed';
                }
            }, 800);
        }

        // Auto-save on any change inside the voices list
        document.getElementById('voices-list').addEventListener('change', () => {
            saveVoicesDebounced();
        });
        document.getElementById('voices-list').addEventListener('input', () => {
            saveVoicesDebounced();
        });

        // --- Editor Tab ---
        let isPlayingSequence = false;
        let isRenderingAll = false;
        let cachedChunks = []; // Cache to track changes
        let loadChunksTimer = null; // Pending poll timer, so re-entrant calls don't stack up

        function buildSpeakerSelect(chunk) {
            const current = (chunk.speaker || '').trim();
            const names = Array.isArray(window._voicesNames) ? window._voicesNames : [];
            const normalized = [...new Set(names.map(n => (n || '').trim()).filter(Boolean))].sort((a, b) => a.localeCompare(b));
            const options = normalized.map(name => `<option value="${escapeHtml(name)}" ${name === current ? 'selected' : ''}>${escapeHtml(name)}</option>`).join('');
            const unknownOption = current && !normalized.includes(current)
                ? `<option value="${escapeHtml(current)}" selected>${escapeHtml(current)} (custom)</option>`
                : '';

            return `<select class="form-select form-select-sm" onchange="updateChunk(${chunk.id}, 'speaker', this.value)">${unknownOption}${options}</select>`;
        }

        // Check if any audio is currently playing
        function isAudioPlaying() {
            const audios = document.querySelectorAll('audio');
            for (const audio of audios) {
                if (!audio.paused && !audio.ended) { return true; }
            }
            return false;
        }

        // Update only changed rows instead of full redraw
        function updateChunkRow(chunk) {
            const tr = document.querySelector(`tr[data-id="${chunk.id}"]`);
            if (!tr) { return false; }

            const statusColor = chunk.status === 'done' ? 'success' :
                              chunk.status === 'generating' ? 'warning' :
                              chunk.status === 'error' ? 'danger' : 'secondary';

            // Update status badge
            const badge = tr.querySelector('.badge');
            if (badge) {
                badge.className = `badge bg-${statusColor}`;
                badge.innerText = chunk.status;
            }

            // Update action area (button/progress)
            const actionContainer = tr.querySelector('.d-flex');
            if (actionContainer) {
                const existingBtn = actionContainer.querySelector('button');
                const existingProgress = actionContainer.querySelector('.progress');

                if (chunk.status === 'generating') {
                    if (existingBtn && !existingProgress) {
                        const progressBar = document.createElement('div');
                        progressBar.className = 'progress';
                        progressBar.style.width = '100px';
                        progressBar.style.height = '20px';
                        progressBar.innerHTML = '<div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>';
                        actionContainer.replaceChild(progressBar, existingBtn);
                    }
                } else {
                    if (existingProgress && !existingBtn) {
                        const btn = document.createElement('button');
                        btn.className = 'btn btn-sm btn-primary';
                        btn.onclick = () => generateChunk(chunk.id);
                        btn.innerHTML = '<i class="fas fa-play"></i> Gen';
                        actionContainer.replaceChild(btn, existingProgress);
                    }
                }

                // Update audio player when status is done - always refresh src to bust cache
                if (chunk.status === 'done' && chunk.audio_path) {
                    const existingAudio = actionContainer.querySelector('audio');
                    const existingNoAudio = actionContainer.querySelector('.text-muted');
                    const newSrc = encodeURI(`/${chunk.audio_path}`) + `?t=${Date.now()}`;

                    if (existingNoAudio) {
                        // No audio element yet, create one
                        const audioHtml = `<audio class="chunk-audio" data-id="${chunk.id}" controls src="${newSrc}" style="width: 200px; height: 30px;" onplay="stopOthers(${chunk.id})"></audio>`;
                        existingNoAudio.outerHTML = audioHtml;
                    } else if (existingAudio) {
                        // Audio exists - just update the src with new cache-busting timestamp
                        // This forces browser to fetch the regenerated file
                        existingAudio.src = newSrc;
                        existingAudio.load(); // Force reload
                    }
                }
            }
            return true;
        }

        async function loadChunks(forceFullRedraw = false) {
            // Cancel any pending poll so re-entrant calls don't stack up
            if (loadChunksTimer) {
                clearTimeout(loadChunksTimer);
                loadChunksTimer = null;
            }

            const tbody = document.getElementById('chunks-table-body');

            // Show loading only if empty
            if (tbody.children.length === 0 || (tbody.children.length === 1 && tbody.children[0].children.length === 1)) {
                tbody.innerHTML = '<tr><td colspan="6" class="text-center">Loading chunks...</td></tr>';
                forceFullRedraw = true;
            }

            try {
                const chunks = await API.get('/api/chunks');
                if (chunks.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="6" class="text-center">No chunks found. Please generate script first.</td></tr>';
                    cachedChunks = [];
                    return;
                }

                // Update Full Progress Bar
                const completed = chunks.filter(c => c.status === 'done').length;
                const total = chunks.length;
                const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
                const progressBar = document.getElementById('full-progress-bar');
                if (progressBar) {
                    progressBar.style.width = `${percentage}%`;
                    progressBar.innerText = `${percentage}% (${completed}/${total})`;
                }

                // Skip redraw if playing audio (unless forced)
                if (!forceFullRedraw && (isPlayingSequence || isAudioPlaying())) {
                    // Only update status badges and progress indicators
                    chunks.forEach(chunk => updateChunkRow(chunk));
                    cachedChunks = chunks;

                    // Continue polling if generating
                    if (chunks.some(c => c.status === 'generating')) {
                        loadChunksTimer = setTimeout(() => loadChunks(false), 2000);
                    }
                    return;
                }

                // Check if we can do incremental update
                const canIncrement = !forceFullRedraw &&
                                    cachedChunks.length === chunks.length &&
                                    tbody.children.length === chunks.length;

                if (canIncrement) {
                    // Incremental update - only update changed rows
                    chunks.forEach((chunk, i) => {
                        const cached = cachedChunks[i];
                        if (!cached || cached.status !== chunk.status || cached.audio_path !== chunk.audio_path) {
                            updateChunkRow(chunk);
                        }
                    });
                } else {
                    // Full redraw needed
                    tbody.innerHTML = chunks.map(chunk => {
                        const statusColor = chunk.status === 'done' ? 'success' :
                                          chunk.status === 'generating' ? 'warning' :
                                          chunk.status === 'error' ? 'danger' : 'secondary';

                        const audioPlayer = chunk.audio_path ?
                            `<audio class="chunk-audio" data-id="${chunk.id}" controls src="${encodeURI(`/${chunk.audio_path}`)}?t=${Date.now()}" style="width: 200px; height: 30px;" onplay="stopOthers(${chunk.id})"></audio>` :
                            '<span class="text-muted small">No audio</span>';

                        const actionArea = chunk.status === 'generating' ?
                            `<div class="progress" style="width: 100px; height: 20px;">
                                <div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>
                             </div>` :
                            `<button class="btn btn-sm btn-primary" onclick="generateChunk(${chunk.id})"><i class="fas fa-play"></i> Gen</button>`;

                        return `
                            <tr data-id="${chunk.id}" class="chunk-row">
                                <td class="text-center align-middle" style="white-space:nowrap;">
                                    <button class="chunk-action-btn chunk-toggle-btn" onclick="toggleChunkExpand(this)" title="Expand/collapse"><i class="fas fa-chevron-down"></i></button><button class="chunk-action-btn" onclick="insertChunkAfter(${chunk.id})" title="Insert line below"><i class="fas fa-plus"></i></button><button class="chunk-action-btn" onclick="deleteChunk(${chunk.id})" title="Delete line"><i class="fas fa-trash" style="color:#dc3545;"></i></button>
                                </td>
                                <td>${buildSpeakerSelect(chunk)}</td>
                                <td><textarea class="form-control form-control-sm chunk-text" rows="2" onchange="updateChunk(${chunk.id}, 'text', this.value)">${escapeHtml(chunk.text)}</textarea></td>
                                <td>
                                    <textarea class="form-control form-control-sm chunk-instruct" rows="2" onchange="updateChunk(${chunk.id}, 'instruct', this.value)" title="Short TTS direction (3-8 words)">${escapeHtml(chunk.instruct || '')}</textarea>
                                    <div class="chunk-pause-row d-none mt-1 align-items-center gap-1">
                                        <small class="text-muted text-nowrap">Pause after (ms):</small>
                                        <input type="number" class="form-control form-control-sm chunk-pause-after" style="width:80px;" value="${chunk.pause_after ?? ''}" placeholder="default" min="0" step="50" onchange="updateChunk(${chunk.id}, 'pause_after', this.value === '' ? null : parseInt(this.value))">
                                    </div>
                                </td>
                                <td><span class="badge bg-${statusColor}">${escapeHtml(chunk.status)}</span></td>
                                <td>
                                    <div class="d-flex align-items-center gap-2">
                                        ${actionArea}
                                        ${audioPlayer}
                                    </div>
                                </td>
                            </tr>
                        `;
                    }).join('');
                }

                cachedChunks = chunks;

                // If any chunk is generating, poll (without full redraw)
                if (chunks.some(c => c.status === 'generating')) {
                    loadChunksTimer = setTimeout(() => loadChunks(false), 2000);
                }

            } catch (e) {
                console.error("Error loading chunks:", e);
            }
        }

        window.toggleChunkExpand = (btn) => {
            const row = btn.closest('tr');
            const expanding = !row.classList.contains('expanded');
            row.classList.toggle('expanded');

            row.querySelectorAll('.chunk-text, .chunk-instruct').forEach(ta => {
                if (expanding) {
                    // Auto-size to content
                    ta.style.height = 'auto';
                    ta.style.height = ta.scrollHeight + 'px';
                    ta.style.overflow = 'visible';
                } else {
                    // Collapse back to 2 rows
                    ta.style.height = '';
                    ta.style.overflow = '';
                }
            });

            // Show/hide pause_after control
            row.querySelectorAll('.chunk-pause-row').forEach(el => {
                if (expanding) {
                    el.classList.remove('d-none');
                    el.classList.add('d-flex');
                } else {
                    el.classList.remove('d-flex');
                    el.classList.add('d-none');
                }
            });
        };

        window.insertChunkAfter = async (id) => {
            try {
                await API.post(`/api/chunks/${id}/insert`, {});
                await loadChunks(true);
            } catch (e) {
                showToast('Failed to insert line: ' + e.message, 'error');
            }
        };

        let _lastDeleted = null;
        let _undoTimer = null;

        window.deleteChunk = async (id) => {
            try {
                const res = await fetch(`/api/chunks/${id}`, { method: 'DELETE' });
                await API._handleError(res);
                const data = await res.json();

                // Store for undo
                _lastDeleted = { chunk: data.deleted, at_index: id };
                clearTimeout(_undoTimer);

                // Show toast with undo action
                const toastId = 'toast-undo-' + Date.now();
                const container = document.getElementById('toast-container');
                const wrapper = document.createElement('div');
                wrapper.innerHTML = `
                    <div id="${toastId}" class="toast align-items-center text-white bg-warning border-0" role="alert">
                        <div class="d-flex">
                            <div class="toast-body text-dark"><span class="deleted-line-summary"></span>
                                <a href="#" class="ms-2 fw-bold text-dark">Undo</a>
                            </div>
                            <button type="button" class="btn-close me-2 m-auto" data-bs-dismiss="toast"></button>
                        </div>
                    </div>`;
                const el = wrapper.firstElementChild;
                el.querySelector('.deleted-line-summary').textContent =
                    `Line deleted (${data.deleted.speaker}: "${(data.deleted.text || '').substring(0, 40)}...")`;
                el.querySelector('a').addEventListener('click', (event) => {
                    event.preventDefault();
                    undoDeleteChunk(toastId);
                });
                container.appendChild(el);
                const toast = new bootstrap.Toast(el, { delay: 8000 });
                toast.show();
                el.addEventListener('hidden.bs.toast', () => { el.remove(); });

                // Clear undo data after timeout
                _undoTimer = setTimeout(() => { _lastDeleted = null; }, 8000);

                await loadChunks(true);
            } catch (e) {
                showToast('Failed to delete line: ' + e.message, 'error');
            }
        };

        window.undoDeleteChunk = async (toastId) => {
            if (!_lastDeleted) {
                showToast('Nothing to undo', 'warning');
                return;
            }

            try {
                await API.post('/api/chunks/restore', {
                    chunk: _lastDeleted.chunk,
                    at_index: _lastDeleted.at_index
                });

                // Dismiss the toast
                const el = document.getElementById(toastId);
                if (el) {
                    const toast = bootstrap.Toast.getInstance(el);
                    if (toast) { toast.hide(); }
                }

                _lastDeleted = null;
                clearTimeout(_undoTimer);
                showToast('Line restored', 'success');
                await loadChunks(true);
            } catch (e) {
                showToast('Undo failed: ' + e.message, 'error');
            }
        };

        window.stopOthers = (id) => {
            if (isPlayingSequence) { return; } // Sequence player handles its own logic
            document.querySelectorAll('audio').forEach(audio => {
                if (audio.dataset.id != id) {
                    audio.pause();
                }
            });
        };

        window.playSequence = async () => {
            isPlayingSequence = true;
            const btn = document.getElementById('btn-play-seq');
            btn.innerHTML = '<i class="fas fa-stop me-1"></i>Stop';
            btn.onclick = stopSequence;
            btn.classList.replace('btn-primary', 'btn-danger');

            let currentIndex = 0;
            let skippedCount = 0;

            const playNext = () => {
                if (!isPlayingSequence) { return; }

                const audios = Array.from(document.querySelectorAll('.chunk-audio'));

                // Find next valid audio
                while (currentIndex < audios.length) {
                    const audio = audios[currentIndex];
                    if (audio.getAttribute('src')) {
                        break;
                    }
                    currentIndex++;
                }

                if (currentIndex >= audios.length) {
                    stopSequence();
                    if (skippedCount > 0) {
                        showToast(`Play Sequence finished - ${skippedCount} chunk(s) skipped due to playback errors`, 'warning');
                    }
                    return;
                }

                const audio = audios[currentIndex];
                const tr = audio.closest('tr');

                // Visual feedback
                document.querySelectorAll('tr').forEach(r => r.classList.remove('table-primary'));
                tr.classList.add('table-primary');
                tr.scrollIntoView({ behavior: 'smooth', block: 'center' });

                // Guards against onerror and the play() promise rejection both
                // firing for this same audio element - only the first should
                // count as a skip and advance currentIndex, or both firing
                // would double the skip count and skip over the next track too.
                let advanced = false;
                const advanceOnce = (isSkip) => {
                    if (advanced) { return; }
                    advanced = true;
                    if (isSkip) { skippedCount++; }
                    currentIndex++;
                    playNext();
                };

                const playPromise = audio.play();

                if (playPromise !== undefined) {
                    playPromise.catch(e => {
                        console.error("Play failed (empty or skipped):", e);
                        advanceOnce(true);
                    });
                }

                audio.onended = () => {
                    advanceOnce(false);
                };

                audio.onerror = () => {
                     console.error("Audio error, skipping");
                     advanceOnce(true);
                }
            };

            playNext();
        };

        window.stopSequence = () => {
            isPlayingSequence = false;
            document.querySelectorAll('audio').forEach(a => {
                a.pause();
                a.currentTime = 0;
                a.onended = null;
            });
            document.querySelectorAll('tr').forEach(r => r.classList.remove('table-primary'));

            const btn = document.getElementById('btn-play-seq');
            if (btn) {
                btn.innerHTML = '<i class="fas fa-play me-1"></i>Play Sequence';
                btn.onclick = playSequence;
                btn.classList.replace('btn-danger', 'btn-primary');
            }
        };

        window.updateChunk = async (id, field, value) => {
            try {
                const data = {};
                data[field] = value;
                await API.post(`/api/chunks/${id}`, data);
                // Don't reload entire table to preserve focus, but maybe update status badge if needed
                // For now, next loadChunks will show updated status (pending)
            } catch (e) {
                console.error("Update failed", e);
                showToast("Failed to update chunk", 'error');
            }
        };

        // Save all pending edits from a row before generation
        async function saveRowEdits(id) {
            const tr = document.querySelector(`tr[data-id="${id}"]`);
            if (!tr) { return; }

            const inputs = tr.querySelectorAll('input, textarea');
            const data = {};

            inputs.forEach(input => {
                const changeHandler = input.getAttribute('onchange');
                if (changeHandler) {
                    // Extract field name from onchange="updateChunk(id, 'field', this.value)"
                    const match = changeHandler.match(/updateChunk\(\d+,\s*'(\w+)'/);
                    if (match) {
                        data[match[1]] = input.value;
                    }
                }
            });

            // Coerce pause_after: empty string means clear the override
            if ('pause_after' in data) {
                data.pause_after = data.pause_after === '' ? null : parseInt(data.pause_after);
            }

            // Save all fields at once
            if (Object.keys(data).length > 0) {
                console.log(`Saving chunk ${id} with data:`, data);
                await API.post(`/api/chunks/${id}`, data);
                console.log(`Chunk ${id} saved successfully`);
            }
        }

        window.generateChunk = async (id) => {
            try {
                // First, save any pending edits in this row
                await saveRowEdits(id);

                // Skip empty lines
                const tr = document.querySelector(`tr[data-id="${id}"]`);
                if (tr) {
                    const textArea = tr.querySelector('.chunk-text');
                    if (textArea && !textArea.value.trim()) {
                        showToast('Cannot generate audio for an empty line', 'error');
                        return;
                    }
                }

                // Optimistic UI update
                if (tr) {
                    const statusBadge = tr.querySelector('.badge');
                    statusBadge.className = 'badge bg-warning';
                    statusBadge.innerText = 'generating';

                    // Replace button with progress bar
                    const container = tr.querySelector('.d-flex');
                    const btn = container.querySelector('button');
                    if (btn) {
                         const progressBar = document.createElement('div');
                         progressBar.className = 'progress';
                         progressBar.style.width = '100px';
                         progressBar.style.height = '20px';
                         progressBar.innerHTML = '<div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>';
                         container.replaceChild(progressBar, btn);
                    }
                }

                await API.post(`/api/chunks/${id}/generate`, {});

                // Start polling with incremental updates (no full redraw)
                setTimeout(() => loadChunks(false), 1000);
            } catch (e) {
                showToast("Failed to start generation: " + e.message, 'error');
                loadChunks(true); // Revert UI with full redraw
            }
        };

        window.cancelRender = async (skipApi = false) => {
            isRenderingAll = false;
            document.getElementById('btn-batch-fast').style.display = 'inline-block';
            document.getElementById('btn-regen-all').style.display = 'inline-block';
            document.getElementById('btn-cancel-render').style.display = 'none';
            if (!skipApi) {
                await cancelTask('/api/cancel_audio', { onSuccess: () => loadChunks(false) });
            }
        };

        window.startRender = (regenerateAll = false) => {
            const mode = document.getElementById('tts-mode').value;
            if (mode === 'external') {
                renderAll(regenerateAll);
            } else {
                renderBatchFast(regenerateAll);
            }
        };

        // Shared by renderAll/renderBatchFast - identical except the endpoint
        // and how each one's response describes what it started (the two
        // endpoints return different fields, so that can't be a static
        // template). See FIXED.md F-065.
        async function _runBatchRender(endpoint, regenerateAll, { label, describeStart }) {
            isRenderingAll = true;
            document.getElementById('btn-batch-fast').style.display = 'none';
            document.getElementById('btn-regen-all').style.display = 'none';
            document.getElementById('btn-cancel-render').style.display = 'inline-block';

            try {
                const chunks = await API.get('/api/chunks');
                let toProcess = (regenerateAll ? chunks : chunks.filter(c => c.status !== 'done'))
                    .filter(c => c.text && c.text.trim());

                if (toProcess.length === 0) {
                    showToast("No non-empty chunks to render!", 'warning');
                    cancelRender(true);
                    return;
                }

                if (regenerateAll) {
                    if (!await showConfirm(`Regenerate all ${toProcess.length} non-empty chunks? This will replace existing audio.`)) {
                        cancelRender(true);
                        return;
                    }
                    // Re-fetch after the user confirms - showConfirm only
                    // resolves on a click, so server-side chunk state can
                    // have changed in the meantime (e.g. another tab
                    // finished a chunk). Using the pre-confirm snapshot here
                    // could send indices for chunks that already moved on.
                    const freshChunks = await API.get('/api/chunks');
                    toProcess = freshChunks.filter(c => c.text && c.text.trim());
                }

                // Mark all chunks as generating in UI
                const indices = toProcess.map(c => c.id);
                for (const id of indices) {
                    const tr = document.querySelector(`tr[data-id="${id}"]`);
                    if (tr) {
                        tr.classList.add('table-info');
                        const badge = tr.querySelector('.badge');
                        if (badge) {
                            badge.className = 'badge bg-warning';
                            badge.innerText = 'generating';
                        }
                    }
                }

                const response = await API.post(endpoint, { indices });
                console.log(`${label} started: ${describeStart(response)}`);

                // Poll for completion via the shared engine - gets the
                // staleness guard and bounded-retry-then-toast behavior every
                // other poller in this file already has, instead of a 9th
                // bespoke setInterval loop with console-only error handling.
                _startPolling('render_batch', () => API.get('/api/chunks'), {
                    intervalMs: 2000,
                    doneCheck: (updated) => {
                        if (!isRenderingAll) { return true; }
                        const stillGenerating = updated.filter(c =>
                            indices.includes(c.id) && c.status === 'generating'
                        );
                        return stillGenerating.length === 0;
                    },
                    onTick: async () => { await loadChunks(false); },
                    onDone: async (updated) => {
                        if (!isRenderingAll) { return; }
                        document.querySelectorAll('tr').forEach(r => r.classList.remove('table-info'));
                        cancelRender(true);
                        await loadChunks(false);

                        const completed = updated.filter(c => indices.includes(c.id) && c.status === 'done').length;
                        const failed = updated.filter(c => indices.includes(c.id) && c.status === 'error').length;
                        if (failed > 0) {
                            showToast(`Batch complete: ${completed} succeeded, ${failed} failed`, 'warning');
                        }
                    },
                });

            } catch (e) {
                console.error(`${label} error:`, e);
                showToast("Error during batch rendering: " + e.message, 'error');
                cancelRender(true);
            }
        }

        window.renderAll = (regenerateAll = false) => _runBatchRender('/api/generate_batch', regenerateAll, {
            label: 'Batch generation',
            describeStart: r => `${r.total_chunks} chunks with ${r.workers} workers`
        });

        window.renderBatchFast = (regenerateAll = false) => _runBatchRender('/api/generate_batch_fast', regenerateAll, {
            label: 'Fast batch',
            describeStart: r => `${r.total_chunks} chunks (batch_size=${r.batch_size}, seed=${r.batch_seed})`
        });

        document.getElementById('btn-merge').addEventListener('click', async () => {
             if (!await showConfirm("Merge all valid audio chunks into final audiobook?")) { return; }

             try {
                 await API.post('/api/merge', {});
                 // Switch to Result tab and poll
                 document.querySelector('[data-tab="audio"]').click();
                 pollLogs('audio', 'audio-logs');
             } catch (e) {
                 showToast("Merge failed: " + e.message, 'error');
             }
        });


        // --- Audacity Export ---
        window.exportAudacity = async () => {
            const statusEl = document.getElementById('audacity-status');
            statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Exporting...</span>';

            try {
                await API.post('/api/export_audacity', {});

                _startPolling('audacity_export', () => API.get('/api/status/audacity_export'), {
                    doneCheck: status => !status.running,
                    onDone: status => {
                        if (status.logs.some(l => l.includes("complete"))) {
                            statusEl.innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Done!</span>';
                            // Auto-download the zip
                            const a = document.createElement('a');
                            a.href = `/api/export_audacity?t=${Date.now()}`;
                            a.download = 'audacity_export.zip';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            setTimeout(() => { statusEl.innerHTML = ''; }, 5000);
                        } else {
                            const lastLog = status.logs[status.logs.length - 1] || 'Unknown error';
                            statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${escapeHtml(lastLog)}</span>`;
                        }
                    }
                });
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${escapeHtml(e.message)}</span>`;
            }
        };

        // Handle M4B cover image upload
        document.getElementById('m4b-cover-input').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            const statusEl = document.getElementById('m4b-cover-status');
            if (!file) { return; }
            const formData = new FormData();
            formData.append('file', file);
            try {
                const resp = await fetch('/api/m4b_cover', { method: 'POST', body: formData });
                if (!resp.ok) { throw new Error((await resp.json()).detail || resp.statusText); }
                statusEl.textContent = 'Uploaded';
                statusEl.className = 'small text-success';
            } catch (err) {
                console.error('Failed to upload M4B cover:', err);
                statusEl.textContent = err.message;
                statusEl.className = 'small text-danger';
            }
        });

        window.removeM4bCover = async () => {
            const statusEl = document.getElementById('m4b-cover-status');
            try {
                await API.del('/api/m4b_cover');
                document.getElementById('m4b-cover-input').value = '';
                statusEl.textContent = 'Removed';
                statusEl.className = 'small text-muted';
            } catch (err) {
                console.error('Failed to remove M4B cover:', err);
                statusEl.textContent = err.message;
                statusEl.className = 'small text-danger';
            }
        };

        window.exportM4B = async () => {
            const statusEl = document.getElementById('m4b-status');
            const perChunk = document.getElementById('m4b-per-chunk').checked;
            statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Exporting M4B...</span>';

            try {
                await API.post('/api/merge_m4b', {
                    per_chunk_chapters: perChunk,
                    title: document.getElementById('m4b-title').value,
                    author: document.getElementById('m4b-author').value,
                    narrator: document.getElementById('m4b-narrator').value,
                    year: document.getElementById('m4b-year').value,
                    description: document.getElementById('m4b-description').value
                });

                _startPolling('m4b_export', () => API.get('/api/status/m4b_export'), {
                    doneCheck: status => !status.running,
                    onDone: status => {
                        if (status.logs.some(l => l.includes("complete"))) {
                            statusEl.innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Done!</span>';
                            const a = document.createElement('a');
                            a.href = `/api/audiobook_m4b?t=${Date.now()}`;
                            a.download = 'audiobook.m4b';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            setTimeout(() => { statusEl.innerHTML = ''; }, 5000);
                        } else {
                            const lastLog = status.logs[status.logs.length - 1] || 'Unknown error';
                            statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${escapeHtml(lastLog)}</span>`;
                        }
                    }
                });
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${escapeHtml(e.message)}</span>`;
            }
        };

        // --- Polling Logic ---
        // Shared polling engine: every hand-rolled status poller in this file
        // used to implement its own setInterval/setTimeout + try/catch with a
        // different, undocumented error policy (some retried forever
        // silently, some gave up on the first error, one showed a toast and
        // gave up). This is the ONE consistent policy: retry silently up to
        // MAX_SILENT_ERRORS times, then a visible-but-non-blocking warning
        // toast (polling keeps going either way - nothing here needs "give
        // up forever", since every poller's done-condition is server-driven).
        // Also generalizes pollLogs's stale-response generation-counter guard
        // to any poller. See FIXED.md F-053/058/072/073/079.
        const _pollGen = {};
        function _startPolling(key, fetchFn, { intervalMs = 1000, doneCheck, onTick, onDone } = {}) {
            const myGen = (_pollGen[key] = (_pollGen[key] || 0) + 1);
            let consecutiveErrors = 0;
            const MAX_SILENT_ERRORS = 3;
            const tick = async () => {
                if (myGen !== _pollGen[key]) { return; }
                try {
                    const data = await fetchFn();
                    if (myGen !== _pollGen[key]) { return; }
                    consecutiveErrors = 0;
                    if (onTick) { await onTick(data); }
                    if (doneCheck(data)) {
                        if (onDone) { onDone(data); }
                        return;
                    }
                } catch (e) {
                    consecutiveErrors++;
                    console.error(`Poll error (${key}):`, e);
                    // Re-toast every MAX_SILENT_ERRORS failures, not just the
                    // first time the threshold is crossed - a permanently
                    // broken endpoint would otherwise warn once and then go
                    // silent for the rest of the failure streak.
                    if (consecutiveErrors % MAX_SILENT_ERRORS === 0) {
                        showToast(`Having trouble reaching the server for "${key}" status updates — still retrying...`, 'warning');
                    }
                }
                if (myGen === _pollGen[key]) { setTimeout(tick, intervalMs); }
            };
            tick();
            // Callers that need to stop a poll before its own doneCheck fires
            // (e.g. a user-initiated cancel) can call the returned function -
            // it just bumps the generation counter, which the next tick (in
            // flight or scheduled) will see and exit on.
            return () => { _pollGen[key] = (_pollGen[key] || 0) + 1; };
        }

        async function pollLogs(taskName, elementId, onDone) {
            const el = document.getElementById(elementId);
            _startPolling(`logs:${taskName}`, () => API.get(`/api/status/${taskName}`), {
                doneCheck: status => !status.running,
                onTick: status => {
                    el.innerText = status.logs.join('\n');
                    el.scrollTop = el.scrollHeight;
                },
                onDone: status => {
                    notifyJobDone(taskName);
                    if (onDone) { onDone(status); }
                    if (taskName === 'audio' && status.logs.some(l => l.includes("complete"))) {
                        // Load audio player
                        const audio = document.getElementById('main-audio');
                        audio.src = `/api/audiobook?t=${new Date().getTime()}`;
                        document.getElementById('audio-player-container').style.display = 'block';
                        document.getElementById('download-link').href = audio.src;
                    }
                    // Refresh editor chunks when script generation or review completes
                    if ((taskName === 'script' || taskName === 'review') && status.logs.some(l => l.includes("completed successfully"))) {
                        // Clear cached chunks table so next load shows fresh data
                        const tbody = document.getElementById('chunks-table-body');
                        if (tbody) { tbody.innerHTML = ''; }
                        // If editor tab is visible, refresh immediately
                        if (document.getElementById('editor-tab').style.display !== 'none') {
                            loadChunks();
                        }
                    }
                }
            });
        }
