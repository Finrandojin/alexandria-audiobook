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
        }
    });
});

// --- API Helpers ---
const API = {
    get: async (url) => {
        const res = await fetch(url);
        if (!res.ok) throw new Error(res.statusText);
        return res.json();
    },
    post: async (url, data) => {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error(res.statusText);
        return res.json();
    },
    upload: async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        if (!res.ok) throw new Error(res.statusText);
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

async function loadConfig() {
    document.getElementById('chunk-size').value = 3000;
    document.getElementById('max-tokens').value = 4096;

    try {
        const config = await API.get('/api/config');
        document.getElementById('llm-url').value = config.llm.base_url;
        document.getElementById('llm-key').value = config.llm.api_key;
        document.getElementById('llm-model').value = config.llm.model_name;
        document.getElementById('tts-mode').value = config.tts.mode || 'external';
        document.getElementById('tts-url').value = config.tts.url || 'http://127.0.0.1:7860';
        document.getElementById('tts-device').value = config.tts.device || 'auto';
        document.getElementById('tts-language').value = config.tts.language || 'English';
        document.getElementById('parallel-workers').value = config.tts.parallel_workers || 2;
        if (config.tts.batch_seed != null) {
            document.getElementById('batch-seed').value = config.tts.batch_seed;
        }
        document.getElementById('compile-codec').checked = !!config.tts.compile_codec;
        document.getElementById('sub-batch-enabled').checked = config.tts.sub_batch_enabled !== false;
        if (config.tts.sub_batch_min_size != null) {
            document.getElementById('sub-batch-min-size').value = config.tts.sub_batch_min_size;
        }
        if (config.tts.sub_batch_ratio != null) {
            document.getElementById('sub-batch-ratio').value = config.tts.sub_batch_ratio;
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
    } catch (e) {
        console.error("Failed to fetch default prompts", e);
        alert("Failed to load default prompts from server.");
    }
    document.getElementById('chunk-size').value = 3000;
    document.getElementById('max-tokens').value = 4096;
    document.getElementById('temperature').value = 0.6;
    document.getElementById('top-p').value = 0.8;
    document.getElementById('top-k').value = 20;
    document.getElementById('min-p').value = 0;
    document.getElementById('presence-penalty').value = 0;
    document.getElementById('banned-tokens').value = '';
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

    const config = {
        llm: {
            base_url: document.getElementById('llm-url').value,
            api_key: document.getElementById('llm-key').value,
            model_name: document.getElementById('llm-model').value
        },
        tts: {
            mode: document.getElementById('tts-mode').value,
            url: document.getElementById('tts-url').value,
            device: document.getElementById('tts-device').value,
            language: document.getElementById('tts-language').value,
            parallel_workers: parallelWorkers,
            batch_seed: document.getElementById('batch-seed').value ? parseInt(document.getElementById('batch-seed').value) : null,
            compile_codec: document.getElementById('compile-codec').checked,
            sub_batch_enabled: document.getElementById('sub-batch-enabled').checked,
            sub_batch_min_size: parseInt(document.getElementById('sub-batch-min-size').value) || 4,
            sub_batch_ratio: parseFloat(document.getElementById('sub-batch-ratio').value) || 5
        },
        prompts: {
            system_prompt: document.getElementById('system-prompt').value,
            user_prompt: document.getElementById('user-prompt').value
        },
        generation: {
            chunk_size: chunkSize,
            max_tokens: parseInt(document.getElementById('max-tokens').value) || 4096,
            temperature: parseFloat(document.getElementById('temperature').value),
            top_p: parseFloat(document.getElementById('top-p').value),
            top_k: parseInt(document.getElementById('top-k').value),
            min_p: parseFloat(document.getElementById('min-p').value),
            presence_penalty: parseFloat(document.getElementById('presence-penalty').value),
            banned_tokens: document.getElementById('banned-tokens').value
                ? document.getElementById('banned-tokens').value.split(',').map(t => t.trim()).filter(t => t)
                : []
        }
    };
    try {
        await API.post('/api/config', config);
        alert('Configuration Saved!');
    } catch (e) {
        alert('Error saving config: ' + e.message);
    }
});

// --- Script Tab ---
document.getElementById('btn-upload').addEventListener('click', async () => {
    const fileInput = document.getElementById('file-upload');
    if (fileInput.files.length === 0) return alert("Select a file first");

    try {
        const res = await API.upload(fileInput.files[0]);
        document.getElementById('upload-status').innerText = `Uploaded: ${res.filename}`;
    } catch (e) {
        alert("Upload failed: " + e.message);
    }
});

document.getElementById('btn-gen-script').addEventListener('click', async () => {
    try {
        await API.post('/api/generate_script', {});
        pollLogs('script', 'script-logs');
    } catch (e) {
        alert("Failed to start script gen: " + e.message);
    }
});

document.getElementById('btn-review-script').addEventListener('click', async () => {
    try {
        await API.post('/api/review_script', {});
        pollLogs('review', 'script-logs');
    } catch (e) {
        alert("Failed to start review: " + e.message);
    }
});

// --- Voices Tab ---
const AVAILABLE_VOICES = ["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"];

function createVoiceCard(voice, index) {
    const config = voice.config || {};
    const isClone = config.type === 'clone';

    return `
        <div class="card voice-card mb-3" data-voice="${voice.name}">
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <h5 class="card-title">${voice.name}</h5>
                    </div>
                    <div class="col-md-9">
                        <div class="mb-2">
                            <div class="form-check form-check-inline">
                                <input class="form-check-input voice-type" type="radio" name="type_${index}" value="custom" ${!isClone ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                <label class="form-check-label">Custom Voice</label>
                            </div>
                            <div class="form-check form-check-inline">
                                <input class="form-check-input voice-type" type="radio" name="type_${index}" value="clone" ${isClone ? 'checked' : ''} onchange="toggleVoiceType(this)">
                                <label class="form-check-label">Voice Clone</label>
                            </div>
                        </div>

                        <!-- Custom Options -->
                        <div class="custom-opts" style="display: ${!isClone ? 'block' : 'none'}">
                            <div class="row g-2">
                                <div class="col-md-6">
                                    <select class="form-select voice-select">
                                        ${AVAILABLE_VOICES.map(v => `<option value="${v}" ${config.voice === v ? 'selected' : ''}>${v}</option>`).join('')}
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <input type="text" class="form-control character-style" placeholder="Character style (e.g. refined aristocratic tone, heavy Scottish accent)" value="${config.character_style || config.default_style || ''}">
                                </div>
                            </div>
                        </div>

                        <!-- Clone Options -->
                        <div class="clone-opts" style="display: ${isClone ? 'block' : 'none'}">
                            <div class="alert alert-warning py-1 mb-2"><small>Note: Cloning requires manual file setup currently.</small></div>
                            <input type="text" class="form-control ref-text mb-2" placeholder="Reference Text" value="${config.ref_text || ''}">
                            <input type="text" class="form-control ref-audio" placeholder="Path to audio file" value="${config.ref_audio || ''}">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

window.toggleVoiceType = (radio) => {
    const card = radio.closest('.card-body');
    const customOpts = card.querySelector('.custom-opts');
    const cloneOpts = card.querySelector('.clone-opts');
    if (radio.value === 'custom') {
        customOpts.style.display = 'block';
        cloneOpts.style.display = 'none';
    } else {
        customOpts.style.display = 'none';
        cloneOpts.style.display = 'block';
    }
};

document.getElementById('btn-refresh-voices').addEventListener('click', async () => {
     // Trigger parse
     try {
        await API.post('/api/parse_voices', {});
        // Wait a bit for it to run (it's fast)
        setTimeout(loadVoices, 1000);
    } catch (e) {
        console.error(e);
    }
});

async function loadVoices() {
    const voices = await API.get('/api/voices');
    const container = document.getElementById('voices-list');
    if (voices.length === 0) {
        container.innerHTML = '<div class="alert alert-info">No voices found. Generate a script first.</div>';
        return;
    }
    container.innerHTML = voices.map((v, i) => createVoiceCard(v, i)).join('');
}

document.getElementById('btn-save-voices').addEventListener('click', async () => {
    const cards = document.querySelectorAll('.voice-card');
    const config = {};

    cards.forEach(card => {
        const name = card.dataset.voice;
        const type = card.querySelector('.voice-type:checked').value;

        if (type === 'custom') {
            config[name] = {
                type: 'custom',
                voice: card.querySelector('.voice-select').value,
                character_style: card.querySelector('.character-style').value,
                seed: "-1"
            };
        } else {
            config[name] = {
                type: 'clone',
                ref_text: card.querySelector('.ref-text').value,
                ref_audio: card.querySelector('.ref-audio').value,
                seed: "-1"
            };
        }
    });

    try {
        await API.post('/api/save_voice_config', config);
        alert('Voice configuration saved!');
    } catch (e) {
        alert('Error saving voices: ' + e.message);
    }
});

// --- Editor Tab ---
let isPlayingSequence = false;
let isRenderingAll = false;
let cachedChunks = []; // Cache to track changes

// Check if any audio is currently playing
function isAudioPlaying() {
    const audios = document.querySelectorAll('audio');
    for (const audio of audios) {
        if (!audio.paused && !audio.ended) return true;
    }
    return false;
}

// Update only changed rows instead of full redraw
function updateChunkRow(chunk) {
    const tr = document.querySelector(`tr[data-id="${chunk.id}"]`);
    if (!tr) return false;

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
            const newSrc = `/${chunk.audio_path}?t=${Date.now()}`;

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
    const tbody = document.getElementById('chunks-table-body');

    // Show loading only if empty
    if (tbody.children.length === 0 || (tbody.children.length === 1 && tbody.children[0].children.length === 1)) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center">Loading chunks...</td></tr>';
        forceFullRedraw = true;
    }

    try {
        const chunks = await API.get('/api/chunks');
        if (chunks.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No chunks found. Please generate script first.</td></tr>';
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
                setTimeout(() => loadChunks(false), 2000);
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
                    `<audio class="chunk-audio" data-id="${chunk.id}" controls src="/${chunk.audio_path}?t=${Date.now()}" style="width: 200px; height: 30px;" onplay="stopOthers(${chunk.id})"></audio>` :
                    '<span class="text-muted small">No audio</span>';

                const actionArea = chunk.status === 'generating' ?
                    `<div class="progress" style="width: 100px; height: 20px;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated bg-warning" role="progressbar" style="width: 100%"></div>
                     </div>` :
                    `<button class="btn btn-sm btn-primary" onclick="generateChunk(${chunk.id})"><i class="fas fa-play"></i> Gen</button>`;

                return `
                    <tr data-id="${chunk.id}">
                        <td><input type="text" class="form-control form-control-sm" value="${chunk.speaker}" onchange="updateChunk(${chunk.id}, 'speaker', this.value)"></td>
                        <td><textarea class="form-control form-control-sm" rows="2" onchange="updateChunk(${chunk.id}, 'text', this.value)">${chunk.text}</textarea></td>
                        <td><input type="text" class="form-control form-control-sm" value="${chunk.instruct || ''}" onchange="updateChunk(${chunk.id}, 'instruct', this.value)" title="Short TTS direction (3-8 words)"></td>
                        <td><span class="badge bg-${statusColor}">${chunk.status}</span></td>
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
            setTimeout(() => loadChunks(false), 2000);
        }

    } catch (e) {
        console.error("Error loading chunks:", e);
    }
}

window.stopOthers = (id) => {
    if (isPlayingSequence) return; // Sequence player handles its own logic
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

    const audios = Array.from(document.querySelectorAll('.chunk-audio'));
    if (audios.length === 0) {
        stopSequence();
        return;
    }

    let currentIndex = 0;

    const playNext = () => {
        if (!isPlayingSequence) return;

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
            return;
        }

        const audio = audios[currentIndex];
        const tr = audio.closest('tr');

        // Visual feedback
        document.querySelectorAll('tr').forEach(r => r.classList.remove('table-primary'));
        tr.classList.add('table-primary');
        tr.scrollIntoView({ behavior: 'smooth', block: 'center' });

        const playPromise = audio.play();

        if (playPromise !== undefined) {
            playPromise.catch(e => {
                console.log("Play failed (empty or skipped):", e);
                // If play fails, move next
                currentIndex++;
                playNext();
            });
        }

        audio.onended = () => {
            currentIndex++;
            playNext();
        };

        audio.onerror = () => {
             console.log("Audio error, skipping");
             currentIndex++;
             playNext();
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
        alert("Failed to update chunk");
    }
};

// Save all pending edits from a row before generation
async function saveRowEdits(id) {
    const tr = document.querySelector(`tr[data-id="${id}"]`);
    if (!tr) return;

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

        // Optimistic UI update
        const tr = document.querySelector(`tr[data-id="${id}"]`);
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
        alert("Failed to start generation: " + e.message);
        loadChunks(true); // Revert UI with full redraw
    }
};

window.cancelRender = () => {
    isRenderingAll = false;
    document.getElementById('btn-render-all').style.display = 'inline-block';
    document.getElementById('btn-batch-fast').style.display = 'inline-block';
    document.getElementById('btn-regen-all').style.display = 'inline-block';
    document.getElementById('btn-cancel-render').style.display = 'none';
};

window.renderAll = async (regenerateAll = false) => {
    isRenderingAll = true;
    document.getElementById('btn-render-all').style.display = 'none';
    document.getElementById('btn-regen-all').style.display = 'none';
    document.getElementById('btn-cancel-render').style.display = 'inline-block';

    try {
        const chunks = await API.get('/api/chunks');
        const toProcess = regenerateAll ? chunks : chunks.filter(c => c.status !== 'done');

        if (toProcess.length === 0) {
            alert("All chunks are already rendered!");
            cancelRender();
            return;
        }

        if (regenerateAll && !confirm(`Regenerate all ${chunks.length} chunks? This will replace existing audio.`)) {
            cancelRender();
            return;
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

        // Call batch endpoint for parallel processing
        const response = await API.post('/api/generate_batch', { indices });
        console.log(`Batch generation started: ${response.total_chunks} chunks with ${response.workers} workers`);

        // Poll for completion
        const pollInterval = setInterval(async () => {
            if (!isRenderingAll) {
                clearInterval(pollInterval);
                return;
            }

            try {
                await loadChunks(false);
                const updated = await API.get('/api/chunks');
                const stillGenerating = updated.filter(c =>
                    indices.includes(c.id) && c.status === 'generating'
                );

                if (stillGenerating.length === 0) {
                    clearInterval(pollInterval);
                    // Clear highlights
                    document.querySelectorAll('tr').forEach(r => r.classList.remove('table-info'));
                    cancelRender();
                    await loadChunks(false);

                    // Show completion summary
                    const completed = updated.filter(c => indices.includes(c.id) && c.status === 'done').length;
                    const failed = updated.filter(c => indices.includes(c.id) && c.status === 'error').length;
                    if (failed > 0) {
                        alert(`Batch complete: ${completed} succeeded, ${failed} failed`);
                    }
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 2000);

    } catch (e) {
        console.error("Render All error:", e);
        alert("Error during batch rendering: " + e.message);
        cancelRender();
    }
};

window.renderBatchFast = async (regenerateAll = false) => {
    isRenderingAll = true;
    document.getElementById('btn-render-all').style.display = 'none';
    document.getElementById('btn-batch-fast').style.display = 'none';
    document.getElementById('btn-regen-all').style.display = 'none';
    document.getElementById('btn-cancel-render').style.display = 'inline-block';

    try {
        const chunks = await API.get('/api/chunks');
        const toProcess = regenerateAll ? chunks : chunks.filter(c => c.status !== 'done');

        if (toProcess.length === 0) {
            alert("All chunks are already rendered!");
            cancelRender();
            return;
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

        // Call fast batch endpoint (requires custom Qwen3-TTS)
        const response = await API.post('/api/generate_batch_fast', { indices });
        console.log(`Fast batch started: ${response.total_chunks} chunks (batch_size=${response.batch_size}, seed=${response.batch_seed})`);

        // Poll for completion
        const pollInterval = setInterval(async () => {
            if (!isRenderingAll) {
                clearInterval(pollInterval);
                return;
            }

            try {
                await loadChunks(false);
                const updated = await API.get('/api/chunks');
                const stillGenerating = updated.filter(c =>
                    indices.includes(c.id) && c.status === 'generating'
                );

                if (stillGenerating.length === 0) {
                    clearInterval(pollInterval);
                    document.querySelectorAll('tr').forEach(r => r.classList.remove('table-info'));
                    cancelRender();
                    await loadChunks(false);

                    const completed = updated.filter(c => indices.includes(c.id) && c.status === 'done').length;
                    const failed = updated.filter(c => indices.includes(c.id) && c.status === 'error').length;
                    if (failed > 0) {
                        alert(`Batch complete: ${completed} succeeded, ${failed} failed`);
                    }
                }
            } catch (e) {
                console.error("Polling error", e);
            }
        }, 2000);

    } catch (e) {
        console.error("Batch Fast error:", e);
        alert("Error during batch rendering: " + e.message);
        cancelRender();
    }
};

document.getElementById('btn-merge').addEventListener('click', async () => {
     if (!confirm("Merge all valid audio chunks into final audiobook?")) return;

     try {
         await API.post('/api/merge', {});
         // Switch to Result tab and poll
         document.querySelector('[data-tab="audio"]').click();
         pollLogs('audio', 'audio-logs');
     } catch (e) {
         alert("Merge failed: " + e.message);
     }
});


// --- Audacity Export ---
window.exportAudacity = async () => {
    const statusEl = document.getElementById('audacity-status');
    statusEl.innerHTML = '<span class="text-info"><i class="fas fa-spinner fa-spin me-1"></i>Exporting...</span>';

    try {
        await API.post('/api/export_audacity', {});

        const poll = setInterval(async () => {
            try {
                const status = await API.get('/api/status/audacity_export');
                if (!status.running) {
                    clearInterval(poll);
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
                        statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${lastLog}</span>`;
                    }
                }
            } catch (e) {
                clearInterval(poll);
                statusEl.innerHTML = `<span class="text-danger">Poll error: ${e.message}</span>`;
            }
        }, 1000);
    } catch (e) {
        statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>${e.message}</span>`;
    }
};

// --- Polling Logic ---
async function pollLogs(taskName, elementId) {
    const el = document.getElementById(elementId);
    const interval = setInterval(async () => {
        try {
            const status = await API.get(`/api/status/${taskName}`);
            el.innerText = status.logs.join('\n');
            el.scrollTop = el.scrollHeight;

            if (!status.running) {
                clearInterval(interval);
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
                    if (tbody) tbody.innerHTML = '';
                    // If editor tab is visible, refresh immediately
                    if (document.getElementById('editor-tab').style.display !== 'none') {
                        loadChunks();
                    }
                }
            }
        } catch (e) {
            console.error("Poll error", e);
            clearInterval(interval);
        }
    }, 1000);
}

// ── Saved Scripts ──────────────────────────────────────

async function loadSavedScripts() {
    try {
        const res = await fetch('/api/scripts');
        const scripts = await res.json();
        const container = document.getElementById('saved-scripts-list');

        if (!scripts.length) {
            container.innerHTML = '<p class="text-muted mb-0">No saved scripts yet.</p>';
            return;
        }

        container.innerHTML = scripts.map(s => {
            const date = new Date(s.created * 1000).toLocaleDateString('en-US', {
                month: 'short', day: 'numeric', year: 'numeric'
            });
            const voiceBadge = s.has_voice_config
                ? '<span class="badge bg-info ms-2" title="Includes voice configuration">voices</span>'
                : '';
            return `
                <div class="d-flex align-items-center justify-content-between py-2 border-bottom">
                    <div>
                        <strong>${s.name}</strong>${voiceBadge}
                        <small class="text-muted ms-2">${date}</small>
                    </div>
                    <div>
                        <button class="btn btn-sm btn-outline-success me-1" onclick="loadScript('${s.name}')"><i class="fas fa-upload me-1"></i>Load</button>
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteScript('${s.name}')"><i class="fas fa-trash"></i></button>
                    </div>
                </div>`;
        }).join('');
    } catch (e) {
        console.error('Failed to load saved scripts:', e);
    }
}

async function saveScript() {
    const nameInput = document.getElementById('save-script-name');
    const name = nameInput.value.trim();
    if (!name) {
        alert('Please enter a name for the script.');
        return;
    }
    try {
        const res = await fetch('/api/scripts/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name})
        });
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Failed to save script.');
            return;
        }
        nameInput.value = '';
        loadSavedScripts();
    } catch (e) {
        alert('Error saving script: ' + e.message);
    }
}

async function loadScript(name) {
    if (!confirm(`Load "${name}"? This will replace your current script and chunks.`)) return;
    try {
        const res = await fetch('/api/scripts/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name})
        });
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Failed to load script.');
            return;
        }
        const data = await res.json();
        document.getElementById('readableScriptTarget').value = '/scripts/' + name + '.json';
        loadSavedScripts();
        loadChunks();
        loadVoices();
    } catch (e) {
        alert('Error loading script: ' + e.message);
    }
}

async function deleteScript(name) {
    if (!confirm(`Delete saved script "${name}"? This cannot be undone.`)) return;
    try {
        const res = await fetch(`/api/scripts/${encodeURIComponent(name)}`, {method: 'DELETE'});
        if (!res.ok) {
            const err = await res.json();
            alert(err.detail || 'Failed to delete script.');
            return;
        }
        loadSavedScripts();
    } catch (e) {
        alert('Error deleting script: ' + e.message);
    }
}

// Init
loadConfig();
loadVoices();
loadSavedScripts();
