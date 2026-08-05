        // ── Saved Scripts ──────────────────────────────────────

        async function loadSavedScripts() {
            await _loadScriptList('saved-scripts-list', (scripts) => {
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
                                <strong>${escapeHtml(s.name)}</strong>${voiceBadge}
                                <small class="text-muted ms-2">${date}</small>
                            </div>
                            <div>
                                <button class="btn btn-sm btn-outline-info me-1" onclick='auditSavedScript(${JSON.stringify(s.name)})'><i class="fas fa-stethoscope me-1"></i>Audit</button>
                                <button class="btn btn-sm btn-outline-success me-1" onclick='loadScript(${JSON.stringify(s.name)})'><i class="fas fa-upload me-1"></i>Load</button>
                                <button class="btn btn-sm btn-outline-danger" onclick='deleteScript(${JSON.stringify(s.name)})'><i class="fas fa-trash"></i></button>
                            </div>
                        </div>`;
                }).join('');
            });
        }

        async function auditSavedScript(name) {
            const panel = document.getElementById('saved-script-preflight');
            panel.style.display = 'block';
            panel.className = 'alert alert-secondary small mt-3 mb-0';
            panel.textContent = `Auditing ${name}…`;
            try {
                const report = await API.post(`/api/scripts/${encodeURIComponent(name)}/preflight`, {});
                const counts = report.counts || {};
                const risks = (report.findings || []).filter(item => item.code === 'nonprose_speech_risk');
                panel.className = `alert ${counts.blocking ? 'alert-danger' : risks.length ? 'alert-warning' : 'alert-success'} small mt-3 mb-0`;
                const items = risks.map(item => {
                    const details = item.details || {};
                    const entries = (item.entry_numbers || []).join(', ');
                    const categories = (details.categories || []).join(', ');
                    return `<li>Entry ${escapeHtml(entries)}: ${escapeHtml(categories)} — preview: ${escapeHtml(details.normalized_preview || '')}. Audio validation: ${escapeHtml((details.validation || {}).status || 'unknown')}.</li>`;
                }).join('');
                panel.innerHTML = `<strong>${escapeHtml(name)}:</strong> ${counts.blocking || 0} blocking, ${counts.manual_review || 0} manual review.`
                    + (items ? `<div class="mt-1">Non-prose entries have elevated TTS error risk:</div><ul class="mb-0">${items}</ul>` : ' No non-prose speech risks detected.');
            } catch (error) {
                panel.className = 'alert alert-danger small mt-3 mb-0';
                panel.textContent = `Audit failed: ${error.message || error}`;
            }
        }

        async function saveScript() {
            const nameInput = document.getElementById('save-script-name');
            const name = nameInput.value.trim();
            if (!name) {
                showToast('Please enter a name for the script.', 'warning');
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
                    showToast(err.detail || 'Failed to save script.', 'error');
                    return;
                }
                nameInput.value = '';
                loadSavedScripts();
            } catch (e) {
                console.error('Error saving script:', e);
                showToast('Error saving script: ' + e.message, 'error');
            }
        }

        async function loadScript(name) {
            if (!await showConfirm(`Load "${name}"? This will replace your current script and chunks.`)) { return; }
            try {
                const res = await fetch('/api/scripts/load', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name})
                });
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to load script.', 'error');
                    return;
                }
                showToast(`Script "${name}" loaded.`, 'success');
                clearVoiceSuggestions();
                await loadChunks(true);
                await loadVoices();
                loadSavedScripts();
                resetDesignerForm();
                loadDesignedVoices();
            } catch (e) {
                console.error('Error loading script:', e);
                showToast('Error loading script: ' + e.message, 'error');
            }
        }

        async function deleteScript(name) {
            if (!await showConfirm(`Delete saved script "${name}"? This cannot be undone.`)) { return; }
            try {
                const res = await fetch(`/api/scripts/${encodeURIComponent(name)}`, {method: 'DELETE'});
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Failed to delete script.', 'error');
                    return;
                }
                loadSavedScripts();
            } catch (e) {
                console.error('Error deleting script:', e);
                showToast('Error deleting script: ' + e.message, 'error');
            }
        }

        // --- Voice Designer ---
        window._designedVoicesCache = [];
        window._cloneVoicesCache = [];
        window._currentPreviewFile = null;

        async function loadDesignedVoices() {
            try {
                const voices = await API.get('/api/voice_design/list');
                window._designedVoicesCache = voices;
                const container = document.getElementById('designed-voices-list');

                if (!voices.length) {
                    container.innerHTML = '<p class="text-muted mb-0">No designed voices yet. Generate and save a preview above.</p>';
                    return;
                }

                container.innerHTML = `
                    <table class="table table-sm table-hover mb-0">
                        <thead><tr><th>Name</th><th>Description</th><th style="width:120px">Actions</th></tr></thead>
                        <tbody>
                            ${voices.map(v => `
                                <tr>
                                    <td><strong>${escapeHtml(v.name)}</strong></td>
                                    <td class="text-muted" style="max-width:400px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${escapeHtml(v.description)}</td>
                                    <td>
                                        <button class="btn btn-sm btn-outline-primary me-1" onclick='playDesignedVoice(${JSON.stringify(v.filename)})' title="Play"><i class="fas fa-play"></i></button>
                                        <button class="btn btn-sm btn-outline-secondary me-1" onclick='openDesignedVoiceForEdit(${JSON.stringify(v.id)})' title="Edit"><i class="fas fa-edit"></i></button>
                                        <button class="btn btn-sm btn-outline-danger" onclick='deleteDesignedVoice(${JSON.stringify(v.id)})' title="Delete"><i class="fas fa-trash"></i></button>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>`;
            } catch (e) {
                console.error('Failed to load designed voices:', e);
                showToast('Failed to load designed voices', 'error');
            }
        }

        function resetDesignerForm() {
            document.getElementById('design-voice-name').value = '';
            document.getElementById('design-source-name').value = '';
            document.getElementById('design-description').value = '';
            document.getElementById('design-sample-text').value = '';
            document.getElementById('design-alias-select').innerHTML = '<option value="">-- None --</option>';
            document.getElementById('design-preview-container').style.display = 'none';
            document.getElementById('design-status').innerHTML = '';
            window._editingDesignedVoiceId = null;
            window._currentPreviewFile = null;
            const previewButton = document.getElementById('btn-design-preview');
            if (previewButton) {
                previewButton.innerHTML = '<i class="fas fa-wand-magic-sparkles me-1"></i>Generate Preview';
            }
        }

        window.generateDesignPreview = async () => {
            const description = document.getElementById('design-description').value.trim();
            const sampleText = document.getElementById('design-sample-text').value.trim();
            const statusEl = document.getElementById('design-status');
            const previewContainer = document.getElementById('design-preview-container');

            if (!description) { showToast('Please enter a voice description.', 'warning'); return; }
            if (!sampleText) { showToast('Please enter sample text.', 'warning'); return; }

            const btn = document.getElementById('btn-design-preview');
            btn.disabled = true;
            statusEl.innerHTML = window._editingDesignedVoiceId
                ? '<i class="fas fa-spinner fa-spin me-1"></i>Re-designing preview (this may take a moment)...'
                : '<i class="fas fa-spinner fa-spin me-1"></i>Generating preview (this may take a moment on first run)...';
            previewContainer.style.display = 'none';

            try {
                const result = await API.post('/api/voice_design/preview', {
                    description: description,
                    sample_text: sampleText
                });

                const audio = document.getElementById('design-preview-audio');
                audio.src = result.audio_url + '?t=' + Date.now();
                previewContainer.style.display = 'block';
                statusEl.innerHTML = window._editingDesignedVoiceId
                    ? '<span class="text-success"><i class="fas fa-check me-1"></i>Voice re-designed</span>'
                    : '<span class="text-success"><i class="fas fa-check me-1"></i>Preview ready</span>';

                // Extract filename from URL for save
                window._currentPreviewFile = result.audio_url.split('/').pop().split('?')[0];
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger"><i class="fas fa-times me-1"></i>Failed: ${escapeHtml(e.message)}</span>`;
            } finally {
                btn.disabled = false;
            }
        };

        window.saveDesignedVoice = async () => {
            const name = document.getElementById('design-voice-name').value.trim();
            if (!name) { showToast('Please enter a name for the voice.', 'warning'); return; }
            if (!window._currentPreviewFile) { showToast('Generate a preview first.', 'warning'); return; }

            try {
                await API.post('/api/voice_design/save', {
                    name: name,
                    description: document.getElementById('design-description').value.trim(),
                    sample_text: document.getElementById('design-sample-text').value.trim(),
                    preview_file: window._currentPreviewFile
                });
                document.getElementById('design-voice-name').value = '';
                window._editingDesignedVoiceId = null;
                loadDesignedVoices();
                // If we're editing a generated persona, propagate alias choice to voice card and trigger save
                const source = document.getElementById('design-source-name').value;
                const selectedAlias = document.getElementById('design-alias-select').value;
                if (source) {
                    const card = document.querySelector(`.voice-card[data-voice="${source}"]`);
                    if (card) {
                        const aliasSel = card.querySelector('.alias-select');
                        if (aliasSel) {
                            aliasSel.value = selectedAlias || '';
                            // Trigger save via existing auto-save debounce
                            saveVoicesDebounced();
                        }
                    }
                }
            } catch (e) {
                showToast('Error saving voice: ' + e.message, 'error');
            }
        };

        window.playDesignedVoice = (filename) => {
            const audio = new Audio(`/designed_voices/${filename}?t=${Date.now()}`);
            audio.play();
        };

        window.deleteDesignedVoice = async (voiceId) => {
            if (!await showConfirm('Delete this designed voice?')) return;
            try {
                const res = await fetch(`/api/voice_design/${encodeURIComponent(voiceId)}`, {method: 'DELETE'});
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete.', 'error'); return; }
                loadDesignedVoices();
            } catch (e) {
                showToast('Error deleting voice: ' + e.message, 'error');
            }
        };

        window.openDesignedVoiceForEdit = async (voiceId) => {
            try {
                const voice = (window._designedVoicesCache || []).find(v => v.id === voiceId);
                if (!voice) {
                    showToast('Designed voice not found', 'error');
                    return;
                }

                // Switch to Designer tab
                document.querySelector('[data-tab="designer"]').click();

                // Populate fields
                document.getElementById('design-voice-name').value = voice.name || '';
                document.getElementById('design-source-name').value = voice.name || '';
                document.getElementById('design-description').value = voice.description || '';
                document.getElementById('design-sample-text').value = voice.sample_text || '';

                // Populate alias dropdown
                const aliasSelect = document.getElementById('design-alias-select');
                aliasSelect.innerHTML = '<option value="">-- None --</option>';
                const names = (window._voicesNames || []).filter(n => n !== voice.name);
                names.forEach(n => {
                    const opt = document.createElement('option');
                    opt.value = n;
                    opt.text = n;
                    aliasSelect.appendChild(opt);
                });

                // Try to read existing alias from voices config
                try {
                    const voices = await API.get('/api/voices');
                    const entry = voices.find(v => v.name === voice.name);
                    if (entry && entry.config && entry.config.alias_of) {
                        aliasSelect.value = entry.config.alias_of;
                    } else {
                        aliasSelect.value = '';
                    }
                } catch (e) {
                    aliasSelect.value = '';
                }

                // Update preview audio and current preview file
                const audio = document.getElementById('design-preview-audio');
                audio.src = `/designed_voices/${voice.filename}?t=${Date.now()}`;
                window._currentPreviewFile = voice.filename;
                window._editingDesignedVoiceId = voice.id;
                document.getElementById('design-preview-container').style.display = 'block';

                const previewButton = document.getElementById('btn-design-preview');
                if (previewButton) {
                    previewButton.innerHTML = '<i class="fas fa-wand-magic-sparkles me-1"></i>Re-design Voice';
                }

                // Focus description for quick edits
                document.getElementById('design-description').focus();
                showToast('Loaded designed voice for editing', 'info');
            } catch (e) {
                showToast('Failed to load voice for edit: ' + e.message, 'error');
            }
        };

        window.openVoiceDesignEditor = (button) => {
            const card = button.closest('.card-body');
            const cardRoot = button.closest('.voice-card');
            const voiceName = cardRoot ? cardRoot.dataset.voice : '';
            const description = card ? (card.querySelector('.design-description')?.value || '') : '';

            document.querySelector('[data-tab="designer"]').click();
            document.getElementById('design-voice-name').value = voiceName;
            document.getElementById('design-source-name').value = voiceName;
            document.getElementById('design-description').value = description;
            window._editingDesignedVoiceId = voiceName || null;
            window._currentPreviewFile = null;
            document.getElementById('design-preview-container').style.display = 'none';

            const previewButton = document.getElementById('btn-design-preview');
            if (previewButton) {
                previewButton.innerHTML = '<i class="fas fa-wand-magic-sparkles me-1"></i>Re-design Voice';
            }

            const statusEl = document.getElementById('design-status');
            if (statusEl) {
                statusEl.innerHTML = '<span class="text-muted"><i class="fas fa-info me-1"></i>Edit the description, then generate a preview.</span>';
            }
        };

        window.onDesignedVoiceSelect = (select) => {
            const card = select.closest('.card-body');
            const refText = card.querySelector('.ref-text');
            const refAudio = card.querySelector('.ref-audio');
            const playBtn = card.querySelector('.clone-play-btn');
            const deleteBtn = card.querySelector('.clone-delete-btn');
            const val = select.value;

            if (val === '' || val === '__manual__') {
                refAudio.readOnly = false;
                if (val === '__manual__') {
                    refAudio.value = '';
                    refText.value = '';
                }
                if (playBtn) playBtn.style.display = 'none';
                if (deleteBtn) deleteBtn.style.display = 'none';
                refAudio.focus();
                return;
            }

            if (val.startsWith('clone:')) {
                const voiceId = val.substring(6);
                const voice = (window._cloneVoicesCache || []).find(v => v.id === voiceId);
                if (voice) {
                    refAudio.value = `clone_voices/${voice.filename}`;
                    refText.value = '';
                    refAudio.readOnly = true;
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'inline-block';
                }
            } else if (val.startsWith('design:')) {
                const voiceId = val.substring(7);
                const voice = (window._designedVoicesCache || []).find(v => v.id === voiceId);
                if (voice) {
                    refAudio.value = `designed_voices/${voice.filename}`;
                    refText.value = voice.sample_text;
                    refAudio.readOnly = true;
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'none';
                }
            } else {
                // Legacy: plain voice ID (backward compat with old designed voice values)
                const voice = (window._designedVoicesCache || []).find(v => v.id === val);
                if (voice) {
                    refAudio.value = `designed_voices/${voice.filename}`;
                    refText.value = voice.sample_text;
                    refAudio.readOnly = true;
                    if (playBtn) playBtn.style.display = 'inline-block';
                    if (deleteBtn) deleteBtn.style.display = 'none';
                }
            }
            saveVoicesDebounced();
        };

        // --- Clone Voice Upload Handlers ---

        window.uploadCloneVoice = (btn) => {
            const card = btn.closest('.card-body');
            card.querySelector('.clone-voice-file-input').click();
        };

        window.handleCloneVoiceUpload = async (input) => {
            const file = input.files[0];
            if (!file) { return; }
            input.value = '';

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/clone_voices/upload', { method: 'POST', body: formData });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Upload failed', 'error'); return; }
                const result = await res.json();

                // Refresh cache and rebuild voice cards
                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
                await loadVoices();

                showToast(`Uploaded "${file.name}"`, 'success');
            } catch (e) {
                showToast('Upload failed: ' + e.message, 'error');
            }
        };

        window.playCloneVoice = (btn) => {
            const card = btn.closest('.card-body');
            const refAudio = card.querySelector('.ref-audio').value;
            if (refAudio) {
                const audio = new Audio(`/${refAudio}?t=${Date.now()}`);
                audio.play();
            }
        };

        window.deleteCloneVoice = async (btn) => {
            if (!await showConfirm('Delete this uploaded clone voice?')) { return; }
            const card = btn.closest('.card-body');
            const select = card.querySelector('.designed-voice-select');
            const val = select.value;
            if (!val.startsWith('clone:')) { return; }
            const voiceId = val.substring(6);

            try {
                const res = await fetch(`/api/clone_voices/${encodeURIComponent(voiceId)}`, { method: 'DELETE' });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete', 'error'); return; }

                window._cloneVoicesCache = await API.get('/api/clone_voices/list');
                await loadVoices();
                showToast('Clone voice deleted', 'success');
            } catch (e) {
                showToast('Error: ' + e.message, 'error');
            }
        };

        // --- LoRA Training ---
        window._loraModelsCache = [];
