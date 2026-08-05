        async function loadLoraDatasets() {
            try {
                const datasets = await API.get('/api/lora/datasets');
                const listEl = document.getElementById('lora-datasets-list');
                const selectEl = document.getElementById('lora-dataset-select');

                // Update dropdown
                const currentVal = selectEl.value;
                selectEl.innerHTML = '<option value="">-- Select dataset --</option>' +
                    datasets.map(d => `<option value="${escapeHtml(d.dataset_id)}">${escapeHtml(d.dataset_id)} (${d.sample_count} samples)</option>`).join('');
                if (currentVal) { selectEl.value = currentVal; }

                // Update list
                if (!datasets.length) {
                    listEl.innerHTML = '<span class="text-muted">No datasets uploaded yet.</span>';
                    return;
                }
                listEl.innerHTML = datasets.map(d => `
                    <div class="d-flex justify-content-between align-items-center py-1">
                        <span><strong>${escapeHtml(d.dataset_id)}</strong> <small class="text-muted">(${d.sample_count} samples)</small></span>
                        <button class="btn btn-sm btn-outline-danger" onclick='deleteLoraDataset(${JSON.stringify(d.dataset_id)})'><i class="fas fa-trash"></i></button>
                    </div>
                `).join('');
            } catch (e) {
                console.error('Failed to load LoRA datasets:', e);
            }
        }

        window.uploadLoraDataset = async () => {
            const fileInput = document.getElementById('lora-dataset-file');
            if (!fileInput.files.length) { showToast('Select a ZIP file first.', 'warning'); return; }

            const file = fileInput.files[0];
            if (!file.name.endsWith('.zip')) { showToast('File must be a .zip archive.', 'warning'); return; }

            const formData = new FormData();
            formData.append('file', file);

            try {
                const res = await fetch('/api/lora/upload_dataset', { method: 'POST', body: formData });
                if (!res.ok) {
                    const err = await res.json();
                    showToast(err.detail || 'Upload failed.', 'error');
                    return;
                }
                const result = await res.json();
                showToast(`Dataset "${result.dataset_id}" uploaded (${result.sample_count} samples).`, 'success');
                fileInput.value = '';
                loadLoraDatasets();
            } catch (e) {
                showToast('Upload error: ' + e.message, 'error');
            }
        };

        window.deleteLoraDataset = async (datasetId) => {
            if (!await showConfirm(`Delete dataset "${datasetId}"?`)) { return; }
            try {
                const res = await fetch(`/api/lora/datasets/${encodeURIComponent(datasetId)}`, { method: 'DELETE' });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete.', 'error'); return; }
                loadLoraDatasets();
            } catch (e) {
                showToast('Error deleting dataset: ' + e.message, 'error');
            }
        };


        window.startLoraTraining = async () => {
            const name = document.getElementById('lora-adapter-name').value.trim();
            const datasetId = document.getElementById('lora-dataset-select').value;
            if (!name) { showToast('Enter an adapter name.', 'warning'); return; }
            if (!datasetId) { showToast('Select a dataset.', 'warning'); return; }

            const request = {
                name: name,
                dataset_id: datasetId,
                epochs: parseInt(document.getElementById('lora-epochs').value) || 5,
                lr: parseFloat(document.getElementById('lora-lr').value) || 5e-6,
                batch_size: parseInt(document.getElementById('lora-batch-size').value) || 1,
                lora_r: parseInt(document.getElementById('lora-rank').value) || 32,
                lora_alpha: parseInt(document.getElementById('lora-alpha').value) || 128,
                gradient_accumulation_steps: parseInt(document.getElementById('lora-grad-accum').value) || 8,
                language: document.getElementById('lora-language').value || 'english'
            };

            const btn = document.getElementById('btn-lora-train');
            btn.disabled = true;
            document.getElementById('lora-train-status').innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Starting...';

            try {
                const result = await API.post('/api/lora/train', request);
                document.getElementById('btn-lora-cancel').style.display = 'inline-block';
                document.getElementById('lora-progress-section').style.display = 'block';
                document.getElementById('lora-train-status').innerHTML = '<span class="text-info">Training in progress...</span>';
                pollLoraTraining(request.epochs);
            } catch (e) {
                showToast('Failed to start training: ' + e.message, 'error');
                btn.disabled = false;
                document.getElementById('lora-train-status').innerHTML = '';
            }
        };

        function pollLoraTraining(totalEpochs) {
            const logsEl = document.getElementById('lora-train-logs');
            const progressBar = document.getElementById('lora-progress-bar');
            const epochDisplay = document.getElementById('lora-epoch-display');
            const lossDisplay = document.getElementById('lora-loss-display');

            _startPolling('lora_training', () => API.get('/api/status/lora_training'), {
                intervalMs: 2000,
                doneCheck: status => !status.running,
                onTick: status => {
                    logsEl.innerText = status.logs.join('\n');
                    logsEl.scrollTop = logsEl.scrollHeight;

                    // Parse latest metrics from log lines
                    for (let i = status.logs.length - 1; i >= 0; i--) {
                        const line = status.logs[i];
                        const epochMatch = line.match(/\[EPOCH\]\s*(\d+)\/(\d+)\s+avg_loss=([\d.]+)/);
                        if (epochMatch) {
                            const epoch = parseInt(epochMatch[1]);
                            const maxEpoch = parseInt(epochMatch[2]);
                            const loss = epochMatch[3];
                            const pct = Math.round((epoch / maxEpoch) * 100);
                            epochDisplay.innerText = `${epoch}/${maxEpoch}`;
                            lossDisplay.innerText = loss;
                            progressBar.style.width = `${pct}%`;
                            progressBar.innerText = `${pct}%`;
                            break;
                        }
                        const trainMatch = line.match(/\[TRAIN\]\s*epoch=(\d+)\/(\d+)\s+step=\d+\/\d+\s+loss=([\d.]+)/);
                        if (trainMatch) {
                            const epoch = parseInt(trainMatch[1]);
                            const maxEpoch = parseInt(trainMatch[2]);
                            const loss = trainMatch[3];
                            const pct = Math.round(((epoch - 1) / maxEpoch) * 100);
                            epochDisplay.innerText = `${epoch}/${maxEpoch}`;
                            lossDisplay.innerText = loss;
                            progressBar.style.width = `${pct}%`;
                            progressBar.innerText = `${pct}%`;
                            break;
                        }
                    }
                },
                onDone: status => {
                    notifyJobDone('lora_training');
                    const btn = document.getElementById('btn-lora-train');
                    btn.disabled = false;
                    const cancelBtn = document.getElementById('btn-lora-cancel');
                    cancelBtn.style.display = 'none';
                    cancelBtn.disabled = false;

                    const isDone = status.logs.some(l => l.includes('[DONE]'));
                    const isError = status.logs.some(l => l.includes('[ERROR]'));

                    if (isDone) {
                        document.getElementById('lora-train-status').innerHTML = '<span class="text-success"><i class="fas fa-check me-1"></i>Training complete!</span>';
                        progressBar.style.width = '100%';
                        progressBar.innerText = '100%';
                        progressBar.classList.remove('progress-bar-animated');
                        progressBar.classList.replace('bg-info', 'bg-success');
                        loadLoraModels();
                    } else if (isError) {
                        document.getElementById('lora-train-status').innerHTML = '<span class="text-danger"><i class="fas fa-times me-1"></i>Training failed</span>';
                        progressBar.classList.remove('progress-bar-animated');
                        progressBar.classList.replace('bg-info', 'bg-danger');
                    } else {
                        document.getElementById('lora-train-status').innerHTML = '<span class="text-warning">Training stopped</span>';
                    }
                }
            });
        }

        window.cancelLoraTraining = async () => {
            const btn = document.getElementById('btn-lora-cancel');
            btn.disabled = true;
            try {
                await API.post('/api/lora/train/cancel', {});
                document.getElementById('lora-train-status').innerHTML = '<span class="text-warning">Cancellation requested…</span>';
            } catch (e) {
                btn.disabled = false;
                showToast('Failed to cancel training: ' + e.message, 'error');
            }
        };

        async function loadLoraModels() {
            try {
                const models = await API.get('/api/lora/models');
                let backupStatus = { backups: [], total_size_bytes: 0, free_bytes: 0, low_space_warning: false };
                try {
                    backupStatus = await API.get('/api/lora/backups');
                } catch (e) {
                    console.debug('LoRA backup status unavailable', e);
                }
                models.forEach(model => {
                    model.rollback_backup = backupStatus.backups.find(
                        backup => backup.adapter_id === model.id) || null;
                });
                window._loraModelsCache = models;
                const container = document.getElementById('lora-models-list');
                const testForm = document.getElementById('lora-test-form');

                if (!models.length) {
                    container.innerHTML = '<p class="text-muted mb-0">No adapters available.</p>';
                    testForm.style.display = 'none';
                    return;
                }

                const backupSummary = backupStatus.backups.length || backupStatus.low_space_warning ? `
                    <div class="alert ${backupStatus.low_space_warning ? 'alert-warning' : 'alert-secondary'} py-2 mb-2 small">
                        ${backupStatus.low_space_warning ? '<strong>Low disk space.</strong> ' : ''}
                        Rollback backups: ${backupStatus.backups.length},
                        ${(backupStatus.total_size_bytes / 1024 ** 3).toFixed(2)} GB;
                        ${(backupStatus.free_bytes / 1024 ** 3).toFixed(1)} GB free.
                    </div>` : '';
                const renderCandidateSummary = model => {
                    const summary = model.candidate_summary;
                    if (!summary || summary.state === 'no_candidates') {
                        return '';
                    }
                    const labels = {
                        awaiting_evaluation: 'awaiting evaluation',
                        candidate_recommended: 'candidate recommended',
                        production_recommended: 'production recommended',
                        promoted: 'candidate promoted',
                        rolled_back: 'promotion rolled back',
                    };
                    const counts = [];
                    if (summary.evaluated_count) {
                        counts.push(`${summary.evaluated_count} evaluated`);
                    }
                    if (summary.retained_count) {
                        counts.push(`${summary.retained_count} retained`);
                    }
                    if (summary.duplicate_count) {
                        counts.push(`${summary.duplicate_count} duplicate skipped`);
                    }
                    const unchanged = summary.production_unchanged ? ' · production unchanged' : '';
                    return `<div class="text-muted small mt-1">${escapeHtml(labels[summary.state] || summary.state)}${counts.length ? ` · ${escapeHtml(counts.join(', '))}` : ''}${unchanged}</div>`;
                };
                // Human review tally, kept visually distinct from the automated
                // recommendation above (the "human" icon vs the evaluation badge).
                const renderReviewSummary = model => {
                    const r = model.review_summary;
                    if (!r || !r.count) { return ''; }
                    const parts = [];
                    if (r.preferred_candidate) { parts.push(`${r.preferred_candidate} prefer candidate`); }
                    if (r.preferred_production) { parts.push(`${r.preferred_production} prefer production`); }
                    if (r.tie) { parts.push(`${r.tie} no preference`); }
                    return `<div class="text-muted small mt-1"><i class="fas fa-user me-1"></i>${r.count} human review${r.count === 1 ? '' : 's'}${parts.length ? ` · ${escapeHtml(parts.join(', '))}` : ''}</div>`;
                };
                container.innerHTML = `${backupSummary}
                    <table class="table table-sm table-hover mb-0">
                        <thead><tr><th>Name</th><th>Dataset</th><th>Epochs</th><th>Final Loss</th><th>Evaluation</th><th>Samples</th><th style="width:240px">Actions</th></tr></thead>
                        <tbody>
                            ${models.map(m => `
                                <tr${m.builtin ? ' class="table-light"' : ''}>
                                    <td><strong>${escapeHtml(m.name)}</strong>${m.builtin ? ` <span class="badge bg-secondary">built-in</span>${m.downloaded === false ? ' <span class="badge bg-warning text-dark">not downloaded</span>' : ''}` : ''}</td>
                                    <td>${escapeHtml(m.dataset_id || (m.builtin ? '--' : '--'))}</td>
                                    <td>${m.epochs || '--'}</td>
                                    <td>${m.final_loss != null ? m.final_loss.toFixed(4) : '--'}</td>
                                    <td>${m.checkpoint_swap ? `<span class="badge bg-danger">recovery required</span>` : m.evaluation ? `<span class="badge ${m.evaluation.status === 'pass' ? 'bg-success' : m.evaluation.status === 'warning' ? 'bg-warning text-dark' : 'bg-danger'}" title="${escapeHtml((m.evaluation.warnings || []).join(', '))}">${escapeHtml(m.evaluation.status || 'unknown')}</span>${m.evaluation.recommended_candidate && m.evaluation.recommended_candidate !== 'production' ? ` <small title="Production remains unchanged">recommend ${escapeHtml(m.evaluation.recommended_candidate)}</small>` : ''}` : '--'}${renderCandidateSummary(m)}${renderReviewSummary(m)}</td>
                                    <td title="Dataset quantity only; not a voice-quality score">${m.sample_count || '--'}</td>
                                    <td>
                                        ${m.builtin && m.downloaded === false ? `
                                            <button class="btn btn-sm btn-outline-warning" id="lora-dl-btn-${escapeHtml(m.id)}" data-adapter-id="${escapeHtml(m.id)}" onclick="downloadBuiltinAdapter(this.dataset.adapterId)" title="Download from HuggingFace"><i class="fas fa-download me-1"></i>Download</button>
                                        ` : `
                                            <button class="btn btn-sm ${m.preview_audio_url ? 'btn-outline-success' : 'btn-outline-secondary'} me-1" id="lora-preview-btn-${escapeHtml(m.id)}" data-adapter-id="${escapeHtml(m.id)}" onclick="playLoraPreview(this.dataset.adapterId)" title="${m.preview_audio_url ? 'Play preview' : 'Generate and play preview (first time may take a moment)'}"><i class="fas fa-volume-up"></i></button>
                                            <button class="btn btn-sm btn-outline-primary me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="testLoraModel(this.dataset.adapterId)" title="Generate test line with custom text"><i class="fas fa-flask me-1"></i>Test</button>
                                            ${!m.builtin && m.checkpoint_swap ? `<button class="btn btn-sm btn-danger me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="recoverLoraCheckpointSwap(this.dataset.adapterId)" title="Restore production from the interrupted operation journal"><i class="fas fa-life-ring me-1"></i>Recover</button>` : ''}
                                            ${!m.builtin && !m.checkpoint_swap && m.evaluation?.recommended_candidate && m.evaluation.recommended_candidate !== 'production' ? `<button class="btn btn-sm btn-outline-info me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="openLoraCandidateComparison(this.dataset.adapterId)" title="Listen to matched production and candidate evaluation probes"><i class="fas fa-headphones me-1"></i>Compare</button><button class="btn btn-sm btn-outline-primary me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="openLoraBlindReview(this.dataset.adapterId)" title="Blind A/B listening review — identities hidden until you submit; never promotes"><i class="fas fa-user-secret me-1"></i>Blind review</button><button class="btn btn-sm btn-outline-secondary me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="openLoraReviewHistory(this.dataset.adapterId)" title="Human review history for this adapter"><i class="fas fa-clock-rotate-left me-1"></i>History</button><button class="btn btn-sm btn-outline-success me-1" data-adapter-id="${escapeHtml(m.id)}" data-candidate-id="${escapeHtml(m.evaluation.recommended_candidate)}" onclick="promoteLoraCandidate(this.dataset.adapterId, this.dataset.candidateId)" title="Preserve production, then promote this evaluated candidate"><i class="fas fa-arrow-up me-1"></i>Promote</button>` : ''}
                                            ${!m.builtin && !m.checkpoint_swap && m.promotion?.status === 'promoted' ? `<button class="btn btn-sm btn-outline-warning me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="rollbackLoraPromotion(this.dataset.adapterId)" title="Restore the production checkpoint saved before promotion"><i class="fas fa-undo me-1"></i>Rollback</button>` : ''}
                                            ${!m.builtin && !m.checkpoint_swap && m.rollback_backup ? `<button class="btn btn-sm btn-outline-danger me-1" data-adapter-id="${escapeHtml(m.id)}" onclick="deleteLoraRollbackBackup(this.dataset.adapterId)" title="Delete ${(m.rollback_backup.size_bytes / 1024 ** 2).toFixed(1)} MB rollback backup"><i class="fas fa-hard-drive me-1"></i>Delete backup</button>` : ''}
                                            ${m.builtin ? '' : `<button class="btn btn-sm btn-outline-danger" data-adapter-id="${escapeHtml(m.id)}" onclick="deleteLoraModel(this.dataset.adapterId)" title="Delete"><i class="fas fa-trash"></i></button>`}
                                        `}
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>`;

                // Populate test dropdown
                const dropdown = document.getElementById('lora-test-adapter');
                const prevVal = dropdown.value;
                dropdown.innerHTML = models.filter(m => m.downloaded !== false).map(m =>
                    `<option value="${escapeHtml(m.id)}">${escapeHtml(m.name)}</option>`
                ).join('');
                if (prevVal && models.some(m => m.id === prevVal)) { dropdown.value = prevVal; }
                testForm.style.display = '';
            } catch (e) {
                console.error('Failed to load LoRA models:', e);
            }
        }

        window.openLoraCandidateComparison = async (adapterId) => {
            const panel = document.getElementById('lora-comparison-panel');
            panel.style.display = '';
            panel.innerHTML = '<div class="text-muted small"><i class="fas fa-spinner fa-spin me-1"></i>Loading comparison…</div>';
            try {
                const comparison = await API.get(`/api/lora/models/${encodeURIComponent(adapterId)}/comparison`);
                const renderMetrics = probe => {
                    const metrics = probe.metrics || {};
                    const values = [
                        ['speaker similarity', metrics.speaker_similarity],
                        ['clipping', metrics.clipping_ratio],
                        ['silence', metrics.silence_ratio],
                    ].filter(item => item[1] != null);
                    return values.length ? values.map(item => `${escapeHtml(item[0])}: ${Number(item[1]).toFixed(3)}`).join(' · ') : 'No metrics recorded';
                };
                panel.innerHTML = `
                    <div class="card border-info">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <div><strong>Candidate comparison: ${escapeHtml(comparison.candidate_id)}</strong><br><small class="text-muted">Advisory only — listening does not change production.</small></div>
                            <button class="btn btn-sm btn-outline-secondary" onclick="document.getElementById('lora-comparison-panel').style.display='none'" title="Close"><i class="fas fa-times"></i></button>
                        </div>
                        <div class="card-body">
                            ${comparison.reason ? `<p class="small mb-3">${escapeHtml(comparison.reason)}</p>` : ''}
                            ${comparison.probe_pairs.map(pair => `
                                <div class="border rounded p-2 mb-2">
                                    <div class="small mb-2"><strong>${escapeHtml(pair.id)}</strong> · seed ${escapeHtml(String(pair.seed))}<br>${escapeHtml(pair.text)}</div>
                                    <div class="row g-2">
                                        <div class="col-md-6"><label class="form-label small mb-1">Production</label><audio controls preload="none" class="w-100" src="${escapeHtml(pair.production.audio_url)}"></audio><div class="text-muted small">${renderMetrics(pair.production)}</div></div>
                                        <div class="col-md-6"><label class="form-label small mb-1">Candidate</label><audio controls preload="none" class="w-100" src="${escapeHtml(pair.candidate.audio_url)}"></audio><div class="text-muted small">${renderMetrics(pair.candidate)}</div></div>
                                    </div>
                                </div>`).join('')}
                        </div>
                    </div>`;
                panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (e) {
                panel.innerHTML = `<div class="alert alert-danger py-2 mb-0">Comparison unavailable: ${escapeHtml(e.message)}</div>`;
            }
        };

        // Blind A/B human review. Identities are hidden by the server until a
        // decision is submitted; this never calls promote (Rule: human feedback
        // never auto-promotes). Reuses the comparison panel area.
        let _blindReview = null;

        function _blindReviewSelectedChoice() {
            const checked = document.querySelector('input[name="blind-pref"]:checked');
            return checked ? checked.value : null;
        }

        window.openLoraBlindReview = async (adapterId) => {
            const panel = document.getElementById('lora-comparison-panel');
            panel.style.display = '';
            panel.innerHTML = '<div class="text-muted small"><i class="fas fa-spinner fa-spin me-1"></i>Opening blind review…</div>';
            try {
                const session = await API.post(`/api/lora/models/${encodeURIComponent(adapterId)}/review/session`, {});
                _blindReview = { adapterId: adapterId, sessionId: session.session_id };
                const pairsHtml = session.pairs.map(pair => `
                    <div class="border rounded p-2 mb-2">
                        <div class="small mb-2"><strong>${escapeHtml(pair.id)}</strong><br>${escapeHtml(pair.text)}</div>
                        <div class="row g-2">
                            <div class="col-md-6"><label class="form-label small mb-1">Sample A</label><audio controls preload="none" class="w-100" src="${escapeHtml(pair.A.audio_url)}"></audio></div>
                            <div class="col-md-6"><label class="form-label small mb-1">Sample B</label><audio controls preload="none" class="w-100" src="${escapeHtml(pair.B.audio_url)}"></audio></div>
                        </div>
                    </div>`).join('');
                panel.innerHTML = `
                    <div class="card border-primary">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <div><strong>Blind review</strong><br><small class="text-muted">Identities are hidden until you submit. This never changes production.</small></div>
                            <button class="btn btn-sm btn-outline-secondary" onclick="document.getElementById('lora-comparison-panel').style.display='none'" title="Close"><i class="fas fa-times"></i></button>
                        </div>
                        <div class="card-body">
                            ${pairsHtml}
                            <div class="mb-2">
                                <label class="form-label small mb-1">Which sounds better?</label>
                                <div class="d-flex gap-3">
                                    <div class="form-check"><input class="form-check-input" type="radio" name="blind-pref" id="blind-pref-a" value="A"><label class="form-check-label small" for="blind-pref-a">Sample A</label></div>
                                    <div class="form-check"><input class="form-check-input" type="radio" name="blind-pref" id="blind-pref-b" value="B"><label class="form-check-label small" for="blind-pref-b">Sample B</label></div>
                                    <div class="form-check"><input class="form-check-input" type="radio" name="blind-pref" id="blind-pref-tie" value="tie"><label class="form-check-label small" for="blind-pref-tie">No preference</label></div>
                                </div>
                            </div>
                            <div class="row g-2 mb-2">
                                <div class="col-md-4"><label class="form-label small mb-1">Rating (optional)</label>
                                    <select class="form-select form-select-sm" id="blind-rating"><option value="">—</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div>
                                <div class="col-md-8"><label class="form-label small mb-1">Notes (optional)</label>
                                    <input type="text" class="form-control form-control-sm" id="blind-notes" maxlength="1000" placeholder="What stood out?"></div>
                            </div>
                            <button id="btn-blind-submit" class="btn btn-sm btn-primary" onclick="submitLoraBlindReview()"><i class="fas fa-check me-1"></i>Submit decision</button>
                        </div>
                    </div>`;
                panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } catch (e) {
                panel.innerHTML = `<div class="alert alert-danger py-2 mb-0">Blind review unavailable: ${escapeHtml(e.message)}</div>`;
            }
        };

        window.submitLoraBlindReview = async () => {
            if (!_blindReview) { return; }
            const choice = _blindReviewSelectedChoice();
            if (!choice) { showToast('Pick Sample A, Sample B, or No preference.', 'warning'); return; }
            const ratingRaw = document.getElementById('blind-rating').value;
            const body = {
                choice: choice,
                rating: ratingRaw ? parseInt(ratingRaw, 10) : null,
                notes: document.getElementById('blind-notes').value || '',
            };
            // Disable on click so a double-click can't fire two submissions for
            // the same session; the backend also guards this, but not sending
            // the second request is cleaner. Re-enable only on error.
            const submitBtn = document.getElementById('btn-blind-submit');
            if (submitBtn) { submitBtn.disabled = true; }
            try {
                const result = await API.post(
                    `/api/lora/models/${encodeURIComponent(_blindReview.adapterId)}/review/session/${encodeURIComponent(_blindReview.sessionId)}`, body);
                renderBlindReviewResult(result);
            } catch (e) {
                if (submitBtn) { submitBtn.disabled = false; }
                showToast('Could not record review: ' + e.message, 'error');
            }
        };

        // Reveal identities AFTER submission. Human preference and the automated
        // recommendation are shown as separate, clearly labelled facts.
        function renderBlindReviewResult(result) {
            const panel = document.getElementById('lora-comparison-panel');
            const labels = result.revealed.labels || {};
            const role = result.revealed.choice_role;
            const yourPick = role === 'tie' ? 'No preference'
                : `the <strong>${escapeHtml(role)}</strong> checkpoint`;
            const automated = (result.automated && result.automated.recommended_candidate) || 'none';
            panel.innerHTML = `
                <div class="card border-success">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <strong>Review recorded</strong>
                        <button class="btn btn-sm btn-outline-secondary" onclick="document.getElementById('lora-comparison-panel').style.display='none'" title="Close"><i class="fas fa-times"></i></button>
                    </div>
                    <div class="card-body small">
                        <div class="mb-1">Sample A was <strong>${escapeHtml(labels.A || '?')}</strong>, Sample B was <strong>${escapeHtml(labels.B || '?')}</strong>.</div>
                        <div class="mb-1"><i class="fas fa-user me-1"></i>Your preference: ${yourPick}.</div>
                        <div class="text-muted"><i class="fas fa-robot me-1"></i>Automated recommendation (separate): ${escapeHtml(automated)}. Human feedback does not change production.</div>
                    </div>
                </div>`;
            _blindReview = null;
        }

        window.openLoraReviewHistory = async (adapterId) => {
            const panel = document.getElementById('lora-comparison-panel');
            panel.style.display = '';
            panel.innerHTML = '<div class="text-muted small"><i class="fas fa-spinner fa-spin me-1"></i>Loading review history…</div>';
            try {
                const data = await API.get(`/api/lora/models/${encodeURIComponent(adapterId)}/reviews`);
                renderLoraReviewHistory(adapterId, data.reviews || []);
            } catch (e) {
                panel.innerHTML = `<div class="alert alert-danger py-2 mb-0">History unavailable: ${escapeHtml(e.message)}</div>`;
            }
        };

        function renderLoraReviewHistory(adapterId, reviews) {
            const panel = document.getElementById('lora-comparison-panel');
            const rows = reviews.map(r => {
                const when = r.created_at ? new Date(r.created_at).toLocaleString() : '';
                const human = r.human || {};
                const rating = human.rating ? ` · ${human.rating}/5` : '';
                const notes = human.notes ? ` · ${escapeHtml(human.notes)}` : '';
                const auto = (r.automated || {}).recommended_candidate || '—';
                return `<div class="border rounded p-2 mb-1 small">
                    <div><i class="fas fa-user me-1"></i>Preferred: <strong>${escapeHtml(human.choice_role || '?')}</strong>${rating}${notes}</div>
                    <div class="text-muted"><i class="fas fa-robot me-1"></i>Automated: ${escapeHtml(auto)} · ${escapeHtml(when)}${r.blind ? ' · blind' : ''}</div>
                </div>`;
            }).join('');
            panel.innerHTML = `
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <strong>Review history (${reviews.length})</strong>
                        <span>
                            ${reviews.length ? `<button class="btn btn-sm btn-outline-danger me-1" data-adapter-id="${escapeHtml(adapterId)}" onclick="clearLoraReviewHistory(this.dataset.adapterId)"><i class="fas fa-trash me-1"></i>Clear</button>` : ''}
                            <button class="btn btn-sm btn-outline-secondary" onclick="document.getElementById('lora-comparison-panel').style.display='none'" title="Close"><i class="fas fa-times"></i></button>
                        </span>
                    </div>
                    <div class="card-body">${rows || '<div class="text-muted small">No human reviews recorded yet.</div>'}</div>
                </div>`;
        }

        window.clearLoraReviewHistory = async (adapterId) => {
            if (!confirm('Delete all human review history for this adapter?')) { return; }
            try {
                const result = await API.post(`/api/lora/models/${encodeURIComponent(adapterId)}/reviews/cleanup`, {});
                const freedKb = (result.freed_bytes / 1024).toFixed(1);
                showToast(`Cleared ${result.removed_count} review(s), freed ${freedKb} KB.`, 'success');
                openLoraReviewHistory(adapterId);
            } catch (e) {
                showToast('Could not clear history: ' + e.message, 'error');
            }
        };

        window.promoteLoraCandidate = async (adapterId, candidateId) => {
            if (!confirm(`Promote ${candidateId}? The current production checkpoint will be preserved for rollback.`)) {
                return;
            }
            try {
                await API.post(`/api/lora/models/${encodeURIComponent(adapterId)}/promote`, {});
                showToast(`Promoted ${candidateId}. Production backup retained.`, 'success');
                await loadLoraModels();
            } catch (e) {
                showToast('Promotion failed: ' + e.message, 'error');
            }
        };

        window.rollbackLoraPromotion = async (adapterId) => {
            if (!confirm('Restore the production checkpoint saved before promotion?')) {
                return;
            }
            try {
                await API.post(`/api/lora/models/${encodeURIComponent(adapterId)}/rollback-promotion`, {});
                showToast('Previous production checkpoint restored.', 'success');
                await loadLoraModels();
            } catch (e) {
                showToast('Rollback failed: ' + e.message, 'error');
            }
        };

        window.recoverLoraCheckpointSwap = async (adapterId) => {
            if (!confirm('Recover production from the checkpoint saved before the interrupted operation?')) {
                return;
            }
            try {
                await API.post(`/api/lora/models/${encodeURIComponent(adapterId)}/recover-checkpoint-swap`, {});
                showToast('Interrupted checkpoint operation recovered.', 'success');
                await loadLoraModels();
            } catch (e) {
                showToast('Recovery failed: ' + e.message, 'error');
            }
        };

        window.deleteLoraRollbackBackup = async (adapterId) => {
            if (!confirm('Permanently delete this rollback backup? You will no longer be able to restore the pre-promotion checkpoint.')) {
                return;
            }
            try {
                const response = await fetch(`/api/lora/models/${encodeURIComponent(adapterId)}/rollback-backup`, { method: 'DELETE' });
                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || `HTTP ${response.status}`);
                }
                showToast('Rollback backup deleted.', 'success');
                await loadLoraModels();
            } catch (e) {
                showToast('Backup deletion failed: ' + e.message, 'error');
            }
        };

        window.playLoraPreview = async (adapterId) => {
            const btn = document.getElementById(`lora-preview-btn-${adapterId}`);
            const origHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

            try {
                const result = await API.post(`/api/lora/preview/${encodeURIComponent(adapterId)}`, {});
                const audio = new Audio(`${result.audio_url}?t=${Date.now()}`);
                audio.play();
                // Update button now that preview is cached
                btn.title = 'Play preview';
                btn.classList.replace('btn-outline-secondary', 'btn-outline-success');
            } catch (e) {
                showToast('Preview failed: ' + e.message, 'error');
            } finally {
                btn.disabled = false;
                btn.innerHTML = origHtml;
            }
        };

        window.testLoraModel = (adapterId) => {
            document.getElementById('lora-test-adapter').value = adapterId;
            document.getElementById('lora-test-form').style.display = '';
            document.getElementById('lora-test-text').focus();
        };

        window.runLoraTest = async () => {
            const adapterId = document.getElementById('lora-test-adapter').value;
            const text = document.getElementById('lora-test-text').value.trim();
            const instruct = document.getElementById('lora-test-instruct').value.trim();
            if (!adapterId) { showToast('Select an adapter.', 'warning'); return; }
            if (!text) { showToast('Enter text to synthesize.', 'warning'); return; }

            const statusEl = document.getElementById('lora-test-status');
            statusEl.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating...';

            try {
                const result = await API.post('/api/lora/test', {
                    adapter_id: adapterId,
                    text: text,
                    instruct: instruct
                });

                statusEl.innerHTML = '';
                const audioDiv = document.getElementById('lora-test-audio');
                audioDiv.innerHTML = `<audio controls autoplay src="${result.audio_url}?t=${Date.now()}"></audio>`;
            } catch (e) {
                statusEl.innerHTML = `<span class="text-danger">Failed: ${escapeHtml(e.message)}</span>`;
            }
        };

        window.deleteLoraModel = async (adapterId) => {
            if (!await showConfirm('Delete this trained adapter? This cannot be undone.')) { return; }
            try {
                const res = await fetch(`/api/lora/models/${encodeURIComponent(adapterId)}`, { method: 'DELETE' });
                if (!res.ok) { const err = await res.json(); showToast(err.detail || 'Failed to delete.', 'error'); return; }
                loadLoraModels();
            } catch (e) {
                showToast('Error deleting adapter: ' + e.message, 'error');
            }
        };

        window.downloadBuiltinAdapter = async (adapterId) => {
            const btn = document.getElementById(`lora-dl-btn-${adapterId}`);
            const origHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Downloading...';

            try {
                await API.post(`/api/lora/download/${encodeURIComponent(adapterId)}`, {});
                showToast('Adapter downloaded successfully.', 'success');
                loadLoraModels();
            } catch (e) {
                showToast('Download failed: ' + e.message, 'error');
                btn.disabled = false;
                btn.innerHTML = origHtml;
            }
        };
