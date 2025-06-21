/**
 * AI Video Translation Service - Frontend Application
 */

class VideoTranslationApp {
    constructor() {
        this.currentJobId = null;
        this.websocket = null;
        this.pollInterval = null;
        this.currentFile = null;
        
        this.init();
    }

    /**
     * Initialize the application
     */
    init() {
        this.bindEvents();
        this.initializeUploadArea();
        
        // Load job from URL parameter if present
        const urlParams = new URLSearchParams(window.location.search);
        const jobId = urlParams.get('job_id');
        if (jobId) {
            this.loadJobStatus(jobId);
        }
    }

    /**
     * Bind all event listeners
     */
    bindEvents() {
        // Upload form events
        const uploadForm = document.getElementById('uploadForm');
        const videoFile = document.getElementById('videoFile');
        const removeFile = document.getElementById('removeFile');
        const uploadBtn = document.getElementById('uploadBtn');

        uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
        videoFile.addEventListener('change', (e) => this.handleFileSelect(e));
        removeFile.addEventListener('click', () => this.removeSelectedFile());

        // Advanced settings toggle
        const advancedToggle = document.getElementById('advancedToggle');
        advancedToggle.addEventListener('click', () => this.toggleAdvancedSettings());

        // Job status check
        const checkStatusBtn = document.getElementById('checkStatusBtn');
        const jobIdInput = document.getElementById('jobIdInput');
        checkStatusBtn.addEventListener('click', () => this.checkJobStatus());
        jobIdInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.checkJobStatus();
        });

        // Progress section events
        const copyJobId = document.getElementById('copyJobId');
        const newTranslationBtn = document.getElementById('newTranslationBtn');
        const cancelJobBtn = document.getElementById('cancelJobBtn');
        const previewBtn = document.getElementById('previewBtn');
        const downloadBtn = document.getElementById('downloadBtn');

        copyJobId.addEventListener('click', () => this.copyJobId());
        newTranslationBtn.addEventListener('click', () => this.startNewTranslation());
        cancelJobBtn.addEventListener('click', () => this.cancelJob());
        previewBtn.addEventListener('click', () => this.showPreview());
        downloadBtn.addEventListener('click', () => this.downloadVideo());

        // Form validation
        const targetLanguage = document.getElementById('targetLanguage');
        targetLanguage.addEventListener('change', () => this.validateForm());
        videoFile.addEventListener('change', () => this.validateForm());
    }

    /**
     * Initialize drag and drop upload area
     */
    initializeUploadArea() {
        const uploadArea = document.getElementById('uploadArea');
        const videoFile = document.getElementById('videoFile');

        // Click to browse
        uploadArea.addEventListener('click', () => videoFile.click());

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });
    }

    /**
     * Handle file selection
     */
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.handleFile(file);
        }
    }

    /**
     * Handle file processing
     */
    handleFile(file) {
        // Validate file type
        if (!file.type.includes('video/mp4') && !file.name.toLowerCase().endsWith('.mp4')) {
            this.showNotification('Please select an MP4 video file', 'error');
            return;
        }

        // Validate file size (200MB limit)
        const maxSize = 200 * 1024 * 1024; // 200MB in bytes
        if (file.size > maxSize) {
            this.showNotification('File size exceeds 200MB limit', 'error');
            return;
        }

        this.currentFile = file;
        this.displayFileInfo(file);
        this.validateForm();
    }

    /**
     * Display selected file information
     */
    displayFileInfo(file) {
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const fileInfo = document.getElementById('fileInfo');
        const uploadArea = document.getElementById('uploadArea');

        fileName.textContent = file.name;
        fileSize.textContent = this.formatFileSize(file.size);

        uploadArea.style.display = 'none';
        fileInfo.style.display = 'block';
    }

    /**
     * Remove selected file
     */
    removeSelectedFile() {
        const uploadArea = document.getElementById('uploadArea');
        const fileInfo = document.getElementById('fileInfo');
        const videoFile = document.getElementById('videoFile');

        this.currentFile = null;
        videoFile.value = '';
        uploadArea.style.display = 'block';
        fileInfo.style.display = 'none';
        this.validateForm();
    }

    /**
     * Toggle advanced settings
     */
    toggleAdvancedSettings() {
        const toggle = document.getElementById('advancedToggle');
        const content = document.getElementById('advancedContent');

        toggle.classList.toggle('active');
        content.classList.toggle('active');
    }

    /**
     * Validate upload form
     */
    validateForm() {
        const uploadBtn = document.getElementById('uploadBtn');
        const targetLanguage = document.getElementById('targetLanguage');

        const isValid = this.currentFile && targetLanguage.value;
        uploadBtn.disabled = !isValid;
    }

    /**
     * Handle form upload
     */
    async handleUpload(event) {
        event.preventDefault();

        if (!this.currentFile) {
            this.showNotification('Please select a video file', 'error');
            return;
        }

        const formData = new FormData();
        const form = document.getElementById('uploadForm');

        // Add file
        formData.append('file', this.currentFile);

        // Add form data
        const formDataObj = new FormData(form);
        for (let [key, value] of formDataObj.entries()) {
            if (key !== 'file') {
                formData.append(key, value);
            }
        }

        this.showLoading(true);

        try {
            const response = await fetch('/api/v1/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.showNotification('Upload successful! Starting translation...', 'success');
                this.currentJobId = result.job_id;
                this.showProgressSection();
                this.startProgressTracking();
                
                // Update URL with job ID
                const url = new URL(window.location);
                url.searchParams.set('job_id', result.job_id);
                history.pushState({}, '', url);
            } else {
                throw new Error(result.detail || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showNotification(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Check job status by job ID
     */
    async checkJobStatus() {
        const jobIdInput = document.getElementById('jobIdInput');
        const jobId = jobIdInput.value.trim();

        if (!jobId) {
            this.showNotification('Please enter a job ID', 'warning');
            return;
        }

        await this.loadJobStatus(jobId);
    }

    /**
     * Load job status
     */
    async loadJobStatus(jobId) {
        this.showLoading(true);

        try {
            const response = await fetch(`/api/v1/jobs/${jobId}/status`);
            const result = await response.json();

            if (response.ok) {
                this.currentJobId = jobId;
                this.updateJobDisplay(result);
                this.showProgressSection();
                
                // Start real-time tracking if job is still processing
                if (result.status === 'processing' || result.status === 'uploaded') {
                    this.startProgressTracking();
                }
                
                // Update URL
                const url = new URL(window.location);
                url.searchParams.set('job_id', jobId);
                history.pushState({}, '', url);
            } else {
                throw new Error(result.detail || 'Job not found');
            }
        } catch (error) {
            console.error('Status check error:', error);
            this.showNotification(`Failed to load job: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Show progress section
     */
    showProgressSection() {
        const uploadSection = document.getElementById('uploadSection');
        const statusSection = document.getElementById('statusSection');
        const progressSection = document.getElementById('progressSection');

        uploadSection.style.display = 'none';
        statusSection.style.display = 'none';
        progressSection.style.display = 'block';

        // Scroll to progress section
        progressSection.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Start progress tracking
     */
    startProgressTracking() {
        // Try WebSocket first, fallback to polling
        this.connectWebSocket();
        
        // Start polling as backup
        this.startPolling();
    }

    /**
     * Connect WebSocket for real-time updates
     */
    connectWebSocket() {
        if (this.websocket) {
            this.websocket.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/v1/jobs/${this.currentJobId}/progress`;

        try {
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.stopPolling(); // Stop polling when WebSocket is connected
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleProgressUpdate(data);
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.startPolling(); // Fallback to polling
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                // Only restart polling if job is still processing
                if (this.isJobProcessing()) {
                    this.startPolling();
                }
            };
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.startPolling();
        }
    }

    /**
     * Handle progress updates
     */
    handleProgressUpdate(data) {
        if (data.type === 'progress_update') {
            const progress = data.data;
            this.updateProgress(progress.percentage, progress.message);
            
            // Update job status if changed
            if (progress.status) {
                this.updateJobStatus(progress.status);
            }
        } else if (data.type === 'ping') {
            // Respond to ping
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.websocket.send(JSON.stringify({ type: 'pong' }));
            }
        }
    }

    /**
     * Start polling for status updates
     */
    startPolling() {
        this.stopPolling();
        
        this.pollInterval = setInterval(async () => {
            try {
                const response = await fetch(`/api/v1/jobs/${this.currentJobId}/status`);
                const result = await response.json();
                
                if (response.ok) {
                    this.updateJobDisplay(result);
                    
                    // Stop polling if job is complete
                    if (!this.isJobProcessing(result.status)) {
                        this.stopPolling();
                        if (this.websocket) {
                            this.websocket.close();
                        }
                    }
                }
            } catch (error) {
                console.error('Polling error:', error);
            }
        }, 3000); // Poll every 3 seconds
    }

    /**
     * Stop polling
     */
    stopPolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }

    /**
     * Check if job is still processing
     */
    isJobProcessing(status = null) {
        const currentStatus = status || document.getElementById('jobStatus').textContent.toLowerCase();
        return ['uploaded', 'processing'].includes(currentStatus);
    }

    /**
     * Update job display
     */
    updateJobDisplay(jobData) {
        // Update job ID display
        document.getElementById('currentJobId').textContent = jobData.job_id;

        // Update job details
        document.getElementById('jobFileName').textContent = jobData.original_filename;
        document.getElementById('jobSourceLang').textContent = this.getLanguageName(jobData.source_language) || 'Auto-detect';
        document.getElementById('jobTargetLang').textContent = this.getLanguageName(jobData.target_language);
        document.getElementById('jobCreated').textContent = this.formatDate(jobData.created_at);
        
        // Update status
        this.updateJobStatus(jobData.status);
        
        // Update progress
        const percentage = jobData.progress_percentage || 0;
        const stage = jobData.progress_stage || jobData.status;
        this.updateProgress(percentage, this.getStatusMessage(jobData.status, stage));
        
        // Update duration
        this.updateDuration(jobData.created_at, jobData.completed_at);
        
        // Handle different status states
        this.handleJobStatusChange(jobData);
    }

    /**
     * Update job status badge
     */
    updateJobStatus(status) {
        const statusElement = document.getElementById('jobStatus');
        statusElement.textContent = status.toUpperCase();
        statusElement.className = `status-badge ${status}`;
    }

    /**
     * Update progress bar
     */
    updateProgress(percentage, message) {
        const progressFill = document.getElementById('progressFill');
        const progressPercent = document.getElementById('progressPercent');
        const progressStatus = document.getElementById('progressStatus');

        progressFill.style.width = `${percentage}%`;
        progressPercent.textContent = `${Math.round(percentage)}%`;
        progressStatus.textContent = message;
    }

    /**
     * Handle job status changes
     */
    handleJobStatusChange(jobData) {
        const errorMessage = document.getElementById('errorMessage');
        const successActions = document.getElementById('successActions');
        const cancelJobBtn = document.getElementById('cancelJobBtn');

        // Hide all status-specific elements
        errorMessage.style.display = 'none';
        successActions.style.display = 'none';
        cancelJobBtn.style.display = 'none';

        switch (jobData.status) {
            case 'uploaded':
            case 'processing':
                cancelJobBtn.style.display = 'inline-flex';
                break;
                
            case 'completed':
                successActions.style.display = 'block';
                this.showNotification('Translation completed successfully!', 'success');
                break;
                
            case 'failed':
                errorMessage.style.display = 'block';
                document.getElementById('errorText').textContent = jobData.error_message || 'Unknown error occurred';
                this.showNotification('Translation failed', 'error');
                break;
                
            case 'cancelled':
                this.showNotification('Job was cancelled', 'warning');
                break;
        }
    }

    /**
     * Copy job ID to clipboard
     */
    async copyJobId() {
        const jobId = document.getElementById('currentJobId').textContent;
        
        try {
            await navigator.clipboard.writeText(jobId);
            this.showNotification('Job ID copied to clipboard', 'success');
        } catch (error) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = jobId;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showNotification('Job ID copied to clipboard', 'success');
        }
    }

    /**
     * Start new translation
     */
    startNewTranslation() {
        // Reset state
        this.currentJobId = null;
        this.currentFile = null;
        this.stopPolling();
        if (this.websocket) {
            this.websocket.close();
        }

        // Show upload section
        const uploadSection = document.getElementById('uploadSection');
        const statusSection = document.getElementById('statusSection');
        const progressSection = document.getElementById('progressSection');
        const previewSection = document.getElementById('previewSection');

        uploadSection.style.display = 'block';
        statusSection.style.display = 'block';
        progressSection.style.display = 'none';
        previewSection.style.display = 'none';

        // Reset form
        document.getElementById('uploadForm').reset();
        this.removeSelectedFile();

        // Clear URL parameter
        const url = new URL(window.location);
        url.searchParams.delete('job_id');
        history.pushState({}, '', url);

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    /**
     * Cancel current job
     */
    async cancelJob() {
        if (!this.currentJobId) return;

        if (!confirm('Are you sure you want to cancel this translation job?')) {
            return;
        }

        this.showLoading(true);

        try {
            const response = await fetch(`/api/v1/jobs/${this.currentJobId}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (response.ok) {
                this.showNotification('Job cancelled successfully', 'success');
                await this.loadJobStatus(this.currentJobId); // Refresh status
            } else {
                throw new Error(result.detail || 'Failed to cancel job');
            }
        } catch (error) {
            console.error('Cancel error:', error);
            this.showNotification(`Failed to cancel job: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Show preview
     */
    async showPreview() {
        if (!this.currentJobId) return;

        this.showLoading(true);

        try {
            const response = await fetch(`/api/v1/jobs/${this.currentJobId}/preview`);
            const result = await response.json();

            if (response.ok) {
                this.displayPreview(result);
            } else {
                throw new Error(result.detail || 'Failed to load preview');
            }
        } catch (error) {
            console.error('Preview error:', error);
            this.showNotification(`Failed to load preview: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Display video preview
     */
    displayPreview(previewData) {
        const previewSection = document.getElementById('previewSection');
        const videoContainer = document.getElementById('videoContainer');
        const previewInfo = document.getElementById('previewInfo');

        // For now, show preview information
        // In a full implementation, this would include video thumbnails and clips
        previewInfo.innerHTML = `
            <h4>Preview Information</h4>
            <div style="margin-top: 15px;">
                <div><strong>File:</strong> ${previewData.original_filename}</div>
                <div><strong>Size:</strong> ${previewData.file_size_mb} MB</div>
                <div><strong>Source:</strong> ${this.getLanguageName(previewData.source_language)}</div>
                <div><strong>Target:</strong> ${this.getLanguageName(previewData.target_language)}</div>
                <div><strong>Status:</strong> ${previewData.status}</div>
            </div>
            <div style="margin-top: 15px; padding: 10px; background: #f3f4f6; border-radius: 8px; font-size: 0.9rem;">
                ${previewData.preview_note}
            </div>
        `;

        previewSection.style.display = 'block';
        previewSection.scrollIntoView({ behavior: 'smooth' });
    }

    /**
     * Download translated video
     */
    async downloadVideo() {
        if (!this.currentJobId) return;

        try {
            // Create download link
            const downloadUrl = `/api/v1/jobs/${this.currentJobId}/download`;
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = `translated_video_${this.currentJobId}.mp4`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);

            this.showNotification('Download started', 'success');
        } catch (error) {
            console.error('Download error:', error);
            this.showNotification('Download failed', 'error');
        }
    }

    /**
     * Update duration display
     */
    updateDuration(startTime, endTime) {
        const durationElement = document.getElementById('jobDuration');
        
        if (endTime) {
            const start = new Date(startTime);
            const end = new Date(endTime);
            const duration = Math.round((end - start) / 1000);
            durationElement.textContent = this.formatDuration(duration);
        } else {
            const start = new Date(startTime);
            const now = new Date();
            const duration = Math.round((now - start) / 1000);
            durationElement.textContent = this.formatDuration(duration);
        }
    }

    /**
     * Get language name from code
     */
    getLanguageName(code) {
        const languages = {
            'eng': 'English',
            'spa': 'Spanish',
            'fra': 'French',
            'deu': 'German',
            'ita': 'Italian',
            'por': 'Portuguese',
            'jpn': 'Japanese',
            'kor': 'Korean',
            'cmn': 'Chinese (Mandarin)',
            'hin': 'Hindi'
        };
        return languages[code] || code;
    }

    /**
     * Get status message
     */
    getStatusMessage(status, stage) {
        const messages = {
            'uploaded': 'File uploaded, waiting to start...',
            'processing': 'Processing video translation...',
            'completed': 'Translation completed successfully',
            'failed': 'Translation failed',
            'cancelled': 'Translation cancelled'
        };
        return messages[status] || stage || status;
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    /**
     * Format date
     */
    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleString();
    }

    /**
     * Format duration
     */
    formatDuration(seconds) {
        if (seconds < 60) {
            return `${seconds}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }

    /**
     * Show loading overlay
     */
    showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.display = show ? 'flex' : 'none';
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        const container = document.getElementById('notificationContainer');
        
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const icon = this.getNotificationIcon(type);
        notification.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);
    }

    /**
     * Get notification icon
     */
    getNotificationIcon(type) {
        const icons = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };
        return icons[type] || icons.info;
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VideoTranslationApp();
});

// Handle page visibility changes to manage WebSocket connections
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is hidden, we might want to pause some operations
    } else {
        // Page is visible again, resume operations if needed
    }
}); 