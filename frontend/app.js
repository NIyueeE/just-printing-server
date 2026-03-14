        // 全局状态
        const state = {
            authToken: localStorage.getItem('auth_token'),
            uploadedFiles: [],
            printSettings: {
                copies: 1,
                sides: 'one-sided',
                color_mode: 'monochrome',
                media: 'iso-a4',
                print_quality: 'normal',
                orientation: 'portrait',
                resolution: ''
            },
            supportedDocumentFormats: ['image/jpeg', 'image/png', 'application/pdf'],  // 支持的文档格式
            capabilities: null,  // 打印机能力
            printerStatus: 'unknown',
            printerStatusInterval: null,
            isMobile: window.innerWidth < 768
        };

        // ==================== 工具函数 ====================

        // Toast通知系统
        function showToast(message, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            const colors = {
                success: { bg: 'rgba(141, 161, 1, 0.12)', border: 'var(--success)', icon: '✓' },
                error: { bg: 'rgba(193, 74, 74, 0.12)', border: 'var(--error)', icon: '✕' },
                info: { bg: 'rgba(10, 125, 140, 0.12)', border: 'var(--accent)', icon: 'ℹ' }
            };
            const style = colors[type] || colors.info;
            toast.className = 'p-4 shadow-lg flex items-center transition-smooth';
            toast.style.cssText = `background: #fff; border-left: 5px solid ${style.border}; border-radius: 14px; color: var(--fg0); animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1); box-shadow: 0 8px 24px rgba(0,0,0,0.1);`;
            toast.innerHTML = `
                <span class="flex items-center justify-center w-6 h-6 mr-3 text-sm font-bold" style="background: ${style.bg}; border-radius: 50%; color: ${style.border};">${style.icon}</span>
                <span class="flex-1 font-medium">${message}</span>
                <button class="ml-4 p-1 transition" style="color: var(--text-muted); border-radius: 6px;" onmouseover="this.style.background='var(--bg2)'" onmouseout="this.style.background='transparent'">✕</button>
            `;
            container.appendChild(toast);

            // 自动消失
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.style.opacity = '0';
                    toast.style.transform = 'translateX(100%)';
                    toast.style.transition = 'all 0.3s ease';
                    setTimeout(() => toast.remove(), 300);
                }
            }, 3500);

            // 手动关闭
            toast.querySelector('button').addEventListener('click', () => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100%)';
                setTimeout(() => toast.remove(), 300);
            });
        };

        // 加载遮罩控制
        function showLoading(text = '处理中...') {
            const overlay = document.getElementById('loading-overlay');
            const textEl = document.getElementById('loading-text');
            textEl.textContent = text;
            overlay.classList.remove('hidden');
            overlay.classList.add('flex');
        }

        function hideLoading() {
            const overlay = document.getElementById('loading-overlay');
            overlay.classList.add('hidden');
            overlay.classList.remove('flex');
        }

        // API请求封装
        async function apiRequest(endpoint, options = {}) {
            const { method = 'GET', body = null, headers = {}, withAuth = true, timeout = 60000 } = options;

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const requestHeaders = { ...headers };
            if (withAuth && state.authToken) {
                requestHeaders['Authorization'] = `Bearer ${state.authToken}`;
            }

            if (body && !(body instanceof FormData) && !(body instanceof Blob)) {
                requestHeaders['Content-Type'] = 'application/json';
            }

            try {
                const response = await fetch(endpoint, {
                    method,
                    headers: requestHeaders,
                    body: body instanceof FormData ? body : (body ? JSON.stringify(body) : null),
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                // 处理HTTP错误状态
                if (!response.ok) {
                    let errorMsg = `HTTP ${response.status}`;
                    try {
                        const errorData = await response.json();
                        errorMsg = errorData.error || errorMsg;
                    } catch (e) {
                        // 忽略JSON解析错误
                    }
                    throw new Error(errorMsg);
                }

                // 尝试解析JSON，对于非JSON响应返回原始响应
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    return await response.json();
                } else {
                    return await response.blob();
                }
            } catch (error) {
                clearTimeout(timeoutId);
                if (error.name === 'AbortError') {
                    throw new Error('请求超时');
                }
                throw error;
            }
        }

        // 文件大小格式化
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        // 设备检测和响应式布局切换
        function checkDeviceWidth() {
            const isMobileNow = window.innerWidth < 768;
            if (isMobileNow !== state.isMobile) {
                state.isMobile = isMobileNow;
                updatePreviewVisibility();
            }
        }

        // ==================== 认证模块 ====================

        function initAuth() {
            const pinInput = document.getElementById('pin-input');
            const loginBtn = document.getElementById('login-btn');
            const authOverlay = document.getElementById('auth-overlay');
            const mainApp = document.getElementById('main-app');
            const authError = document.getElementById('auth-error');

            // 设置输入框属性
            pinInput.setAttribute('inputmode', 'numeric');

            // 检查现有token
            if (state.authToken) {
                // 验证token有效性（通过简单的状态查询）
                verifyTokenAndShowMain();
            } else {
                authOverlay.classList.remove('hidden');
                mainApp.classList.add('hidden');
            }

            // 回车登录
            pinInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    performLogin();
                }
            });

            // 点击登录
            loginBtn.addEventListener('click', performLogin);

            async function performLogin() {
                const token = pinInput.value.trim();
                if (!token) {
                    showAuthError('请输入访问码');
                    return;
                }

                showLoading('登录中...');
                try {
                    const response = await apiRequest('/auth', {
                        method: 'POST',
                        body: { token },
                        withAuth: false
                    });

                    // 登录成功
                    state.authToken = token;
                    localStorage.setItem('auth_token', token);
                    hideLoading();
                    authOverlay.classList.add('hidden');
                    mainApp.classList.remove('hidden');
                    showToast('登录成功', 'success');

                    // 开始轮询打印机状态
                    startPrinterStatusPolling();
                    // 获取打印机能力
                    fetchPrinterCapabilities();
                    // 初始化打印设置
                    initPrintSettings();

                } catch (error) {
                    hideLoading();
                    showAuthError('访问码无效');
                    console.error('登录失败:', error);
                }
            }

            function showAuthError(message) {
                authError.textContent = message;
                authError.classList.remove('hidden');
            }

            async function verifyTokenAndShowMain() {
                try {
                    // 尝试获取打印机状态来验证token
                    await apiRequest('/printer/status');
                    // token有效
                    authOverlay.classList.add('hidden');
                    mainApp.classList.remove('hidden');
                    startPrinterStatusPolling();
                    fetchPrinterCapabilities();
                    initPrintSettings();
                } catch (error) {
                    // token无效，清除并显示认证界面
                    localStorage.removeItem('auth_token');
                    state.authToken = null;
                    authOverlay.classList.remove('hidden');
                    mainApp.classList.add('hidden');
                }
            }
        }

        // ==================== 文件上传模块 ====================

        function initFileUpload() {
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const fileList = document.getElementById('file-list');

            // 点击上传区域
            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });

            // 拖拽支持
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = 'var(--accent)';
                uploadArea.style.background = '#fff';
            });

            uploadArea.addEventListener('dragleave', () => {
                uploadArea.style.borderColor = 'var(--border)';
                uploadArea.style.background = 'var(--bg-elevated)';
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.style.borderColor = 'var(--border)';
                uploadArea.style.background = 'var(--bg-elevated)';
                const files = Array.from(e.dataTransfer.files);
                handleFiles(files);
            });

            // 文件选择变化
            fileInput.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                handleFiles(files);
                // 清空input以便选择相同文件
                e.target.value = '';
            });

            async function handleFiles(files) {
                // 过滤支持的文件类型（使用打印机支持的格式）
                const supportedTypes = state.supportedDocumentFormats;
                const validFiles = files.filter(file => {
                    // 检查 MIME 类型
                    if (supportedTypes.includes(file.type.toLowerCase())) {
                        return true;
                    }
                    // 检查文件扩展名
                    const ext = file.name.toLowerCase().split('.').pop();
                    const extMap = {
                        'jpg': 'image/jpeg',
                        'jpeg': 'image/jpeg',
                        'png': 'image/png',
                        'pdf': 'application/pdf'
                    };
                    return extMap[ext] && supportedTypes.includes(extMap[ext]);
                });

                if (validFiles.length === 0) {
                    showToast('文件格式不支持', 'error');
                    return;
                }

                // 顺序上传每个文件
                for (const file of validFiles) {
                    await uploadFile(file);
                }
            }

            async function uploadFile(file) {
                // 添加文件到列表（上传中状态）
                const fileId = Date.now() + Math.random();
                const fileItem = {
                    id: fileId,
                    name: file.name,
                    size: file.size,
                    status: 'uploading',
                    progress: 0
                };
                state.uploadedFiles.push(fileItem);
                updateFileList();

                // 创建FormData
                const formData = new FormData();
                formData.append('file', file);

                // 使用XHR以支持进度显示
                return new Promise((resolve, reject) => {
                    const xhr = new XMLHttpRequest();

                    xhr.upload.addEventListener('progress', (e) => {
                        if (e.lengthComputable) {
                            const progress = Math.round((e.loaded / e.total) * 100);
                            fileItem.progress = progress;
                            updateFileItem(fileId);
                        }
                    });

                    xhr.addEventListener('load', () => {
                        if (xhr.status === 200) {
                            try {
                                const response = JSON.parse(xhr.responseText);
                                fileItem.status = 'success';
                                fileItem.pages = response.pages;
                                fileItem.size = response.size;
                                updateFileItem(fileId);
                                showToast(`${file.name} 上传成功`, 'success');

                                // 刷新预览
                                updatePreview();
                                resolve(response);
                            } catch (error) {
                                fileItem.status = 'error';
                                updateFileItem(fileId);
                                showToast(`${file.name} 上传失败: 响应解析错误`, 'error');
                                reject(error);
                            }
                        } else {
                            fileItem.status = 'error';
                            updateFileItem(fileId);
                            let errorMsg = `HTTP ${xhr.status}`;
                            try {
                                const errorData = JSON.parse(xhr.responseText);
                                errorMsg = errorData.error || errorMsg;
                            } catch (e) {}
                            showToast(`${file.name} 上传失败: ${errorMsg}`, 'error');
                            reject(new Error(errorMsg));
                        }
                    });

                    xhr.addEventListener('error', () => {
                        fileItem.status = 'error';
                        updateFileItem(fileId);
                        showToast(`${file.name} 上传失败: 网络错误`, 'error');
                        reject(new Error('Network error'));
                    });

                    xhr.addEventListener('abort', () => {
                        fileItem.status = 'error';
                        updateFileItem(fileId);
                        showToast(`${file.name} 上传已取消`, 'error');
                        reject(new Error('Upload aborted'));
                    });

                    xhr.open('POST', '/upload');
                    xhr.setRequestHeader('Authorization', `Bearer ${state.authToken}`);
                    xhr.send(formData);
                });
            }

            function updateFileItem(fileId) {
                const items = document.querySelectorAll('#file-list .file-item');
                items.forEach(item => {
                    if (item.dataset.fileId === fileId.toString()) {
                        const fileItem = state.uploadedFiles.find(f => f.id === fileId);
                        if (fileItem) {
                            item.innerHTML = createFileItemHTML(fileItem);
                        }
                    }
                });
            }
        }

        // ==================== 文件列表渲染 ====================

        function updateFileList() {
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '';

            state.uploadedFiles.forEach(fileItem => {
                const div = document.createElement('div');
                div.className = 'file-item p-3 flex items-center justify-between';
                div.style.cssText = 'background: #fff; border: 1px solid var(--border-light); border-radius: 16px; transition: all 0.3s ease;';
                div.dataset.fileId = fileItem.id;
                div.innerHTML = createFileItemHTML(fileItem);
                fileList.appendChild(div);
            });

            // 如果没有文件
            if (state.uploadedFiles.length === 0) {
                fileList.innerHTML = '<div class="text-center p-10" style="color: var(--text-muted); background: #fff; border-radius: 16px; border: 1px dashed var(--border);"><div class="text-5xl mb-3">📭</div><p class="font-medium">暂无上传文件</p><p class="text-sm mt-1">上传文件后将显示在这里</p></div>';
            }
        }

        function createFileItemHTML(fileItem) {
            // 左边框颜色
            let borderColor = 'var(--border-light)';
            if (fileItem.status === 'success') borderColor = 'var(--success)';
            else if (fileItem.status === 'error') borderColor = 'var(--error)';
            else if (fileItem.status === 'uploading') borderColor = 'var(--accent)';

            let statusHTML = '';
            if (fileItem.status === 'uploading') {
                statusHTML = `
                    <div class="flex items-center">
                        <div class="w-24 h-2 rounded-full mr-3 overflow-hidden" style="background: var(--bg2);">
                            <div class="h-full transition-all duration-300" style="width: ${fileItem.progress}%; background: linear-gradient(90deg, var(--accent), var(--accent-aqua));"></div>
                        </div>
                        <span class="text-xs font-medium" style="color: var(--fg2);">${fileItem.progress}%</span>
                    </div>
                `;
            } else if (fileItem.status === 'success') {
                statusHTML = `
                    <div class="flex items-center px-3 py-1.5" style="background: rgba(141, 161, 1, 0.12); border-radius: 10px;">
                        <span class="mr-1.5 text-sm">✓</span>
                        <span class="text-sm font-medium" style="color: var(--success);">已上传</span>
                    </div>
                `;
            } else {
                statusHTML = `
                    <div class="flex items-center px-3 py-1.5" style="background: rgba(193, 74, 74, 0.12); border-radius: 10px;">
                        <span class="mr-1.5 text-sm">✕</span>
                        <span class="text-sm font-medium" style="color: var(--error);">失败</span>
                    </div>
                `;
            }

            // 文件类型图标
            const fileExt = fileItem.name.split('.').pop().toLowerCase();
            const fileIcon = fileExt === 'pdf' ? '📕' : (fileExt === 'jpg' || fileExt === 'jpeg' ? '🖼️' : (fileExt === 'png' ? '🖼️' : '📄'));

            return `
                <div class="flex items-center p-4 card-hover" style="background: #fff; border-left: 4px solid ${borderColor}; border-radius: 14px; transition: all 0.3s ease;"
                     onmouseover="this.style.borderLeftColor='var(--accent)';this.style.boxShadow='0 4px 16px rgba(10, 125, 140, 0.15)';this.style.transform='translateX(4px)'"
                     onmouseout="this.style.borderLeftColor='${borderColor}' ;this.style.boxShadow='none';this.style.transform='translateX(0)'">
                    <div class="text-3xl mr-4">${fileIcon}</div>
                    <div class="flex-1 min-w-0">
                        <div class="font-semibold truncate" style="color: var(--fg0);">${fileItem.name}</div>
                        <div class="text-sm mt-0.5" style="color: var(--text-muted);">${formatFileSize(fileItem.size)}${fileItem.pages ? ' · ' + fileItem.pages + ' 页' : ''}</div>
                    </div>
                    ${statusHTML}
                </div>
            `;
        }

        // ==================== 预览模块 ====================

        function updatePreview() {
            if (state.uploadedFiles.length === 0 || state.uploadedFiles.every(f => f.status !== 'success')) {
                // 没有成功上传的文件，显示空状态
                document.getElementById('preview-desktop').classList.add('hidden');
                document.getElementById('preview-mobile').classList.add('hidden');
                document.getElementById('preview-empty').classList.remove('hidden');
                return;
            }

            // 有成功上传的文件
            document.getElementById('preview-empty').classList.add('hidden');
            updatePreviewVisibility();

            // 更新PDF预览源（添加时间戳避免缓存）
            const timestamp = new Date().getTime();
            const pdfUrl = `/preview.pdf?token=${encodeURIComponent(state.authToken)}&t=${timestamp}`;

            // 桌面端：更新embed
            const embed = document.getElementById('pdf-embed');
            embed.src = pdfUrl;

            // 移动端：更新预览按钮链接
            const previewBtn = document.getElementById('preview-fullscreen');
            previewBtn.onclick = () => {
                window.open(pdfUrl, '_blank');
            };

            // 更新页数信息（需要从后端获取，这里简化显示）
            const totalPages = state.uploadedFiles.reduce((sum, file) => sum + (file.pages || 1), 0);
            document.getElementById('preview-info').textContent = `PDF已生成（共 ${totalPages} 页）`;
        }

        function updatePreviewVisibility() {
            if (state.isMobile) {
                document.getElementById('preview-desktop').classList.add('hidden');
                document.getElementById('preview-mobile').classList.remove('hidden');
            } else {
                document.getElementById('preview-desktop').classList.remove('hidden');
                document.getElementById('preview-mobile').classList.add('hidden');
            }
        }

        // ==================== 打印设置UI交互 ====================

        function initPrintSettings() {
            const copiesInput = document.getElementById('copies-input');
            const copiesDecrement = document.getElementById('copies-decrement');
            const copiesIncrement = document.getElementById('copies-increment');

            const sidesOne = document.getElementById('sides-one');
            const sidesTwo = document.getElementById('sides-two');

            const colorBw = document.getElementById('color-bw');
            const colorColor = document.getElementById('color-color');

            const mediaSelect = document.getElementById('media-select');
            const resolutionSelect = document.getElementById('resolution-select');
            const qualityDraft = document.getElementById('quality-draft');
            const qualityNormal = document.getElementById('quality-normal');
            const qualityHigh = document.getElementById('quality-high');
            const orientationPortrait = document.getElementById('orientation-portrait');
            const orientationLandscape = document.getElementById('orientation-landscape');

            // 初始化显示
            copiesInput.value = state.printSettings.copies;
            updateSidesButtons();
            updateColorButtons();
            updateQualityButtons();
            updateOrientationButtons();

            // 份数控制
            copiesDecrement.addEventListener('click', () => {
                if (state.printSettings.copies > 1) {
                    state.printSettings.copies--;
                    copiesInput.value = state.printSettings.copies;
                }
            });

            copiesIncrement.addEventListener('click', () => {
                if (state.printSettings.copies < 99) {
                    state.printSettings.copies++;
                    copiesInput.value = state.printSettings.copies;
                }
            });

            // 单双面控制
            sidesOne.addEventListener('click', () => {
                state.printSettings.sides = 'one-sided';
                updateSidesButtons();
            });

            sidesTwo.addEventListener('click', () => {
                state.printSettings.sides = 'two-sided';
                updateSidesButtons();
            });

            // 色彩控制
            colorBw.addEventListener('click', () => {
                state.printSettings.color_mode = 'monochrome';
                updateColorButtons();
            });

            colorColor.addEventListener('click', () => {
                state.printSettings.color_mode = 'color';
                updateColorButtons();
            });

            // 纸张尺寸控制
            mediaSelect.addEventListener('change', () => {
                state.printSettings.media = mediaSelect.value;
            });

            // 打印分辨率控制
            resolutionSelect.addEventListener('change', () => {
                state.printSettings.resolution = resolutionSelect.value;
            });

            // 打印质量控制
            qualityDraft.addEventListener('click', () => {
                state.printSettings.print_quality = 'draft';
                updateQualityButtons();
            });

            qualityNormal.addEventListener('click', () => {
                state.printSettings.print_quality = 'normal';
                updateQualityButtons();
            });

            qualityHigh.addEventListener('click', () => {
                state.printSettings.print_quality = 'high';
                updateQualityButtons();
            });

            // 打印方向控制
            orientationPortrait.addEventListener('click', () => {
                state.printSettings.orientation = 'portrait';
                updateOrientationButtons();
            });

            orientationLandscape.addEventListener('click', () => {
                state.printSettings.orientation = 'landscape';
                updateOrientationButtons();
            });

            function updateSidesButtons() {
                if (state.printSettings.sides === 'one-sided') {
                    sidesOne.style.background = 'var(--accent)';
                    sidesOne.style.color = '#fbf1c7';
                    sidesTwo.style.background = 'var(--bg2)'
                    sidesTwo.style.color = 'var(--fg0)';
                } else {
                    sidesTwo.style.background = 'var(--accent)';
                    sidesTwo.style.color = '#fbf1c7';
                    sidesOne.style.background = 'var(--bg2)';
                    sidesOne.style.color = 'var(--fg0)';
                }
            }

            function updateColorButtons() {
                if (state.printSettings.color_mode === 'monochrome') {
                    colorBw.style.background = 'var(--accent)';
                    colorBw.style.color = '#fbf1c7';
                    colorColor.style.background = 'var(--bg2)';
                    colorColor.style.color = 'var(--fg0)';
                } else {
                    colorColor.style.background = 'var(--accent)';
                    colorColor.style.color = '#fbf1c7';
                    colorBw.style.background = 'var(--bg2)';
                    colorBw.style.color = 'var(--fg0)';
                }
            }

            function updateQualityButtons() {
                const buttons = {
                    'draft': qualityDraft,
                    'normal': qualityNormal,
                    'high': qualityHigh
                };
                const activeQuality = state.printSettings.print_quality;

                Object.keys(buttons).forEach(quality => {
                    const btn = buttons[quality];
                    if (quality === activeQuality) {
                        btn.style.background = 'var(--accent)';
                        btn.style.color = '#fbf1c7';
                    } else {
                        btn.style.background = 'var(--bg-elevated)';
                        btn.style.color = 'var(--fg0)';
                    }
                });
            }

            function updateOrientationButtons() {
                if (state.printSettings.orientation === 'portrait') {
                    orientationPortrait.style.background = 'var(--accent)';
                    orientationPortrait.style.color = '#fbf1c7';
                    orientationLandscape.style.background = 'var(--bg-elevated)';
                    orientationLandscape.style.color = 'var(--fg0)';
                } else {
                    orientationLandscape.style.background = 'var(--accent)';
                    orientationLandscape.style.color = '#fbf1c7';
                    orientationPortrait.style.background = 'var(--bg-elevated)';
                    orientationPortrait.style.color = 'var(--fg0)';
                }
            }
        }

        // ==================== 打印机能力获取 ====================

        async function fetchPrinterCapabilities() {
            try {
                const caps = await apiRequest('/printer/capabilities');
                state.capabilities = caps;
                updateUIWithCapabilities(caps);
                console.log('打印机能力:', caps);
            } catch (error) {
                console.error('获取打印机能力失败:', error);
                // 使用默认能力
                state.capabilities = {
                    media: ['iso-a4'],
                    sides: ['one-sided'],
                    color_mode: ['monochrome'],
                    print_quality: ['normal'],
                    orientation: ['portrait'],
                    document_formats: ['image/jpeg', 'image/png', 'application/pdf'],
                    printer_resolution: [],
                    printer_name: ''
                };
            }
        }

        function updateUIWithCapabilities(caps) {
            // 更新纸张尺寸下拉框
            const mediaSelect = document.getElementById('media-select');
            if (caps.media && Array.isArray(caps.media)) {
                mediaSelect.innerHTML = '';
                caps.media.forEach(media => {
                    const option = document.createElement('option');
                    option.value = media;
                    // 简化显示名称
                    let displayName = media;
                    if (media.startsWith('iso-')) {
                        displayName = media.replace('iso-', 'ISO ').toUpperCase();
                    } else if (media === 'letter') {
                        displayName = 'Letter (Letter)';
                    } else if (media === 'legal') {
                        displayName = 'Legal (Legal)';
                    }
                    option.textContent = displayName;
                    mediaSelect.appendChild(option);
                });
                // 设置默认选择第一个
                if (caps.media.length > 0) {
                    state.printSettings.media = caps.media[0];
                    mediaSelect.value = caps.media[0];
                }
            }

            // 更新打印分辨率下拉框
            const resolutionSelect = document.getElementById('resolution-select');
            if (caps.printer_resolution && Array.isArray(caps.printer_resolution) && caps.printer_resolution.length > 0) {
                enableSelect(resolutionSelect);
                resolutionSelect.innerHTML = '';
                caps.printer_resolution.forEach(res => {
                    const option = document.createElement('option');
                    option.value = res;
                    // 简化分辨率显示（如 "300dpi" 或 "300x300dpi"）
                    option.textContent = res;
                    resolutionSelect.appendChild(option);
                });
                if (caps.printer_resolution.length > 0) {
                    state.printSettings.resolution = caps.printer_resolution[0];
                    resolutionSelect.value = caps.printer_resolution[0];
                }
            } else {
                resolutionSelect.innerHTML = '<option value="">不支持</option>';
                disableSelect(resolutionSelect);
            }

            // 更新打印机名称显示
            const printerNameEl = document.getElementById('printer-name');
            if (caps.printer_name) {
                printerNameEl.textContent = caps.printer_name;
            }

            // 更新支持的文档格式
            if (caps.document_formats && Array.isArray(caps.document_formats) && caps.document_formats.length > 0) {
                state.supportedDocumentFormats = caps.document_formats;
            }

            // 更新单双面按钮（不禁用双面选项，因为有些打印机支持半自动双面打印）
            const sidesOne = document.getElementById('sides-one');
            const sidesTwo = document.getElementById('sides-two');
            if (caps.sides && Array.isArray(caps.sides)) {
                const supportsOneSided = caps.sides.includes('one-sided');

                // 只禁用单面选项（如果不支持的话），不禁用双面（支持半自动双面打印）
                if (!supportsOneSided) {
                    disableButton(sidesOne);
                }
                // 注意：不禁用 sidesTwo，因为有些打印机可以半自动双面打印
                // 打印机会吸回纸张但属性仍显示为 one-sided
            }

            // 更新色彩模式按钮
            const colorBw = document.getElementById('color-bw');
            const colorColor = document.getElementById('color-color');
            if (caps.color_mode && Array.isArray(caps.color_mode)) {
                const supportsMonochrome = caps.color_mode.includes('monochrome');
                const supportsColor = caps.color_mode.includes('color');

                if (!supportsMonochrome) {
                    disableButton(colorBw);
                }
                if (!supportsColor) {
                    disableButton(colorColor);
                }
            }

            // 更新打印质量按钮
            const qualityDraft = document.getElementById('quality-draft');
            const qualityNormal = document.getElementById('quality-normal');
            const qualityHigh = document.getElementById('quality-high');
            if (caps.print_quality && Array.isArray(caps.print_quality)) {
                const supportsDraft = caps.print_quality.includes('draft');
                const supportsNormal = caps.print_quality.includes('normal');
                const supportsHigh = caps.print_quality.includes('high');

                if (!supportsDraft) {
                    disableButton(qualityDraft);
                }
                if (!supportsNormal) {
                    disableButton(qualityNormal);
                }
                if (!supportsHigh) {
                    disableButton(qualityHigh);
                }
            }

            // 更新打印方向按钮
            const orientationPortrait = document.getElementById('orientation-portrait');
            const orientationLandscape = document.getElementById('orientation-landscape');
            if (caps.orientation && Array.isArray(caps.orientation)) {
                const supportsPortrait = caps.orientation.includes('portrait');
                const supportsLandscape = caps.orientation.includes('landscape');

                if (!supportsPortrait) {
                    disableButton(orientationPortrait);
                }
                if (!supportsLandscape) {
                    disableButton(orientationLandscape);
                }
            }
        }

        function disableButton(btn) {
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
            btn.style.pointerEvents = 'none';
            btn.title = '当前打印机不支持此选项';
        }

        function disableSelect(select) {
            select.disabled = true;
            select.style.opacity = '0.5';
            select.style.cursor = 'not-allowed';
            select.title = '当前打印机不支持此选项';
        }

        function enableSelect(select) {
            select.disabled = false;
            select.style.opacity = '1';
            select.style.cursor = 'pointer';
            select.style.pointerEvents = 'auto';
            select.title = '';
        }

        // ==================== 打印机状态轮询 ====================

        function startPrinterStatusPolling() {
            // 先清除现有定时器
            if (state.printerStatusInterval) {
                clearInterval(state.printerStatusInterval);
            }

            // 立即查询一次
            updatePrinterStatus();

            // 每30秒轮询
            state.printerStatusInterval = setInterval(updatePrinterStatus, 30000);
        }

        async function updatePrinterStatus() {
            try {
                const statusData = await apiRequest('/printer/status');
                state.printerStatus = statusData.status;

                const text = document.getElementById('status-text');

                // 添加空值检查：如果元素不存在，停止轮询
                if (!text) {
                    console.warn('status-text 元素不存在，停止轮询');
                    if (state.printerStatusInterval) {
                        clearInterval(state.printerStatusInterval);
                        state.printerStatusInterval = null;
                    }
                    return;
                }

                if (statusData.status === 'online') {
                    text.textContent = (statusData.printer_name ? '🟢 ' + statusData.printer_name : '🟢 打印机在线');
                    text.style.color = 'var(--fg1)';
                    enablePrintControls();
                } else {
                    text.textContent = (statusData.error ? '🔴 ' + statusData.error : '🔴 打印机离线');
                    text.style.color = 'var(--fg1)';
                    disablePrintControls();
                }
            } catch (error) {
                console.error('获取打印机状态失败:', error);
                const text = document.getElementById('status-text');

                // 同样添加空值检查
                if (!text) {
                    console.warn('status-text 元素不存在，停止轮询');
                    if (state.printerStatusInterval) {
                        clearInterval(state.printerStatusInterval);
                        state.printerStatusInterval = null;
                    }
                    return;
                }

                text.textContent = '🟡 状态查询失败';
                text.style.color = 'var(--fg1)';
                disablePrintControls();
            }
        }

        // ==================== 打印控件启用/禁用 ====================

        function disablePrintControls() {
            // 禁用打印按钮
            const printButtons = ['print-btn', 'print-btn-desktop'];
            printButtons.forEach(id => {
                const btn = document.getElementById(id);
                if (btn) {
                    btn.disabled = true;
                    btn.classList.add('opacity-50', 'cursor-not-allowed');
                }
            });

            // 禁用所有配置选项
            const configControls = [
                'copies-decrement', 'copies-increment',
                'sides-one', 'sides-two',
                'color-bw', 'color-color',
                'quality-draft', 'quality-normal', 'quality-high',
                'orientation-portrait', 'orientation-landscape'
            ];
            configControls.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.disabled = true;
                    el.style.opacity = '0.5';
                    el.style.cursor = 'not-allowed';
                }
            });

            // 禁用纸张尺寸下拉框
            const mediaSelect = document.getElementById('media-select');
            if (mediaSelect) {
                mediaSelect.disabled = true;
                mediaSelect.style.opacity = '0.5';
                mediaSelect.style.cursor = 'not-allowed';
            }

            // 禁用打印分辨率下拉框
            const resolutionSelect = document.getElementById('resolution-select');
            if (resolutionSelect) {
                resolutionSelect.disabled = true;
                resolutionSelect.style.opacity = '0.5';
                resolutionSelect.style.cursor = 'not-allowed';
            }
        }

        function enablePrintControls() {
            // 启用打印按钮
            const printButtons = ['print-btn', 'print-btn-desktop'];
            printButtons.forEach(id => {
                const btn = document.getElementById(id);
                if (btn) {
                    btn.disabled = false;
                    btn.classList.remove('opacity-50', 'cursor-not-allowed');
                }
            });

            // 恢复配置选项的基础状态（重新应用打印机能力）
            const configControls = [
                'copies-decrement', 'copies-increment',
                'sides-one', 'sides-two',
                'color-bw', 'color-color',
                'quality-draft', 'quality-normal', 'quality-high',
                'orientation-portrait', 'orientation-landscape'
            ];
            configControls.forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.disabled = false;
                    el.style.opacity = '1';
                    el.style.cursor = 'pointer';
                }
            });

            // 启用纸张尺寸下拉框
            const mediaSelect = document.getElementById('media-select');
            if (mediaSelect) {
                mediaSelect.disabled = false;
                mediaSelect.style.opacity = '1';
                mediaSelect.style.cursor = 'pointer';
            }

            // 启用打印分辨率下拉框
            const resolutionSelect = document.getElementById('resolution-select');
            if (resolutionSelect) {
                resolutionSelect.disabled = false;
                resolutionSelect.style.opacity = '1';
                resolutionSelect.style.cursor = 'pointer';
            }

            // 重新应用打印机能力（禁用不支持的选项）
            if (state.capabilities) {
                updateUIWithCapabilities(state.capabilities);
            }
        }

        // ==================== 打印和取消功能 ====================

        function initPrintAndCancel() {
            const cancelBtn = document.getElementById('cancel-btn');
            const printBtn = document.getElementById('print-btn');
            const cancelBtnDesktop = document.getElementById('cancel-btn-desktop');
            const printBtnDesktop = document.getElementById('print-btn-desktop');

            // 移动端按钮
            cancelBtn.addEventListener('click', handleCancel);
            printBtn.addEventListener('click', handlePrint);

            // 桌面端按钮
            cancelBtnDesktop.addEventListener('click', handleCancel);
            printBtnDesktop.addEventListener('click', handlePrint);

            // 退出按钮
            document.getElementById('logout-btn').addEventListener('click', handleLogout);
        }

        async function handlePrint() {
            // 检查是否有上传的文件
            if (state.uploadedFiles.length === 0) {
                showToast('请先上传文件', 'error');
                return;
            }

            // 检查是否有成功上传的文件
            const hasSuccessFile = state.uploadedFiles.some(f => f.status === 'success');
            if (!hasSuccessFile) {
                showToast('请等待文件上传完成', 'error');
                return;
            }

            // 禁用按钮，显示加载
            disablePrintButtons(true);
            showLoading('打印中...');

            try {
                const printData = {
                    copies: state.printSettings.copies,
                    sides: state.printSettings.sides,
                    color_mode: state.printSettings.color_mode,
                    media: state.printSettings.media,
                    print_quality: state.printSettings.print_quality,
                    orientation: state.printSettings.orientation,
                    resolution: state.printSettings.resolution
                };

                await apiRequest('/print', {
                    method: 'POST',
                    body: printData
                });

                showToast('打印任务已提交', 'success');

                // 重置界面
                resetAfterPrint();

            } catch (error) {
                showToast(`打印失败: ${error.message}`, 'error');
            } finally {
                hideLoading();
                disablePrintButtons(false);
            }
        }

        async function handleCancel() {
            if (state.uploadedFiles.length === 0) {
                showToast('没有文件需要取消', 'info');
                return;
            }

            showLoading('取消中...');

            try {
                await apiRequest('/cancel', {
                    method: 'POST'
                });

                showToast('已取消打印任务', 'success');

                // 清空文件列表和预览
                state.uploadedFiles = [];
                updateFileList();
                updatePreview();

            } catch (error) {
                showToast(`取消失败: ${error.message}`, 'error');
            } finally {
                hideLoading();
            }
        }

        function handleLogout() {
            localStorage.removeItem('auth_token');
            // 清除轮询
            if (state.printerStatusInterval) {
                clearInterval(state.printerStatusInterval);
                state.printerStatusInterval = null;
            }
            // 刷新页面
            window.location.reload();
        }

        function disablePrintButtons(disabled) {
            const buttons = [
                document.getElementById('print-btn'),
                document.getElementById('print-btn-desktop'),
                document.getElementById('cancel-btn'),
                document.getElementById('cancel-btn-desktop')
            ];

            buttons.forEach(btn => {
                if (btn) {
                    btn.disabled = disabled;
                    if (disabled) {
                        if (btn.id.includes('print')) {
                            btn.textContent = '🖨️ 打印中...';
                        }
                        btn.classList.add('opacity-50', 'cursor-not-allowed');
                        btn.style.pointerEvents = 'none';
                    } else {
                        if (btn.id.includes('print')) {
                            btn.textContent = '🖨️ 打印';
                        }
                        btn.classList.remove('opacity-50', 'cursor-not-allowed');
                        btn.style.pointerEvents = 'auto';
                    }
                }
            });
        }

        function resetAfterPrint() {
            state.uploadedFiles = [];
            updateFileList();
            updatePreview();
            // 重置打印设置为默认
            state.printSettings.copies = 1;
            state.printSettings.sides = 'one-sided';
            state.printSettings.color_mode = 'monochrome';
            state.printSettings.print_quality = 'normal';
            state.printSettings.orientation = 'portrait';
            // 保持 media 和 resolution 不变，因为通常不需要每次重置

            // 更新UI
            document.getElementById('copies-input').value = 1;
            initPrintSettings();
        }

        // ==================== 页面初始化 ====================

        function init() {
            console.log('Just-Printing-Server 前端初始化');

            // 初始化认证
            initAuth();

            // 初始化文件上传
            initFileUpload();

            // 初始化打印和取消
            initPrintAndCancel();

            // 初始化打印设置（在认证成功后调用）
            // initPrintSettings() 会在认证成功后调用

            // 监听窗口大小变化
            window.addEventListener('resize', checkDeviceWidth);

            // 初始设备检测
            checkDeviceWidth();
        }

        // 启动应用
        document.addEventListener('DOMContentLoaded', init);
