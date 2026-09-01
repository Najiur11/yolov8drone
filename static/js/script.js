// 全局变量
let socket;
let heartbeatInterval;

// 初始化 Socket.IO 连接
function initSocket() {
    if (typeof io !== 'undefined') {
        console.log('Socket.IO 库已加载');

        // 明确指定连接 URL 和端口
        socket = io('http://localhost:5000', {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionAttempts: Infinity, // 无限重试
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            timeout: 20000,
            forceNew: true,
            maxHttpBufferSize: 100 * 1024 * 1024 // 增加缓冲区大小到100MB
        });

        // 连接事件
        socket.on('connect', function() {
            console.log('已连接到服务器');
            updateConnectionStatus(true);
            startHeartbeat();
            // 发送连接确认
            socket.emit('client_connected', {message: '客户端已连接'});
        });

        socket.on('disconnect', function(reason) {
            console.log('与服务器断开连接，原因:', reason);
            updateConnectionStatus(false);
            stopHeartbeat();
        });

        socket.on('connect_error', function(error) {
            console.error('连接错误:', error);
            updateConnectionStatus(false);
            stopHeartbeat();
        });

        socket.on('connect_timeout', function(timeout) {
            console.error('连接超时:', timeout);
            updateConnectionStatus(false);
            stopHeartbeat();
        });

        socket.on('connection_status', function(data) {
            console.log('服务器连接状态:', data);
            if (data.status === 'connected') {
                updateConnectionStatus(true);
                startHeartbeat();
            }
        });

        socket.on('connection_established', function(data) {
            console.log('服务器确认:', data.message);
            // 启用所有按钮
            document.querySelectorAll('button').forEach(btn => {
                btn.disabled = false;
            });
        });

        // 处理服务器返回的图像处理结果
        socket.on('image_processed', function(data) {
            document.getElementById('processed-frame').src = data.processed_image;
            updateProcessingStatus('completed', '图像处理完成');
        });

        // 处理服务器返回的视频帧
        socket.on('video_frame_processed', function(data) {
            document.getElementById('processed-frame').src = data.processed_frame;
            updateProgress(data.progress);
        });

        // 视频处理开始
        socket.on('video_processing_started', function(data) {
            updateProcessingStatus('processing', '视频处理中...');
        });

        // 处理完成
        socket.on('processing_complete', function(data) {
            updateProcessingStatus('completed', '视频处理完成');
            resetProgress();
        });

        // 处理错误
        socket.on('processing_error', function(data) {
            updateProcessingStatus('error', '处理错误: ' + data.message);
            resetProgress();
        });

        // 心跳响应
        socket.on('heartbeat_response', function(data) {
            console.log('心跳响应:', data);
        });
    } else {
        console.error('Socket.IO 未加载');
        // 延迟重试
        setTimeout(initSocket, 1000);
    }
}

// 开始心跳检测
function startHeartbeat() {
    // 每30秒发送一次心跳
    heartbeatInterval = setInterval(() => {
        if (socket && socket.connected) {
            socket.emit('heartbeat', {timestamp: Date.now()});
        }
    }, 30000);
}

// 停止心跳检测
function stopHeartbeat() {
    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
    }
}

// 更新连接状态
function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.textContent = '已连接';
        statusElement.style.backgroundColor = '#51cf66';

        // 启用所有按钮
        document.querySelectorAll('button').forEach(btn => {
            btn.disabled = false;
        });
    } else {
        statusElement.textContent = '未连接';
        statusElement.style.backgroundColor = '#ff6b6b';

        // 禁用所有处理按钮
        document.getElementById('process-image-btn').disabled = true;
        document.getElementById('process-video-btn').disabled = true;
        document.getElementById('track-video-btn').disabled = true;
    }
}

// 更新处理状态
function updateProcessingStatus(status, message = "") {
    const statusElement = document.getElementById('processing-status');
    statusElement.textContent = message || status;

    switch(status) {
        case 'processing':
            statusElement.style.backgroundColor = '#fcc419';
            break;
        case 'completed':
            statusElement.style.backgroundColor = '#51cf66';
            break;
        case 'error':
            statusElement.style.backgroundColor = '#ff6b6b';
            break;
        default:
            statusElement.style.backgroundColor = '#868e96';
    }
}

// 处理图像上传和显示
document.getElementById('image-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById('image-info').textContent = file.name;
        document.getElementById('image-info').style.backgroundColor = '#e0ffe0';

        // 显示原始图像
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('original-frame').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// 处理视频上传并显示预览
document.getElementById('video-input').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        // 检查文件类型
        const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'];
        if (!allowedTypes.includes(file.type)) {
            alert('请选择支持的视频格式: MP4, AVI, MOV, MKV, WEBM');
            this.value = ''; // 清空文件选择
            return;
        }

        // 检查文件大小 (限制为100MB)
        const maxSize = 100 * 1024 * 1024; // 100MB
        if (file.size > maxSize) {
            alert('视频文件太大，请选择小于100MB的文件');
            this.value = ''; // 清空文件选择
            return;
        }

        document.getElementById('video-info').textContent = file.name;
        document.getElementById('video-info').style.backgroundColor = '#e0ffe0';

        // 创建视频预览
        const videoURL = URL.createObjectURL(file);
        const videoElement = document.createElement('video');
        videoElement.controls = true;
        videoElement.src = videoURL;
        videoElement.style.maxWidth = '100%';
        videoElement.style.maxHeight = '400px';

        // 清空原始图像区域并添加视频元素
        const originalFrame = document.getElementById('original-frame');
        originalFrame.src = '';
        originalFrame.style.display = 'none';

        const container = originalFrame.parentNode;
        // 移除可能已存在的视频元素
        const existingVideo = container.querySelector('video');
        if (existingVideo) {
            container.removeChild(existingVideo);
        }

        container.appendChild(videoElement);
    }
});

// 处理图像
function processImage() {
    if (!socket || !socket.connected) {
        alert('请等待连接建立后再尝试');
        return;
    }

    const fileInput = document.getElementById('image-input');
    const file = fileInput.files[0];

    if (!file) {
        alert('请先选择图片');
        return;
    }

    updateProcessingStatus('processing', '图像处理中...');

    const reader = new FileReader();
    reader.onload = function(e) {
        socket.emit('upload_image', {
            image: e.target.result,
            filename: file.name
        });
    };
    reader.readAsDataURL(file);
}

// 处理视频
function processVideo(enableTracking) {
    if (!socket || !socket.connected) {
        alert('请等待连接建立后再尝试');
        return;
    }

    const fileInput = document.getElementById('video-input');
    const file = fileInput.files[0];

    if (!file) {
        alert('请先选择视频');
        return;
    }

    // 显示进度条
    document.getElementById('progress-container').style.display = 'block';
    updateProcessingStatus('processing', '上传视频中...');

    const reader = new FileReader();
    reader.onload = function(e) {
        console.log("开始上传视频");
        updateProcessingStatus('processing', '视频处理中...');

        socket.emit('upload_video', {
            video: e.target.result,
            filename: file.name,
            enable_tracking: enableTracking
        });
    };
    reader.readAsDataURL(file);
}

// 停止处理
function stopProcessing() {
    if (socket && socket.connected) {
        socket.emit('stop_processing');
        updateProcessingStatus('completed', '处理已停止');
        resetProgress();
    } else {
        // 如果连接已断开，直接更新状态
        updateProcessingStatus('completed', '处理已停止');
        resetProgress();
    }
}

// 更新进度条
function updateProgress(percent) {
    document.getElementById('progress').style.width = percent + '%';
    document.getElementById('progress-text').textContent = percent + '%';
}

// 重置进度条
function resetProgress() {
    setTimeout(() => {
        document.getElementById('progress').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('progress-container').style.display = 'none';
    }, 2000);
}

// 初始化页面
document.addEventListener('DOMContentLoaded', function() {
    // 隐藏进度条
    document.getElementById('progress-container').style.display = 'none';

    // 禁用所有处理按钮，直到连接建立
    document.getElementById('process-image-btn').disabled = true;
    document.getElementById('process-video-btn').disabled = true;
    document.getElementById('track-video-btn').disabled = true;

    // 初始化 Socket 连接
    initSocket();

    // 添加控制台日志，帮助调试
    console.log('页面初始化完成');
});