# Flow Diagrams

This document provides visual representations of all major flows and processes in the AI Video Translation Service. These diagrams help understand system interactions, data flow, and decision-making processes.

## 🎬 System Overview Flow

### High-Level System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        A[Web Browser]
        B[Upload Interface]
        C[Progress Monitor]
        D[Download Interface]
    end
    
    subgraph "API Gateway"
        E[FastAPI Server]
        F[REST Endpoints]
        G[WebSocket Endpoints]
        H[Static File Server]
    end
    
    subgraph "Core Services"
        I[Translation Service]
        J[Job Queue Service]
        K[Database Service]
        L[AI Service Factory]
    end
    
    subgraph "Processing Pipeline"
        M[Dubbing Pipeline]
        N[Audio Processing]
        O[Video Processing]
        P[AI Models]
    end
    
    subgraph "Storage"
        Q[SQLite Database]
        R[File System]
        S[Model Cache]
    end
    
    A --> E
    B --> F
    C --> G
    D --> H
    E --> I
    F --> J
    G --> J
    I --> M
    J --> K
    K --> Q
    L --> P
    M --> N
    M --> O
    N --> R
    O --> R
    P --> S
    
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style I fill:#e8f5e8
    style M fill:#fff3e0
    style Q fill:#fce4ec
```

## 📤 Upload and Job Creation Flow

### Complete Upload Process

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant API as FastAPI
    participant V as Validator
    participant FS as File System
    participant DB as Database
    participant Q as Job Queue
    
    U->>F: Select MP4 file
    F->>F: Validate file size/format
    F->>API: POST /api/v1/upload
    API->>V: Validate upload request
    V->>V: Check file format
    V->>V: Check file size
    V->>V: Validate languages
    V-->>API: Validation result
    
    alt Validation Success
        API->>FS: Save uploaded file
        FS-->>API: File path
        API->>DB: Create job record
        DB-->>API: Job ID
        API->>Q: Submit job to queue
        Q-->>API: Queue confirmation
        API-->>F: Job ID + URLs
        F-->>U: Upload success + progress URL
    else Validation Failure
        API-->>F: Error message
        F-->>U: Show error
    end
```

### File Upload Decision Tree

```mermaid
flowchart TD
    A[File Selected] --> B{File Size OK?}
    B -->|No| C[Show Size Error]
    B -->|Yes| D{File Format MP4?}
    D -->|No| E[Show Format Error]
    D -->|Yes| F{Languages Valid?}
    F -->|No| G[Show Language Error]
    F -->|Yes| H[Save File]
    H --> I[Create Job Record]
    I --> J[Submit to Queue]
    J --> K[Return Job ID]
    
    C --> L[End]
    E --> L
    G --> L
    K --> M[Start Processing]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style D fill:#fff3e0
    style F fill:#fff3e0
    style H fill:#e8f5e8
    style K fill:#e8f5e8
```

## 🔄 Job Queue Processing Flow

### Queue Management System

```mermaid
graph TB
    subgraph "Job Submission"
        A[New Job] --> B[Job Queue]
    end
    
    subgraph "Queue Processing"
        B --> C{Queue Empty?}
        C -->|Yes| D[Wait for Jobs]
        C -->|No| E{Capacity Available?}
        E -->|No| F[Wait for Capacity]
        E -->|Yes| G[Dequeue Job]
        G --> H[Start Processing]
        D --> C
        F --> E
    end
    
    subgraph "Job Processing"
        H --> I[Translation Service]
        I --> J[Dubbing Pipeline]
        J --> K{Processing Success?}
        K -->|Yes| L[Update Job Status]
        K -->|No| M[Handle Error]
        L --> N[Notify Completion]
        M --> O[Update Error Status]
    end
    
    subgraph "Cleanup"
        N --> P[Remove from Active Jobs]
        O --> P
        P --> Q[Release Capacity]
        Q --> E
    end
    
    style A fill:#e3f2fd
    style H fill:#e8f5e8
    style K fill:#fff3e0
    style N fill:#e8f5e8
    style O fill:#ffebee
```

### Concurrency Control Flow

```mermaid
sequenceDiagram
    participant Q as Job Queue
    participant S as Semaphore
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant W3 as Worker 3
    
    Note over S: Max Concurrent: 2
    
    Q->>S: Request slot for Job 1
    S->>W1: Acquire slot (1/2)
    W1->>W1: Process Job 1
    
    Q->>S: Request slot for Job 2
    S->>W2: Acquire slot (2/2)
    W2->>W2: Process Job 2
    
    Q->>S: Request slot for Job 3
    S-->>Q: No slots available
    Q->>Q: Wait for slot
    
    W1->>S: Release slot (1/2)
    S->>W3: Acquire slot (2/2)
    W3->>W3: Process Job 3
    
    W2->>S: Release slot (1/2)
    W3->>S: Release slot (0/2)
```

## 🎭 Dubbing Pipeline Flow

### Seven-Stage Processing Pipeline

```mermaid
flowchart TD
    A[Input Video] --> B[Stage 1: Preprocessing]
    B --> C[Stage 2: Speech-to-Text]
    C --> D[Stage 3: Translation]
    D --> E[Stage 4: Voice Configuration]
    E --> F[Stage 5: Text-to-Speech]
    F --> G[Stage 6: Postprocessing]
    G --> H[Stage 7: Cleanup]
    H --> I[Output Video]
    
    subgraph "Stage 1 Details"
        B1[Split Audio/Video] --> B2[Speaker Diarization]
        B2 --> B3[Audio Segmentation]
    end
    
    subgraph "Stage 2 Details"
        C1[Transcribe Segments] --> C2[Gender Classification]
        C2 --> C3[Language Detection]
    end
    
    subgraph "Stage 3 Details"
        D1[Text Translation] --> D2[Context Preservation]
    end
    
    subgraph "Stage 4 Details"
        E1[Voice Assignment] --> E2[Speaker Mapping]
    end
    
    subgraph "Stage 5 Details"
        F1[Generate Speech] --> F2[Timing Alignment]
    end
    
    subgraph "Stage 6 Details"
        G1[Combine Audio] --> G2[Merge with Video]
    end
    
    subgraph "Stage 7 Details"
        H1[Remove Temp Files] --> H2[Optimize Output]
    end
    
    B -.-> B1
    C -.-> C1
    D -.-> D1
    E -.-> E1
    F -.-> F1
    G -.-> G1
    H -.-> H1
    
    style A fill:#e3f2fd
    style I fill:#e8f5e8
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
    style H fill:#fff3e0
```

### Detailed Processing Flow with Error Handling

```mermaid
sequenceDiagram
    participant D as Dubber
    participant AP as Audio Processing
    participant VP as Video Processing
    participant STT as Speech-to-Text
    participant T as Translation
    participant TTS as Text-to-Speech
    participant FS as File System
    
    D->>VP: Split audio/video
    VP-->>D: Audio + Video files
    
    D->>AP: Perform speaker diarization
    AP-->>D: Speaker segments
    
    D->>STT: Transcribe audio segments
    STT-->>D: Transcribed text + metadata
    
    D->>T: Translate text
    T-->>D: Translated text
    
    D->>TTS: Generate speech
    TTS-->>D: Dubbed audio segments
    
    D->>AP: Combine audio segments
    AP-->>D: Final dubbed audio
    
    D->>VP: Merge audio with video
    VP-->>D: Final dubbed video
    
    D->>FS: Save output files
    FS-->>D: File paths
    
    Note over D: Error handling at each stage
    Note over D: Progress updates sent
    Note over D: Resource cleanup performed
```

## 📡 Real-time Progress Updates Flow

### WebSocket Communication Flow

```mermaid
sequenceDiagram
    participant F as Frontend
    participant WS as WebSocket Server
    participant Q as Job Queue
    participant P as Processing Pipeline
    
    F->>WS: Connect to job progress
    WS->>Q: Register progress callback
    
    loop Processing Stages
        P->>Q: Send progress update
        Q->>WS: Forward progress
        WS->>F: Broadcast update
        F->>F: Update UI
    end
    
    P->>Q: Send completion
    Q->>WS: Forward completion
    WS->>F: Broadcast completion
    F->>F: Show download button
    
    F->>WS: Disconnect
    WS->>Q: Remove callback
```

### Progress Update State Machine

```mermaid
stateDiagram-v2
    [*] --> Uploaded
    Uploaded --> Processing
    Processing --> Preprocessing
    Preprocessing --> SpeechToText
    SpeechToText --> Translation
    Translation --> VoiceConfig
    VoiceConfig --> TextToSpeech
    TextToSpeech --> Postprocessing
    Postprocessing --> Completed
    
    Processing --> Failed
    Preprocessing --> Failed
    SpeechToText --> Failed
    Translation --> Failed
    VoiceConfig --> Failed
    TextToSpeech --> Failed
    Postprocessing --> Failed
    
    Completed --> [*]
    Failed --> [*]
    
    note right of Processing
        Progress: 0-10%
        Stage: "initializing"
    end note
    
    note right of Preprocessing
        Progress: 10-20%
        Stage: "preprocessing"
    end note
    
    note right of SpeechToText
        Progress: 20-40%
        Stage: "speech_to_text"
    end note
    
    note right of Translation
        Progress: 40-60%
        Stage: "translation"
    end note
    
    note right of VoiceConfig
        Progress: 60-70%
        Stage: "voice_assignment"
    end note
    
    note right of TextToSpeech
        Progress: 70-90%
        Stage: "text_to_speech"
    end note
    
    note right of Postprocessing
        Progress: 90-100%
        Stage: "postprocessing"
    end note
```

## 🤖 AI Service Factory Flow

### Model Loading and Caching Flow

```mermaid
flowchart TD
    A[Request AI Service] --> B{Model in Cache?}
    B -->|Yes| C[Return Cached Model]
    B -->|No| D{Memory Available?}
    D -->|No| E[Evict LRU Models]
    D -->|Yes| F[Load Model]
    E --> F
    F --> G{Load Successful?}
    G -->|Yes| H[Cache Model]
    G -->|No| I[Return Error]
    H --> J[Create Service Instance]
    J --> K[Return Service]
    C --> K
    
    style A fill:#e3f2fd
    style C fill:#e8f5e8
    style F fill:#fff3e0
    style I fill:#ffebee
    style K fill:#e8f5e8
```

### Model Cache Management

```mermaid
sequenceDiagram
    participant R as Request
    participant F as AI Factory
    participant C as Cache
    participant M as Memory Monitor
    participant L as Model Loader
    
    R->>F: Request STT model
    F->>C: Check cache
    C-->>F: Cache miss
    
    F->>M: Check memory usage
    M-->>F: 90% used (over threshold)
    
    F->>C: Evict LRU models
    C->>C: Remove oldest models
    C-->>F: Memory freed
    
    F->>L: Load model from disk/remote
    L-->>F: Model loaded
    
    F->>C: Cache new model
    C-->>F: Model cached
    
    F-->>R: Return service instance
```

## 🗄️ Database Operations Flow

### Job Lifecycle in Database

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Uploaded
    Uploaded --> Processing
    Processing --> Completed
    Processing --> Failed
    
    Completed --> Archived
    Failed --> Archived
    Archived --> Deleted
    
    Completed --> [*]
    Failed --> [*]
    Deleted --> [*]
    
    note right of Created
        INSERT job record
        Status: "created"
    end note
    
    note right of Uploaded
        UPDATE status
        Add file paths
    end note
    
    note right of Processing
        UPDATE status
        Add progress data
    end note
    
    note right of Completed
        UPDATE status
        Add output paths
        Set completion time
    end note
    
    note right of Failed
        UPDATE status
        Add error message
    end note
```

### Database Query Flow

```mermaid
sequenceDiagram
    participant API as API Endpoint
    participant DB as Database Service
    participant Pool as Connection Pool
    participant SQLite as SQLite DB
    
    API->>DB: Create job request
    DB->>Pool: Acquire connection
    Pool-->>DB: Connection
    
    DB->>SQLite: INSERT job
    SQLite-->>DB: Job ID
    
    DB->>Pool: Release connection
    DB-->>API: Job object
    
    Note over Pool: Connection reused
    
    API->>DB: Update job status
    DB->>Pool: Acquire connection
    Pool-->>DB: Same/different connection
    
    DB->>SQLite: UPDATE job
    SQLite-->>DB: Rows affected
    
    DB->>Pool: Release connection
    DB-->>API: Updated job
```

## 🔄 Error Handling and Recovery Flow

### Error Propagation Flow

```mermaid
flowchart TD
    A[Error Occurs] --> B{Error Type}
    
    B -->|Validation Error| C[Return 400 Bad Request]
    B -->|File Not Found| D[Return 404 Not Found]
    B -->|Processing Error| E[Update Job Status]
    B -->|System Error| F[Return 500 Internal Error]
    B -->|Timeout Error| G[Retry Operation]
    
    E --> H[Send Error via WebSocket]
    H --> I[Frontend Shows Error]
    
    F --> J[Log Error Details]
    J --> K[Health Check Reports Unhealthy]
    
    G --> L{Retry Successful?}
    L -->|Yes| M[Continue Processing]
    L -->|No| N[Mark as Failed]
    
    style C fill:#ffebee
    style D fill:#ffebee
    style E fill:#fff3e0
    style F fill:#ffebee
    style I fill:#ffebee
    style M fill:#e8f5e8
    style N fill:#ffebee
```

### Circuit Breaker Pattern Flow

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open : Failure threshold reached
    Open --> HalfOpen : Timeout expired
    HalfOpen --> Closed : Success
    HalfOpen --> Open : Failure
    
    note right of Closed
        Normal operation
        Failures < threshold
    end note
    
    note right of Open
        Reject all requests
        Fast fail mode
    end note
    
    note right of HalfOpen
        Allow limited requests
        Test if service recovered
    end note
```

## 🔐 Authentication and Security Flow

### Request Validation Flow

```mermaid
flowchart TD
    A[Incoming Request] --> B[Rate Limiting Check]
    B --> C{Rate Limit OK?}
    C -->|No| D[Return 429 Too Many Requests]
    C -->|Yes| E[Input Validation]
    E --> F{Input Valid?}
    F -->|No| G[Return 400 Bad Request]
    F -->|Yes| H[File Security Scan]
    H --> I{File Safe?}
    I -->|No| J[Return 400 Malicious File]
    I -->|Yes| K[Process Request]
    
    style D fill:#ffebee
    style G fill:#ffebee
    style J fill:#ffebee
    style K fill:#e8f5e8
```

### File Upload Security Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant V as Validator
    participant S as Security Scanner
    participant FS as File System
    
    U->>F: Upload file
    F->>V: Validate file
    V->>V: Check file size
    V->>V: Check file extension
    V->>V: Check MIME type
    V-->>F: Validation result
    
    alt Validation Passed
        F->>S: Security scan
        S->>S: Virus scan
        S->>S: Content analysis
        S-->>F: Security result
        
        alt Security Passed
            F->>FS: Save file
            FS-->>F: File saved
        else Security Failed
            F-->>U: Security error
        end
    else Validation Failed
        F-->>U: Validation error
    end
```

## 📊 Performance Monitoring Flow

### Resource Usage Monitoring

```mermaid
graph TB
    subgraph "Monitoring System"
        A[Resource Monitor] --> B[CPU Monitor]
        A --> C[Memory Monitor]
        A --> D[Disk Monitor]
        A --> E[Network Monitor]
    end
    
    subgraph "Metrics Collection"
        B --> F[CPU Usage %]
        C --> G[Memory Usage MB]
        D --> H[Disk I/O]
        E --> I[Network I/O]
    end
    
    subgraph "Alerting"
        F --> J{CPU > 80%?}
        G --> K{Memory > 85%?}
        H --> L{Disk Full?}
        I --> M{Network Slow?}
        
        J -->|Yes| N[CPU Alert]
        K -->|Yes| O[Memory Alert]
        L -->|Yes| P[Disk Alert]
        M -->|Yes| Q[Network Alert]
    end
    
    subgraph "Actions"
        N --> R[Scale Up]
        O --> S[Clear Cache]
        P --> T[Cleanup Files]
        Q --> U[Check Network]
    end
    
    style N fill:#ffebee
    style O fill:#ffebee
    style P fill:#ffebee
    style Q fill:#ffebee
```

### Performance Optimization Flow

```mermaid
flowchart TD
    A[Performance Issue Detected] --> B{Issue Type}
    
    B -->|High Memory| C[Clear Model Cache]
    B -->|High CPU| D[Reduce Concurrency]
    B -->|Slow Processing| E[Optimize Pipeline]
    B -->|Database Slow| F[Optimize Queries]
    
    C --> G[Monitor Memory]
    D --> H[Monitor CPU]
    E --> I[Monitor Processing Time]
    F --> J[Monitor Query Time]
    
    G --> K{Improved?}
    H --> K
    I --> K
    J --> K
    
    K -->|Yes| L[Continue Monitoring]
    K -->|No| M[Escalate Issue]
    
    style A fill:#fff3e0
    style L fill:#e8f5e8
    style M fill:#ffebee
```

## 🔄 Deployment and Scaling Flow

### Container Deployment Flow

```mermaid
sequenceDiagram
    participant D as Developer
    participant G as Git Repository
    participant CI as CI/CD Pipeline
    participant R as Docker Registry
    participant K as Kubernetes
    participant S as Service
    
    D->>G: Push code changes
    G->>CI: Trigger build
    CI->>CI: Run tests
    CI->>CI: Build Docker image
    CI->>R: Push image to registry
    CI->>K: Deploy to cluster
    K->>S: Create/update pods
    S->>S: Health check
    S-->>K: Ready status
    K-->>CI: Deployment success
```

### Auto-scaling Flow

```mermaid
graph TB
    subgraph "Metrics Collection"
        A[CPU Metrics] --> D[Metrics Aggregator]
        B[Memory Metrics] --> D
        C[Queue Length] --> D
    end
    
    subgraph "Scaling Decision"
        D --> E{Scale Up Needed?}
        D --> F{Scale Down Needed?}
        
        E -->|Yes| G[Increase Replicas]
        F -->|Yes| H[Decrease Replicas]
        E -->|No| I[Maintain Current]
        F -->|No| I
    end
    
    subgraph "Scaling Actions"
        G --> J[Create New Pods]
        H --> K[Terminate Pods]
        I --> L[Monitor Continuously]
    end
    
    J --> M[Load Balancer Update]
    K --> M
    M --> L
    L --> D
    
    style G fill:#e8f5e8
    style H fill:#fff3e0
    style I fill:#e3f2fd
```

## 🧪 Testing Flow

### Automated Testing Pipeline

```mermaid
flowchart TD
    A[Code Commit] --> B[Unit Tests]
    B --> C{Unit Tests Pass?}
    C -->|No| D[Fail Build]
    C -->|Yes| E[Integration Tests]
    E --> F{Integration Tests Pass?}
    F -->|No| D
    F -->|Yes| G[End-to-End Tests]
    G --> H{E2E Tests Pass?}
    H -->|No| D
    H -->|Yes| I[Performance Tests]
    I --> J{Performance OK?}
    J -->|No| K[Performance Alert]
    J -->|Yes| L[Deploy to Staging]
    L --> M[Manual Testing]
    M --> N{Manual Tests Pass?}
    N -->|No| O[Fix Issues]
    N -->|Yes| P[Deploy to Production]
    
    O --> A
    
    style D fill:#ffebee
    style K fill:#fff3e0
    style P fill:#e8f5e8
```

### Test Coverage Flow

```mermaid
graph LR
    subgraph "Unit Tests"
        A[Service Tests] --> D[Coverage Report]
        B[Model Tests] --> D
        C[Utility Tests] --> D
    end
    
    subgraph "Integration Tests"
        E[API Tests] --> F[Integration Report]
        G[Database Tests] --> F
        H[Queue Tests] --> F
    end
    
    subgraph "E2E Tests"
        I[Upload Flow] --> J[E2E Report]
        K[Processing Flow] --> J
        L[Download Flow] --> J
    end
    
    D --> M[Combined Coverage]
    F --> M
    J --> M
    M --> N{Coverage > 80%?}
    N -->|Yes| O[Quality Gate Pass]
    N -->|No| P[Quality Gate Fail]
    
    style O fill:#e8f5e8
    style P fill:#ffebee
```

## 🎯 Key Flow Characteristics

### Asynchronous Processing Benefits

1. **Non-blocking Operations**: File uploads don't block other requests
2. **Scalable Concurrency**: Multiple jobs processed simultaneously
3. **Resource Management**: Intelligent queuing prevents resource exhaustion
4. **Real-time Feedback**: Users get immediate job IDs and progress updates
5. **Fault Tolerance**: Failed jobs don't affect other processing

### Data Flow Patterns

1. **Upload → Queue → Process → Store → Download**: Linear progression with checkpoints
2. **Progress Broadcasting**: Real-time updates via WebSocket connections
3. **State Persistence**: Job state maintained in database throughout lifecycle
4. **Resource Cleanup**: Automatic cleanup of temporary files and resources
5. **Error Recovery**: Graceful error handling with user-friendly messages

### Scalability Considerations

1. **Horizontal Scaling**: Stateless design allows multiple service instances
2. **Queue Management**: Configurable concurrency limits prevent overload
3. **Model Caching**: Shared model cache reduces memory usage
4. **Database Optimization**: Efficient queries and indexing for job management
5. **File Storage**: Configurable storage backends for different deployment scenarios

### Security Measures

1. **Input Validation**: Comprehensive file and parameter validation
2. **Rate Limiting**: Prevents abuse and ensures fair resource usage
3. **Error Sanitization**: Secure error messages prevent information leakage
4. **Resource Limits**: Prevents denial of service attacks
5. **Content Scanning**: Security scanning for uploaded files

---

*This completes the comprehensive documentation for the AI Video Translation Service. All major flows, components, and architectural decisions have been documented with visual diagrams and detailed explanations.*
