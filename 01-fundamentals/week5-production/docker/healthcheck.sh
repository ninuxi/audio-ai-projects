#!/bin/bash
# Production Audio System Health Check Script
# Comprehensive health verification for containerized deployment

set -euo pipefail

# Configuration
API_HOST="${HOST:-localhost}"
API_PORT="${PORT:-8000}"
HEALTH_ENDPOINT="/health"
METRICS_ENDPOINT="/metrics"
TIMEOUT=10
RETRY_COUNT=3
RETRY_DELAY=2

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1" >&2
}

# Check if curl is available
check_curl() {
    if ! command -v curl &> /dev/null; then
        log_error "curl is not available"
        return 1
    fi
    return 0
}

# Check API health endpoint
check_health_endpoint() {
    local url="http://${API_HOST}:${API_PORT}${HEALTH_ENDPOINT}"
    local retry=0
    
    while [ $retry -lt $RETRY_COUNT ]; do
        log_info "Checking health endpoint (attempt $((retry + 1))/$RETRY_COUNT): $url"
        
        if response=$(curl -sf --max-time $TIMEOUT "$url" 2>/dev/null); then
            # Parse JSON response
            if echo "$response" | grep -q '"status"'; then
                status=$(echo "$response" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
                health_score=$(echo "$response" | grep -o '"health_score":[0-9.]*' | cut -d':' -f2)
                
                log_info "Health status: $status"
                log_info "Health score: $health_score"
                
                if [ "$status" = "healthy" ] || [ "$status" = "degraded" ]; then
                    return 0
                else
                    log_warn "System status is: $status"
                fi
            else
                log_error "Invalid health response format"
            fi
        else
            log_error "Health endpoint check failed"
        fi
        
        retry=$((retry + 1))
        if [ $retry -lt $RETRY_COUNT ]; then
            log_info "Retrying in ${RETRY_DELAY}s..."
            sleep $RETRY_DELAY
        fi
    done
    
    return 1
}

# Check metrics endpoint
check_metrics_endpoint() {
    local url="http://${API_HOST}:${API_PORT}${METRICS_ENDPOINT}"
    
    log_info "Checking metrics endpoint: $url"
    
    if curl -sf --max-time $TIMEOUT "$url" > /dev/null 2>&1; then
        log_info "Metrics endpoint is accessible"
        return 0
    else
        log_warn "Metrics endpoint is not accessible"
        return 1
    fi
}

# Check process health
check_process_health() {
    log_info "Checking process health..."
    
    # Check if main process is running
    if pgrep -f "production_audio_system" > /dev/null; then
        log_info "Main application process is running"
    else
        log_error "Main application process is not running"
        return 1
    fi
    
    # Check memory usage
    if command -v ps &> /dev/null; then
        memory_usage=$(ps -o pid,ppid,%mem,cmd -C python | grep production_audio_system | awk '{sum+=$3} END {print sum}')
        if [ -n "$memory_usage" ]; then
            log_info "Memory usage: ${memory_usage}%"
            
            # Alert if memory usage is too high
            if (( $(echo "$memory_usage > 80" | bc -l) )); then
                log_warn "High memory usage detected: ${memory_usage}%"
            fi
        fi
    fi
    
    return 0
}

# Check file system health
check_filesystem_health() {
    log_info "Checking filesystem health..."
    
    # Check required directories
    required_dirs=("/app/data" "/app/logs" "/app/uploads")
    
    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            # Check if directory is writable
            if [ -w "$dir" ]; then
                log_info "Directory $dir is accessible and writable"
            else
                log_warn "Directory $dir is not writable"
            fi
        else
            log_error "Required directory $dir does not exist"
            return 1
        fi
    done
    
    # Check disk space
    if command -v df &> /dev/null; then
        disk_usage=$(df /app | tail -1 | awk '{print $5}' | sed 's/%//')
        log_info "Disk usage: ${disk_usage}%"
        
        if [ "$disk_usage" -gt 90 ]; then
            log_error "Critical disk usage: ${disk_usage}%"
            return 1
        elif [ "$disk_usage" -gt 80 ]; then
            log_warn "High disk usage: ${disk_usage}%"
        fi
    fi
    
    return 0
}

# Check database connectivity
check_database_connectivity() {
    if [ -n "${DB_HOST:-}" ] && [ -n "${DB_PORT:-}" ]; then
        log_info "Checking database connectivity..."
        
        # Check if database port is accessible
        if command -v nc &> /dev/null; then
            if nc -z "${DB_HOST}" "${DB_PORT}" 2>/dev/null; then
                log_info "Database is accessible at ${DB_HOST}:${DB_PORT}"
                return 0
            else
                log_error "Database is not accessible at ${DB_HOST}:${DB_PORT}"
                return 1
            fi
        else
            log_warn "netcat not available, skipping database connectivity check"
        fi
    else
        log_info "Database configuration not found, skipping connectivity check"
    fi
    
    return 0
}

# Check Redis connectivity
check_redis_connectivity() {
    if [ -n "${REDIS_HOST:-}" ] && [ -n "${REDIS_PORT:-}" ]; then
        log_info "Checking Redis connectivity..."
        
        # Check if Redis port is accessible
        if command -v nc &> /dev/null; then
            if nc -z "${REDIS_HOST}" "${REDIS_PORT}" 2>/dev/null; then
                log_info "Redis is accessible at ${REDIS_HOST}:${REDIS_PORT}"
                return 0
            else
                log_error "Redis is not accessible at ${REDIS_HOST}:${REDIS_PORT}"
                return 1
            fi
        else
            log_warn "netcat not available, skipping Redis connectivity check"
        fi
    else
        log_info "Redis configuration not found, skipping connectivity check"
    fi
    
    return 0
}

# Check MAXXI integration
check_maxxi_integration() {
    if [ "${MAXXI_COMPATIBILITY_MODE:-false}" = "true" ]; then
        log_info "Checking MAXXI integration..."
        
        # Check MAXXI output directory
        maxxi_output_dir="${MAXXI_OUTPUT_DIR:-/app/data/maxxi_outputs}"
        if [ -d "$maxxi_output_dir" ] && [ -w "$maxxi_output_dir" ]; then
            log_info "MAXXI output directory is accessible: $maxxi_output_dir"
        else
            log_warn "MAXXI output directory is not accessible: $maxxi_output_dir"
        fi
        
        # Check if MAXXI integration script exists
        if [ -f "/app/maxxi-compatibility.py" ]; then
            log_info "MAXXI compatibility layer is present"
        else
            log_warn "MAXXI compatibility layer is missing"
        fi
        
        # Check if original MAXXI system is accessible
        if [ -f "/app/maxxi-installationmaxxi_audio_system.py" ]; then
            log_info "Original MAXXI system file is present"
        else
            log_warn "Original MAXXI system file is missing"
        fi
    else
        log_info "MAXXI integration is disabled"
    fi
    
    return 0
}

# Main health check function
main() {
    local exit_code=0
    
    log_info "Starting comprehensive health check..."
    log_info "Environment: ${ENVIRONMENT:-unknown}"
    log_info "Host: ${API_HOST}:${API_PORT}"
    
    # Check curl availability
    if ! check_curl; then
        exit_code=1
    fi
    
    # Core health checks
    if ! check_health_endpoint; then
        log_error "Health endpoint check failed"
        exit_code=1
    fi
    
    if ! check_metrics_endpoint; then
        log_warn "Metrics endpoint check failed"
        # Don't fail on metrics endpoint - it's not critical
    fi
    
    if ! check_process_health; then
        log_error "Process health check failed"
        exit_code=1
    fi
    
    if ! check_filesystem_health; then
        log_error "Filesystem health check failed"
        exit_code=1
    fi
    
    # External service checks (non-critical)
    if ! check_database_connectivity; then
        log_warn "Database connectivity check failed"
        # Don't fail on database connectivity - might be temporary
    fi
    
    if ! check_redis_connectivity; then
        log_warn "Redis connectivity check failed"
        # Don't fail on Redis connectivity - might be temporary
    fi
    
    # MAXXI integration check
    check_maxxi_integration
    
    # Summary
    if [ $exit_code -eq 0 ]; then
        log_info "Health check completed successfully"
    else
        log_error "Health check failed with errors"
    fi
    
    return $exit_code
}

# Signal handling
trap 'log_error "Health check interrupted"; exit 1' INT TERM

# Run main function
main "$@"
